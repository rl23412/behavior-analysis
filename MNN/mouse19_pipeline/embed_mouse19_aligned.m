clear; close all; clc;

fprintf('Adding SocialMapper-main to MATLAB path...\n');
thisFile = mfilename('fullpath');
if isempty(thisFile)
    scriptDir = pwd;
else
    scriptDir = fileparts(thisFile);
end

projectRoot = fileparts(scriptDir);
candidatePaths = {
    fullfile(projectRoot, 'SocialMapper-main'), ...
    fullfile(fileparts(projectRoot), 'SocialMapper-main')
};
socialMapperPath = '';
for cIdx = 1:numel(candidatePaths)
    if exist(candidatePaths{cIdx}, 'dir')
        socialMapperPath = candidatePaths{cIdx};
        addpath(genpath(socialMapperPath));
        fprintf('Successfully added SocialMapper-main from %s\n', socialMapperPath);
        break;
    end
end

if isempty(socialMapperPath)
    warning('SocialMapper-main directory not found in expected locations. Please ensure it exists.');
end
fprintf('\n');

%% Aligned Embedding Pipeline for Training + Re-embedding
% This script aligns every video to a common body plane inferred from the four
% paws (front/rear, left/right), trains the behavioral embedding on the aligned
% poses, and then re-embeds all files using the learned manifold.
%
% Key Features:
% 1. Computes a single rigid transform per video from paw geometry (no
%    per-frame wobble) so mice share a consistent floor-aligned coordinate
%    system.
% 2. Trains PCA, wavelets, t-SNE, and watershed on the aligned data to produce
%    an updated manifold- while preserving the MNN-corrected workflow.
% 3. Re-embeds every file, exports analysis CSV/plots to dedicated directories,
%    and writes updated MATLAB artifacts for downstream tooling.

fprintf('=== ALIGN + TRAIN + RE-EMBED PIPELINE (PAW-PLANE ALIGNED) ===\n');
fprintf('Training files: all files listed in all_files (aligned)\n');
fprintf('Re-embedding: same set as training (aligned)\n\n');

%% SECTION 1: SETUP AND CONFIGURATION
fprintf('[SECTION 1] Setup and Configuration\n');

% Define data directory
data_dir = '/work/rl349/dannce_predictions_train200/mouse19';

% Define all available files
all_files = {
    'cpfull/save_data_AVG_cpfull_vid1.mat', ...
    'cpfull/save_data_AVG_cpfull_vid2.mat', ...
    'cpfull/save_data_AVG_cpfull_vid3.mat', ...
    'cpfull/save_data_AVG_cpfull_vid4.mat', ...
    'cpfull/save_data_AVG_cpfull_vid5.mat', ...
    'wt/save_data_AVG_wt_vid1.mat', ...
    'wt/save_data_AVG_wt_vid2.mat', ...
    'wt/save_data_AVG_wt_vid3.mat', ...
    'wt/save_data_AVG_wt_vid4.mat', ...
    'wt/save_data_AVG_wt_vid5.mat'
};

% Define training files (use all files for robust embedding)
training_files = all_files;

% Define re-embedding files (re-embed all files to produce outputs for every dataset)
reembedding_files = all_files;

fprintf('Total files: %d\n', length(all_files));
fprintf('Training files: %d (%s)\n', length(training_files), strjoin(training_files, ', '));
fprintf('Re-embedding files: %d\n', length(reembedding_files));

% Create metadata for file organization
metadata = create_file_metadata(all_files);

% Joint groupings for paw-plane alignment
paws_idx = [10 13 16 19];
front_idx = [10 13];
hind_idx = [16 19];
left_idx = [10 16];
right_idx = [13 19];

alignmentIndices = struct('paws', paws_idx, ...
    'front', front_idx, 'hind', hind_idx, ...
    'left', left_idx, 'right', right_idx);

%% SECTION 2: LOAD AND PREPARE TRAINING DATA
fprintf('\n[SECTION 2] Loading Training Data\n');

% Load training data
training_data = {};
training_labels = {};
training_preprocess_info = cell(length(training_files), 1);
training_alignment_info = cell(length(training_files), 1);
total_training_frames = 0;

for i = 1:length(training_files)
    file_path = fullfile(data_dir, training_files{i});
    fprintf('Loading training file: %s\n', training_files{i});
    
    if exist(file_path, 'file')
        data = load(file_path);
        if isfield(data, 'pred')
            raw_pred = data.pred;
            [pred_data, preprocessInfo] = preprocess_pose_data(raw_pred, training_files{i});
            [aligned_data, alignInfo] = align_pose_sequence(pred_data, alignmentIndices);
            training_data{i} = aligned_data;
            training_labels{i} = training_files{i};
            training_preprocess_info{i} = preprocessInfo;
            training_alignment_info{i} = alignInfo;
            total_training_frames = total_training_frames + size(aligned_data, 1);
            fprintf('  Loaded + aligned: %s with %d frames (format %s, orientation %s, score %.3f)\n', ...
                training_files{i}, size(aligned_data, 1), preprocessInfo.formatSummary, ...
                preprocessInfo.orientationLabel, preprocessInfo.orientationScore);
            fprintf('    Alignment: forward=[%.3f %.3f %.3f], normal=[%.3f %.3f %.3f]\n', ...
                alignInfo.forwardAxis(:), alignInfo.normalAxis(:));
        else
            error('File %s does not contain pred field', training_files{i});
        end
    else
        error('Training file not found: %s', file_path);
    end
end

fprintf('Total training frames: %d\n', total_training_frames);

%% SECTION 3: PREPARE TRAINING SET (no flipping)
fprintf('\n[SECTION 3] Preparing Training Set (no flipping)\n');
training_data_combined = training_data;
training_labels_combined = training_labels;
fprintf('Training datasets: %d\n', length(training_data_combined));

%% SECTION 4: SET UP SKELETON AND PARAMETERS
fprintf('\n[SECTION 4] Setting Up Skeleton and Parameters\n');

% Skeleton for mouse19 format (19 joints)
joints_idx = [
    1 2; 1 3; 2 3; ...            % head connections
    1 4; 4 5; 5 6; 6 7; ...       % spine and tail
    5 8; 8 9; 9 10; ...           % left front limb
    5 11; 11 12; 12 13; ...       % right front limb
    6 14; 14 15; 15 16; ...       % left hind limb
    6 17; 17 18; 18 19];          % right hind limb

% Define colors for visualization
chead = [1 .6 .2];      % orange
cspine = [.2 .635 .172];% green
cLF = [0 0 1];          % blue
cRF = [1 0 0];          % red
cLH = [0 1 1];          % cyan
cRH = [1 0 1];          % magenta

scM = [
    chead; chead; chead; ...       % head connections (3)
    cspine; cspine; cspine; cspine; ... % spine/tail (4)
    cLF; cLF; cLF; ...             % left front limb (3)
    cRF; cRF; cRF; ...             % right front limb (3)
    cLH; cLH; cLH; ...             % left hind limb (3)
    cRH; cRH; cRH];                % right hind limb (3)

skeleton.color = scM;
skeleton.joints_idx = joints_idx;

% Set up parameters
parameters = setRunParameters([]);
parameters.samplingFreq = 50;
parameters.minF = 0.5;
parameters.maxF = 20;
parameters.numModes = 20;

% PCA parameters
nPCA = 15;
pcaModes = 20;
numModes = pcaModes;
featureDescriptor = 'MNN-corrected (mouse19 aligned)';

fprintf('Skeleton configured with %d joints and %d connections\n', 14, size(joints_idx, 1));
fprintf('Parameters: samplingFreq=%d, minF=%.1f, maxF=%.1f, PCA modes=%d\n', ...
    parameters.samplingFreq, parameters.minF, parameters.maxF, nPCA);

%% SECTION 5: EXTRACT FEATURES AND PERFORM PCA ON TRAINING DATA
fprintf('\n[SECTION 5] Feature Extraction and PCA on Training Data\n');

% Helper functions
returnDist3d = @(x,y) sqrt(sum((x-y).^2,2));

% Define joint pairs for distance calculations
xIdx = 1:19; yIdx = 1:19;
[Xi, Yi] = meshgrid(xIdx, yIdx);
Xi = Xi(:); Yi = Yi(:);
IDX = find(Xi ~= Yi);
nx = length(xIdx);

% Calculate characteristic length for each training dataset
lengtht = zeros(length(training_data_combined), 1);
for i = 1:length(training_data_combined)
    ma1 = training_data_combined{i};
    % Distance between keypoints 1 and 6 (snout to tail base)  
    sj = returnDist3d(squeeze(ma1(:,:,1)), squeeze(ma1(:,:,6)));  
    % Use 95th percentile as the characteristic length  
    lengtht(i) = prctile(sj, 95);
    fprintf('Characteristic length for %s: %.2f\n', training_labels_combined{i}, lengtht(i));
end

% PCA on training data only
fprintf('Performing PCA on training data...\n');
firstBatch = true;
currentImage = 0;
batchSize = 30000;
mu = zeros(1, 506);

for j = 1:length(training_data_combined)
    fprintf('Processing training dataset %d/%d (%s)\n', j, length(training_data_combined), training_labels_combined{j});
    ma1 = training_data_combined{j};
    nn1 = size(ma1,1);
    
    % Calculate pairwise distances
    p1Dist = zeros(nx^2,size(ma1,1));
    for i = 1:size(p1Dist,1)
        p1Dist(i,:) = returnDist3d(squeeze(ma1(:,:,Xi(i))),squeeze(ma1(:,:,Yi(i))));
    end
    
    % Smooth distances
    p1Dsmooth = zeros(size(p1Dist));
    for i = 1:size(p1Dist,1)
        if exist('medfilt1', 'file')
            p1Dsmooth(i,:) = smooth(medfilt1(p1Dist(i,:),3),3);
        else
            p1Dsmooth(i,:) = smooth(p1Dist(i,:),3);
        end
    end
    
    p1Dist = p1Dsmooth(IDX,:)';
    
    % Scale by characteristic length
    scaleVal = lengtht(j)./90;
    p1Dist = p1Dist.*scaleVal;
    
    % PCA computation
    if firstBatch
        firstBatch = false;
        if size(p1Dist,1) < batchSize
            cBatchSize = size(p1Dist,1);
            X = p1Dist;
        else
            cBatchSize = batchSize;
            X = p1Dist;
        end
        currentImage = cBatchSize;
        mu = sum(X);
        C = cov(X).*cBatchSize + (mu'*mu)./ cBatchSize;
    else
        if size(p1Dist,1) < batchSize
            cBatchSize = size(p1Dist,1);
            X = p1Dist;
        else
            cBatchSize = batchSize;
            X = p1Dist(randperm(size(p1Dist,1),cBatchSize),:);
        end
        tempMu = sum(X);
        mu = mu + tempMu;
        C = C + cov(X).*cBatchSize + (tempMu'*tempMu)./cBatchSize;
        currentImage = currentImage + cBatchSize;
    end
end

L = currentImage; mu = mu ./ L; C = C ./ L - mu'*mu;
fprintf('Computing Principal Components...\n');
[vecs,vals] = eig(C); vals = flipud(diag(vals)); vecs = fliplr(vecs);
mus = mu;

% Save PCA results
save('vecsMus_mouse19_aligned_training.mat','C','L','mus','vals','vecs');
fprintf('PCA complete. Saved to vecsMus_mouse19_aligned_training.mat\n');

%% SECTION 6: CREATE TRAINING EMBEDDING
fprintf('\n[SECTION 6] Creating Training Embedding\n');

vecs15 = vecs(:,1:nPCA);
numPerDataSet = 320;  % Standard subsampling

% Collect raw high-D samples per file (no per-file t-SNE/templates)
mD_training_samples = cell(size(training_data_combined)); 
mA_training_samples = cell(size(training_data_combined));

fprintf('Collecting behavioral feature samples for training (pre-MNN correction)...\n');
for j = 1:length(training_data_combined)
    fprintf('Processing features %d/%d (%s)\n', j, length(training_data_combined), training_labels_combined{j});
    ma1 = training_data_combined{j};

    nn1 = size(ma1,1);
    p1Dist = zeros(nx^2,size(ma1,1));
    for i = 1:size(p1Dist,1)
        p1Dist(i,:) = returnDist3d(squeeze(ma1(:,:,Xi(i))),squeeze(ma1(:,:,Yi(i))));
    end

    p1Dsmooth = zeros(size(p1Dist));
    for i = 1:size(p1Dist,1)
        if exist('medfilt1', 'file')
            p1Dsmooth(i,:) = smooth(medfilt1(p1Dist(i,:),3),3);
        else
            p1Dsmooth(i,:) = smooth(p1Dist(i,:),3);
        end
    end
    p1Dist = p1Dsmooth(IDX,:)';
    
    % Get floor value and scale
    allz = squeeze(ma1(:,3,[12 14])); 
    zz = allz(:);
    fz = prctile(zz,10);
    sj = returnDist3d(squeeze(ma1(:,:,1)),squeeze(ma1(:,:,6)));
    lz = prctile(sj,95);
    scaleVal = 90./lz;
    p1Dist = p1Dist.*scaleVal;
    
    p1 = bsxfun(@minus,p1Dist,mus);
    proj = p1*vecs15;
   
    [data,~] = findWavelets(proj,numModes,parameters);
    
    n = size(p1Dist,1);
    amps = sum(data,2);
    data2 = log(data);
    data2(data2<-5) = -5;
    
    % Joint velocities
    jv = zeros(n,length(xIdx));
    for i = 1:length(xIdx)
        if exist('medfilt1', 'file')
            jv(:,i) = [0; medfilt1(sqrt(sum(diff(squeeze(ma1(:,:,xIdx(i)))).^2,2)),10)];
        else
            jv(:,i) = [0; sqrt(sum(diff(squeeze(ma1(:,:,xIdx(i)))).^2,2))];
        end
    end
    jv = jv.*scaleVal;
    jv(jv>=5) = 5;
    
    % Z-coordinates
    p1z = zeros(nx,nn1);
    for i = 1:nx
        if exist('medfilt1', 'file')
            p1z(i,:) = smooth(medfilt1(squeeze(ma1(:,3,xIdx(i))),3),3);
        else
            p1z(i,:) = smooth(squeeze(ma1(:,3,xIdx(i))),3);
        end
    end
    allz1 = squeeze(ma1(:,3,[12 14])); 
    zz1 = allz1(:); 
    fz1 = prctile(zz1,10);
    floorval = fz1;
    p1z = (p1z-floorval).*scaleVal; 
    
    nnData = [data2 .25*p1z' .5*jv];
    
    % Subsample for training templates (store for global MNN correction)
    sampleIdx = 1:20:size(nnData,1);
    mD_training_samples{j} = nnData(sampleIdx,:);
    mA_training_samples{j} = amps(sampleIdx,:);
end

% === MNN BATCH CORRECTION (replaces ComBat) ================================
fprintf('Applying MNN batch correction on training samples...\n');

% Concatenate the training samples you already collected
allD_train_raw = combineCells(mD_training_samples);  % [N x F]
allA_train     = combineCells(mA_training_samples);  % [N x 1]

% Build batch labels using your existing logic (one batch per file)
allBatchLabels_points = {};
for j = 1:length(mD_training_samples)
    nRows = size(mD_training_samples{j},1);
    batchLabelJ = infer_batch_label(training_labels_combined{j});
    allBatchLabels_points = [allBatchLabels_points; repmat({batchLabelJ}, nRows, 1)];
end
[~,~,batchVector] = unique(allBatchLabels_points);   % integer batch ids

% Remove NaNs/Infs per feature to be safe
X = allD_train_raw;
if any(~isfinite(X(:)))
    featMed = nanmedian(X,1);
    for f = 1:size(X,2)
        bad = ~isfinite(X(:,f));
        if any(bad)
            X(bad,f) = featMed(f);
        end
    end
end

% Fit MNN on training samples (returns corrected training features + a model)
mnnOpts = struct('k', 20, ...                        % MNN neighbors
                 'ndim', min(50, size(X,2)), ...     % PCs for neighbor search
                 'sigma', [] , ...                   % auto bandwidth if []
                 'distance','euclidean', ...
                 'verbose', true);
[mnnModel, allD_train_corrected] = fit_mnn_correction(X, batchVector, mnnOpts);

% Persist the model so future re-embeds can use the same reference
save('mnn_model_mouse19_aligned.mat','mnnModel','-v7.3');
fprintf('MNN correction complete. Model saved to mnn_model_mouse19_aligned.mat\n');

% Proceed exactly as before, but using MNN-corrected features
fprintf('Running t-SNE on MNN-corrected training samples (%d points) ...\n', size(allD_train_corrected,1));
yData = tsne(allD_train_corrected);

% Global template selection on corrected samples (unchanged)
[signalData,signalAmps] = findTemplatesFromData( allD_train_corrected, yData, allA_train, numPerDataSet, parameters);

% Use selected templates as training set
mD_training = {signalData};
mA_training = {signalAmps};
save('trainingSignalData_mouse19_aligned.mat','mA_training','mD_training');
fprintf('Training embeddings saved to trainingSignalData_mouse19_aligned.mat\n');
% ==========================================================================

%% SECTION 7: CREATE t-SNE EMBEDDING ON TRAINING DATA
fprintf('\n[SECTION 7] Creating t-SNE Embedding on Training Data\n');

% Combine training embeddings (already global)
allD_training = signalData; 
allA_training = signalAmps;

fprintf('Running t-SNE on training data (%d points)...\n', size(allD_training, 1));
Y_training = tsne(allD_training);
save('train_mouse19_aligned.mat','Y_training','allD_training');

fprintf('Training t-SNE complete. Saved to train_mouse19_aligned.mat\n');

%% SECTION 8: CREATE WATERSHED REGIONS FROM TRAINING DATA
fprintf('\n[SECTION 8] Creating Watershed Regions from Training Data\n');

% Dynamic symmetric bounds around zero with small padding
yMin = min(Y_training, [], 1); yMax = max(Y_training, [], 1);
maxAbs = max([abs(yMin(1)), abs(yMax(1)), abs(yMin(2)), abs(yMax(2))]);
maxAbs = maxAbs * 1.05;  % 5%% padding
sigma_density = 0.8;     % slightly smaller sigma to increase region granularity

[xx, d] = findPointDensity(Y_training, sigma_density, 501, [-maxAbs maxAbs]);
D = d;

% Limit map extent to regions with meaningful training support
[Xgrid, Ygrid] = meshgrid(xx, xx);
trainRadius = sqrt(sum(Y_training.^2, 2));
if isempty(trainRadius)
    supportRadius = maxAbs;
else
    supportRadius = prctile(trainRadius, 99.5);
    if ~isfinite(supportRadius) || supportRadius <= 0
        supportRadius = max(trainRadius);
    end
    supportRadius = supportRadius * 1.05; % small safety margin
end
radialMask = sqrt(Xgrid.^2 + Ygrid.^2) > supportRadius;
D(radialMask) = 0;
d(radialMask) = 0;

positiveVals = D(D>0);
if isempty(positiveVals)
    densityFloor = 0;
else
    densityFloor = prctile(positiveVals, 2);
end
lowDensityMask = D < densityFloor;
D(lowDensityMask) = 0;
d(lowDensityMask) = 0;

% Watershed
LL = watershed(-d,18);
LL2 = LL; 
LL2(d < 1e-6) = -1;
LL2(radialMask | lowDensityMask) = -1;

% Smooth boundary extraction for cleaner visualization
validRegionIds = unique(LL2(LL2>0));
boundaryPolys = cell(numel(validRegionIds),1);
boundaryMask = false(size(LL2));
for idx = 1:numel(validRegionIds)
    regionMask = (LL2 == validRegionIds(idx));
    if ~any(regionMask(:))
        continue;
    end
    filledMask = imfill(regionMask, 'holes');
    boundaries = bwboundaries(filledMask, 8);
    if isempty(boundaries)
        continue;
    end
    poly = boundaries{1};
    if size(poly,1) >= 5
        poly(:,1) = smoothdata(poly(:,1), 'movmean', 5);
        poly(:,2) = smoothdata(poly(:,2), 'movmean', 5);
    end
    poly(:,1) = min(max(poly(:,1), 1), size(LL2,1));
    poly(:,2) = min(max(poly(:,2), 1), size(LL2,2));
    boundaryPolys{idx} = poly;
    boundaryMask = boundaryMask | (poly2mask(poly(:,2), poly(:,1), size(LL2,1), size(LL2,2)) & filledMask);
end

if any(boundaryMask(:))
    [boundaryRows, boundaryCols] = find(boundaryMask);
    llbwb = [boundaryRows, boundaryCols];
else
    llbwb = zeros(0,2);
end

% Plot density map
figure('Name', 'Training Data Behavioral Density Map');
imagesc(D); 
axis equal off; 
colormap(flipud(gray)); 
caxis([0 max(D(:))*0.8]);
hold on;
if ~isempty(boundaryPolys)
    for idx = 1:numel(boundaryPolys)
        poly = boundaryPolys{idx};
        if isempty(poly)
            continue;
        end
        plot(poly(:,2), poly(:,1), 'Color', [0 0 0], 'LineWidth', 1.5);
    end
else
    scatter(llbwb(:,2),llbwb(:,1),'.','k');
end
title('Training Data Behavioral Density Map (MNN-corrected, dynamic bounds)');

save('watershed_mouse19_aligned.mat', 'D', 'LL', 'LL2', 'llbwb', 'boundaryPolys', 'xx');
fprintf('Watershed regions saved to watershed_mouse19_aligned.mat\n');

%% SECTION 9: LOAD AND RE-EMBED ALL OTHER FILES
fprintf('\n[SECTION 9] Re-embedding All Other Files\n');

% Load the trained embedding space
load('train_mouse19_aligned.mat','Y_training','allD_training');
trainingSetData = allD_training; 
trainingEmbeddingZ = Y_training;

% Load all re-embedding files
reembedding_data = {};
reembedding_labels = {};
reembedding_metadata = {};
reembedding_preprocess_info = cell(length(reembedding_files), 1);
reembedding_alignment_info = cell(length(reembedding_files), 1);

fprintf('Loading %d files for re-embedding...\n', length(reembedding_files));
for i = 1:length(reembedding_files)
    file_path = fullfile(data_dir, reembedding_files{i});
    fprintf('Loading: %s\n', reembedding_files{i});
    
    if exist(file_path, 'file')
        data = load(file_path);
        if isfield(data, 'pred')
            raw_pred = data.pred;
            [pred_data, preprocessInfo] = preprocess_pose_data(raw_pred, reembedding_files{i});
            [aligned_data, alignInfo] = align_pose_sequence(pred_data, alignmentIndices);
            reembedding_data{i} = aligned_data;
            reembedding_labels{i} = reembedding_files{i};
            reembedding_preprocess_info{i} = preprocessInfo;
            reembedding_alignment_info{i} = alignInfo;
            
            % Extract metadata
            [group, week] = extract_metadata_from_filename(reembedding_files{i});
            reembedding_metadata{i} = struct('group', group, 'week', week);
            
            fprintf('  Loaded + aligned: %s (%s, %s) with %d frames (format %s, orientation %s, score %.3f)\n', ...
                reembedding_files{i}, group, week, size(aligned_data, 1), ...
                preprocessInfo.formatSummary, preprocessInfo.orientationLabel, ...
                preprocessInfo.orientationScore);
            fprintf('    Alignment: forward=[%.3f %.3f %.3f], normal=[%.3f %.3f %.3f]\n', ...
                alignInfo.forwardAxis(:), alignInfo.normalAxis(:));
        else
            warning('File %s does not contain pred field', reembedding_files{i});
        end
    else
        warning('File not found: %s', file_path);
    end
end

% Remove empty entries
valid_idx = ~cellfun(@isempty, reembedding_data);
reembedding_data = reembedding_data(valid_idx);
reembedding_labels = reembedding_labels(valid_idx);
reembedding_metadata = reembedding_metadata(valid_idx);
reembedding_preprocess_info = reembedding_preprocess_info(valid_idx);
reembedding_alignment_info = reembedding_alignment_info(valid_idx);

fprintf('Successfully loaded %d files for re-embedding\n', length(reembedding_data));

%% SECTION 10: CREATE FLIPPED VERSIONS FOR RE-EMBEDDING DATA
% Disabled (no flipping requested). Proceed with original data only.
% (Intentionally left as a no-op to avoid jointMapping errors.)

% Prepare re-embedding dataset list (already set to all_files above)

%% SECTION 11: RE-EMBED ALL FILES ONTO TRAINED SPACE
fprintf('\n[SECTION 11] Re-embedding All Files onto Trained Space\n');

zEmbeddings_all = cell(length(reembedding_data), 1);
wrFINE_all = cell(length(reembedding_data), 1);
parameters.batchSize = 10000;

fprintf('Re-embedding %d datasets...\n', length(reembedding_data));

% Load saved MNN model once so each dataset can be corrected before projection
if exist('mnn_model_mouse19_aligned.mat','file')
    tmp = load('mnn_model_mouse19_aligned.mat','mnnModel');
    mnnModel = tmp.mnnModel;
else
    error('MNN model file mnn_model_mouse19_aligned.mat not found. Run training section first.');
end

for j = 1:length(reembedding_data)
    if mod(j, 5) == 0
        fprintf('Re-embedding dataset %d/%d (%s)\n', j, length(reembedding_data), reembedding_labels{j});
    end
    
    ma1 = reembedding_data{j};
    
    % Extract features using same pipeline as training
    nn1 = size(ma1,1);
    p1Dist = zeros(nx^2,size(ma1,1));
    for i = 1:size(p1Dist,1)
        p1Dist(i,:) = returnDist3d(squeeze(ma1(:,:,Xi(i))),squeeze(ma1(:,:,Yi(i))));
    end

    p1Dsmooth = zeros(size(p1Dist));
    for i = 1:size(p1Dist,1)
        if exist('medfilt1', 'file')
            p1Dsmooth(i,:) = smooth(medfilt1(p1Dist(i,:),3),3);
        else
            p1Dsmooth(i,:) = smooth(p1Dist(i,:),3);
        end
    end
    p1Dist = p1Dsmooth(IDX,:)';
    
    allz = squeeze(ma1(:,3,[12 14])); 
    zz = allz(:);
    fz = prctile(zz,10);
    sj = returnDist3d(squeeze(ma1(:,:,1)),squeeze(ma1(:,:,6)));
    lz = prctile(sj,95);
    scaleVal = 90./lz;
    p1Dist = p1Dist.*scaleVal;

    p1 = bsxfun(@minus,p1Dist,mus);
    proj = p1*vecs15;

    [data,~] = findWavelets(proj,numModes,parameters);

    n = size(p1Dist,1);
    amps = sum(data,2);
    data2 = log(data);
    data2(data2<-5) = -5;
    
    jv = zeros(n,length(xIdx));
    for i = 1:length(xIdx)
        if exist('medfilt1', 'file')
            jv(:,i) = [0; medfilt1(sqrt(sum(diff(squeeze(ma1(:,:,xIdx(i)))).^2,2)),10)];
        else
            jv(:,i) = [0; sqrt(sum(diff(squeeze(ma1(:,:,xIdx(i)))).^2,2))];
        end
    end
    jv = jv.*scaleVal;
    jv(jv>=5) = 5;

    p1z = zeros(nx,nn1);
    for i = 1:nx
        if exist('medfilt1', 'file')
            p1z(i,:) = smooth(medfilt1(squeeze(ma1(:,3,xIdx(i))),3),3);
        else
            p1z(i,:) = smooth(squeeze(ma1(:,3,xIdx(i))),3);
        end
    end
    allz1 = squeeze(ma1(:,3,[12 14])); 
    zz1 = allz1(:); 
    fz1 = prctile(zz1,10);
    floorval = fz1;
    p1z = (p1z-floorval).*scaleVal;

    nnData = [data2 .25*p1z' .5*jv];

    % Correct new data into the shared training space using saved MNN model
    nnData_corr = apply_mnn_correction(nnData, mnnModel);

    % Find embeddings using trained space
    [zValues,zCosts,zGuesses,inConvHull,meanMax,exitFlags] = ...
        findTDistributedProjections_fmin(nnData_corr,trainingSetData,...
        trainingEmbeddingZ,[],parameters);

    z = zValues; 
    z(~inConvHull,:) = zGuesses(~inConvHull,:);
    
    % Save the embedding coordinates
    zEmbeddings_all{j} = z;
    
    % Find watershed regions
    vSmooth = .5;

    medianLength = 1;
    pThreshold = [];
    minRest = 5;
    obj = [];
    fitOnly = false;
    numGMM = 2;

    [wr,~,~,~,~,~,~,~] = findWatershedRegions_v2(z,xx,LL,vSmooth,...
        medianLength,pThreshold,minRest,obj,fitOnly,numGMM);
    
    wrFINE_all{j} = wr;
end

fprintf('Re-embedding complete for all %d datasets\n', length(reembedding_data));

%% SECTION 12: SAVE RESULTS AND CREATE ANALYSIS
fprintf('\n[SECTION 12] Saving Results and Creating Analysis\n');

% Save comprehensive results
results = struct();
results.training_files = training_files;
results.reembedding_files = reembedding_files;
results.reembedding_labels_all = reembedding_labels;
results.reembedding_metadata_all = reembedding_metadata;
results.zEmbeddings_all = zEmbeddings_all;
results.wrFINE_all = wrFINE_all;
results.Y_training = Y_training;
results.D = D;
results.LL = LL;
results.LL2 = LL2;
results.llbwb = llbwb;
results.parameters = parameters;
results.nPCA = nPCA;
results.skeleton = skeleton;
results.featureDescriptor = featureDescriptor;
results.training_preprocess_info = training_preprocess_info;
results.reembedding_preprocess_info = reembedding_preprocess_info;
results.training_alignment_info = training_alignment_info;
results.reembedding_alignment_info = reembedding_alignment_info;

save('complete_embedding_results_mouse19_aligned.mat', 'results', '-v7.3');
fprintf('Complete results saved to complete_embedding_results_mouse19_aligned.mat\n');

%% SECTION 13: CREATE VISUALIZATION AND ANALYSIS
fprintf('\n[SECTION 13] Creating Visualizations and Analysis\n');

% Define colors for each group
groupColors = struct();
groupColors.DRG = [1 0 0];         % Red
groupColors.SC = [0 0 1];          % Blue
groupColors.IT = [0 1 0];          % Green
groupColors.SNI = [1 0.5 0];       % Orange
groupColors.TBI = [0.5 0 1];       % Purple
groupColors.CTRL = [0.5 0.5 0.5];  % Gray (Control)
groupColors.week4_DRG = [0.7 0 0]; % Dark Red
groupColors.week4_SC = [0 0 0.7];  % Dark Blue
groupColors.week4_SNI = [0.7 0.3 0]; % Dark Orange
groupColors.week4_TBI = [0.3 0 0.7]; % Dark Purple

% Create group comparison plots
create_group_comparison_plots(results, groupColors);

% Create individual mouse plots
create_individual_mouse_plots(results, groupColors);

% Create temporal analysis plots
create_temporal_analysis_plots(results, groupColors);

fprintf('\nAnalysis complete! All visualizations created.\n');

%% DISPLAY SUMMARY
fprintf('\n=== PIPELINE SUMMARY ===\n');
fprintf('Training files used: %s\n', strjoin(training_files, ', '));
fprintf('Total files re-embedded: %d\n', length(reembedding_files));
fprintf('Training frames: %d\n', total_training_frames);
fprintf('PCA components: %d\n', nPCA);
fprintf('Wavelet modes: %d\n', numModes);
fprintf('Watershed regions: %d\n', max(LL(:)));

% Show group distribution
unique_groups = {};
group_counts = struct();
for i = 1:length(reembedding_metadata)
    if ~isempty(reembedding_metadata{i})
        group = reembedding_metadata{i}.group;
        if ~any(strcmp(unique_groups, group))
            unique_groups{end+1} = group;
            group_counts.(matlab.lang.makeValidName(group)) = 1;
        else
            field_name = matlab.lang.makeValidName(group);
            if isfield(group_counts, field_name)
                group_counts.(field_name) = group_counts.(field_name) + 1;
            else
                group_counts.(field_name) = 1;
            end
        end
    end
end

fprintf('\nGroup distribution in re-embedded data:\n');
for i = 1:length(unique_groups)
    field_name = matlab.lang.makeValidName(unique_groups{i});
    if isfield(group_counts, field_name)
        fprintf('  %s: %d files\n', unique_groups{i}, group_counts.(field_name));
    end
end

fprintf('\nResults saved to: complete_embedding_results_mouse19_aligned.mat\n');
fprintf('All visualizations have been created and saved.\n');

%% SECTION 14: CSV ANALYSIS USING ANALYZE SCRIPT PARAMETERS
fprintf('\n[SECTION 14] Generating CSV Analysis\n');

outDir = 'analysis_outputs_mouse19_aligned';
figDir = fullfile(outDir, 'figures');
perVideoDir = fullfile(figDir, 'per_video');
temporalDir = fullfile(figDir, 'temporal_progression');
csvDir = fullfile(outDir, 'csv');
if ~exist(outDir,'dir'), mkdir(outDir); end
if ~exist(figDir,'dir'), mkdir(figDir); end
if ~exist(perVideoDir,'dir'), mkdir(perVideoDir); end
if ~exist(temporalDir,'dir'), mkdir(temporalDir); end
if ~exist(csvDir,'dir'), mkdir(csvDir); end

load('watershed_mouse19_aligned.mat', 'D', 'LL', 'LL2', 'llbwb', 'boundaryPolys', 'xx');
if ~exist('boundaryPolys','var') || isempty(boundaryPolys)
    boundaryPolys = {};
end

gridSize = length(xx);
mapBounds = [min(xx), max(xx)];
displayRangeExpand = 1.0;
if displayRangeExpand ~= 1
    ctr = mean(mapBounds);
    halfR = (mapBounds(2) - mapBounds(1)) / 2;
    halfR = halfR * displayRangeExpand;
    mapBounds = [ctr - halfR, ctr + halfR];
end

mapZToImage = @(z) deal(...
    round((z(:,1) - mapBounds(1)) / (mapBounds(2) - mapBounds(1)) * (gridSize-1) + 1), ...
    round((z(:,2) - mapBounds(1)) / (mapBounds(2) - mapBounds(1)) * (gridSize-1) + 1));

validRegionIds = unique(LL2(LL2>0));
validRegionIds = validRegionIds(:)';
numRegions = numel(validRegionIds);

fprintf('Found %d valid watershed regions for CSV analysis\n', numRegions);
if ~exist('videoOptions','var') || ~isstruct(videoOptions)
    videoOptions = struct();
end
if ~isfield(videoOptions, 'maxWatershedRegions') || videoOptions.maxWatershedRegions <= 0
    videoOptions.maxWatershedRegions = numRegions;
else
    videoOptions.maxWatershedRegions = min(videoOptions.maxWatershedRegions, numRegions);
end

fig = figure('Name','Behavioral Map with Region Indices','Position',[100 100 900 900]);
imagesc(D); axis equal off; colormap(flipud(gray)); caxis([0 max(D(:))*0.8]); hold on;
if ~isempty(boundaryPolys)
    for idx = 1:numel(boundaryPolys)
        poly = boundaryPolys{idx};
        if isempty(poly)
            continue;
        end
        plot(poly(:,2), poly(:,1), 'Color', [0 0 0], 'LineWidth', 1.2);
    end
elseif ~isempty(llbwb)
    scatter(llbwb(:,2), llbwb(:,1), 1, 'k', '.');
end
for k = 1:numRegions
    oldId = validRegionIds(k);
    [yy,xxi] = find(LL2==oldId);
    if isempty(yy), continue; end
    cx = round(mean(xxi)); cy = round(mean(yy));
    text(cx, cy, sprintf('%d', k), 'Color','y','FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
end
if exist('featureDescriptor','var') && ~isempty(featureDescriptor)
    mapTitle = sprintf('Behavioral Map with Region Indices (%s features)', featureDescriptor);
else
    mapTitle = 'Behavioral Map with Region Indices';
end
title(mapTitle);
saveas(fig, fullfile(figDir, 'behavioral_map_with_indices.png'));
close(fig);

fileNames = results.reembedding_labels_all;
zAll = results.zEmbeddings_all;
numFiles = numel(fileNames);
counts = zeros(numFiles, numRegions);

baseGray = mat2gray(D);
baseRGB = ind2rgb(uint8(baseGray * 255), flipud(gray(256)));
overlayColormap = parula(256);

fprintf('Analyzing %d files for region counts...\n', numFiles);

for i = 1:numFiles
    z = zAll{i};
    if isempty(z) || size(z,2)~=2
        continue;
    end
    [xImg, yImg] = mapZToImage(z);
    valid = xImg>=1 & xImg<=gridSize & yImg>=1 & yImg<=gridSize;
    xImg = xImg(valid); yImg = yImg(valid);

    linIdx = sub2ind(size(LL2), yImg, xImg);
    regIdsOld = double(LL2(linIdx));
    validRegionMask = regIdsOld >= 1;
    regIdsOld = regIdsOld(validRegionMask);
    
    regIds = zeros(size(regIdsOld));
    if ~isempty(regIdsOld)
        [tf,loc] = ismember(regIdsOld, validRegionIds);
        regIds(tf) = loc(tf);
    end
    
    if ~isempty(regIds)
        edges = 0.5:1:(double(numRegions)+0.5);
        c = histcounts(regIds, edges);
        counts(i,:) = c;
    end

    if isempty(xImg)
        densityCounts = zeros(size(D));
    else
        linPix = sub2ind(size(D), yImg, xImg);
        densityCounts = accumarray(linPix, 1, [numel(D(:)) 1]);
        densityCounts = reshape(densityCounts, size(D));
    end
    if exist('imgaussfilt','file')
        densitySmooth = imgaussfilt(densityCounts, 2);
    else
        hKernel = fspecial('gaussian',[7 7],2);
        densitySmooth = imfilter(densityCounts, hKernel, 'replicate');
    end
    if max(densitySmooth(:)) > 0
        densityNorm = densitySmooth ./ max(densitySmooth(:));
    else
        densityNorm = densitySmooth;
    end

    figv = figure('Name', sprintf('Overlay %s', fileNames{i}), 'Position',[50 50 900 900]);
    image(baseRGB);
    axis image off;
    set(gca,'YDir','normal');
    hold on;
    alphaOverlay = min(1, densityNorm.^0.75 * 0.9);
    alphaOverlay(densityNorm < 1e-3) = 0;
    hOverlay = imagesc(densityNorm);
    set(hOverlay, 'AlphaData', alphaOverlay);
    colormap(gca, overlayColormap);
    caxis([0 1]);
    cb = colorbar('Location','eastoutside');
    cb.Label.String = 'Relative density';
    if ~isempty(boundaryPolys)
        for bIdx = 1:numel(boundaryPolys)
            poly = boundaryPolys{bIdx};
            if isempty(poly)
                continue;
            end
            plot(poly(:,2), poly(:,1), 'Color', [0 0 0], 'LineWidth', 1.0);
        end
    elseif ~isempty(llbwb)
        plot(llbwb(:,2), llbwb(:,1), '.', 'Color', [0 0 0], 'MarkerSize', 1);
    end
    title(strrep(fileNames{i},'_','\_'));
    sanitizedName = regexprep(fileNames{i}, '[^\w\-\.]+', '_');
    saveas(figv, fullfile(perVideoDir, sprintf('%03d_%s.png', i, sanitizedName)));
    close(figv);

    if length(xImg) > 100
        figT = figure('Name', sprintf('Temporal %s', fileNames{i}), 'Position',[50 50 900 900]);
        imagesc(D); axis equal off; colormap(flipud(gray)); caxis([0 max(D(:))*0.8]); hold on;
        if ~isempty(llbwb)
            scatter(llbwb(:,2), llbwb(:,1), 1, 'k', '.');
        end
        numFrames = length(xImg);
        colors = [linspace(0, 1, numFrames)', zeros(numFrames, 1), linspace(1, 0, numFrames)'];
        temporalShowN = min(10000, numFrames);
        if temporalShowN < numFrames
            temporalIdx = round(linspace(1, numFrames, temporalShowN));
            xTemporal = xImg(temporalIdx);
            yTemporal = yImg(temporalIdx);
            colorsTemporal = colors(temporalIdx, :);
        else
            xTemporal = xImg;
            yTemporal = yImg;
            colorsTemporal = colors;
        end
        for t = 1:length(xTemporal)
            scatter(xTemporal(t), yTemporal(t), 3, colorsTemporal(t, :), 'filled', 'MarkerEdgeColor', 'none');
        end
        title(sprintf('%s - Temporal Progression (Blue=Start, Red=End)', strrep(fileNames{i},'_','\_')));
        c = colorbar('Location', 'eastoutside');
        c.Label.String = 'Time Progression';
        c.Ticks = [0 1];
        c.TickLabels = {'Start', 'End'};
        saveas(figT, fullfile(temporalDir, sprintf('%03d_%s_temporal.png', i, sanitizedName)));
        close(figT);
    end
end

T = array2table(counts, 'VariableNames', compose('Region_%d', 1:numRegions));
T.File = fileNames(:);
T = movevars(T, 'File', 'Before', 1);
writetable(T, fullfile(csvDir, 'per_file_region_counts.csv'));

mapTable = table(validRegionIds', (1:numRegions)', 'VariableNames', {'OriginalLabel','ConsecutiveIndex'});
writetable(mapTable, fullfile(csvDir, 'region_label_mapping.csv'));

regionTotals = sum(counts, 1);
regionUsage = sum(counts > 0, 1);
totalFrames = sum(counts(:));

regionSummary = table();
regionSummary.RegionIndex = (1:numRegions)';
regionSummary.OriginalLabel = validRegionIds';
regionSummary.TotalFrames = regionTotals';
regionSummary.VideosUsing = regionUsage';
regionSummary.AvgFramesPerVideo = regionTotals' ./ max(1, regionUsage');
if totalFrames > 0
    regionSummary.PercentOfTotalFrames = 100 * regionTotals' / totalFrames;
else
    regionSummary.PercentOfTotalFrames = zeros(numRegions, 1);
end
regionSummary = sortrows(regionSummary, 'TotalFrames', 'descend');
writetable(regionSummary, fullfile(csvDir, 'region_summary.csv'));

transposedTable = table();
transposedTable.RegionIndex = (1:numRegions)';
transposedTable.OriginalLabel = validRegionIds';
for i = 1:numFiles
    videoName = regexprep(fileNames{i}, '[^\w\-\.]+', '_');
    videoName = strrep(videoName, '.mat', '');
    transposedTable.(videoName) = counts(i,:)';
end
writetable(transposedTable, fullfile(csvDir, 'region_counts_transposed.csv'));

fprintf('CSV Analysis complete!\n');
fprintf('Output directory: %s\n', csvDir);
fprintf('Files created:\n');
fprintf('  - per_file_region_counts.csv\n');
fprintf('  - region_counts_transposed.csv\n');
fprintf('  - region_summary.csv\n');
fprintf('  - region_label_mapping.csv\n');
fprintf('  - behavioral_map_with_indices.png\n');
fprintf('  - per_video/*.png\n');
fprintf('  - temporal_progression/*.png\n');
fprintf('Total behavioral frames analyzed: %d\n', totalFrames);
fprintf('Regions with data: %d/%d\n', sum(regionTotals > 0), numRegions);

%% SECTION 15: EXPORT FRAME INDICES PER VIDEO FOR PYTHON OVERLAY GENERATION
fprintf('\n[SECTION 15] Exporting Frame Indices Per Video for Python Overlay Generation\n');

frameIndicesDir = fullfile(csvDir, 'frame_indices_per_video');
if ~exist(frameIndicesDir,'dir'), mkdir(frameIndicesDir); end

fprintf('Exporting frame indices for %d videos to %s\n', numFiles, frameIndicesDir);

for i = 1:numFiles
    z = zAll{i};
    if isempty(z) || size(z,2)~=2
        continue;
    end
    [xImg, yImg] = mapZToImage(z);
    valid = xImg>=1 & xImg<=gridSize & yImg>=1 & yImg<=gridSize;
    xImg = xImg(valid); yImg = yImg(valid);
    if isempty(xImg)
        continue;
    end
    originalFrameIndices = find(valid);
    linIdx = sub2ind(size(LL2), yImg, xImg);
    regIdsOld = double(LL2(linIdx));
    validRegionMask = regIdsOld >= 1;
    validFrameIndices = originalFrameIndices(validRegionMask);
    validRegIds = regIdsOld(validRegionMask);

    regIds = zeros(size(validRegIds));
    [tf,loc] = ismember(validRegIds, validRegionIds);
    regIds(tf) = loc(tf);

    regionCounts = arrayfun(@(r) sum(regIds == r), 1:numRegions);
    maxFrames = max(regionCounts);
    if isempty(maxFrames) || maxFrames == 0
        maxFrames = 1;
    end

    regionFrameTable = table();
    for regionIdx = 1:numRegions
        regionFrames = validFrameIndices(regIds == regionIdx);
        paddedFrames = NaN(maxFrames, 1);
        if ~isempty(regionFrames)
            paddedFrames(1:length(regionFrames)) = regionFrames(:);
        end
        regionFrameTable.(sprintf('Region_%d', regionIdx)) = paddedFrames;
    end
    if isempty(regionFrameTable)
        for regionIdx = 1:numRegions
            regionFrameTable.(sprintf('Region_%d', regionIdx)) = NaN;
        end
    end
    numRows = height(regionFrameTable);
    for regionIdx = 1:numRegions
        colName = sprintf('Region_%d', regionIdx);
        currentCol = regionFrameTable.(colName);
        if length(currentCol) < numRows
            regionFrameTable.(colName) = [currentCol; NaN(numRows - length(currentCol), 1)];
        end
    end

    sanitizedFileName = regexprep(fileNames{i}, '[^\w\-\.]+', '_');
    outputFileName = sprintf('%03d_%s_frame_indices.csv', i, sanitizedFileName);
    outputPath = fullfile(frameIndicesDir, outputFileName);
    writetable(regionFrameTable, outputPath);

    fprintf('  Exported frame indices for %s: %d valid frames across %d regions\n', ...
        fileNames{i}, length(validFrameIndices), numRegions);
end

fprintf('Frame indices export complete!\n');
fprintf('Python script can now use: %s\n', frameIndicesDir);
fprintf('Files created: %d frame index CSV files\n', numFiles);

summaryData = table();
summaryData.VideoIndex = (1:numFiles)';
summaryData.VideoFileName = fileNames(:);
summaryData.TotalFrames = arrayfun(@(i) size(results.zEmbeddings_all{i}, 1), 1:numFiles)';
summaryData.ValidFrames = zeros(numFiles, 1);
summaryData.RegionsWithFrames = zeros(numFiles, 1);

for i = 1:numFiles
    z = zAll{i};
    if ~isempty(z) && size(z,2)==2
        [xImg, yImg] = mapZToImage(z);
        valid = xImg>=1 & xImg<=gridSize & yImg>=1 & yImg<=gridSize;
        summaryData.ValidFrames(i) = sum(valid);
        if sum(valid) > 0
            xImg = xImg(valid); yImg = yImg(valid);
            linIdx = sub2ind(size(LL2), yImg, xImg);
            regIdsOld = double(LL2(linIdx));
            validRegionMask = regIdsOld >= 1;
            if sum(validRegionMask) > 0
                validRegIds = regIdsOld(validRegionMask);
                summaryData.RegionsWithFrames(i) = length(unique(validRegIds));
            end
        end
    end
end

writetable(summaryData, fullfile(csvDir, 'video_summary_for_python.csv'));
fprintf('Created video summary file: video_summary_for_python.csv\n');

function [pose, info] = preprocess_pose_data(rawPred, fileLabel)
    if nargin < 2
        fileLabel = '';
    end
    info = struct();
    info.file = fileLabel;
    info.originalSize = size(rawPred);
    info.orientationLabel = 'none';
    info.orientationScore = NaN;
    info.formatSummary = '';

    [pose, formatInfo] = normalize_pose_format(rawPred);
    info.format = formatInfo;
    info.finalSize = size(pose);
    info.formatSummary = sprintf('%dx%dx%d', size(pose, 1), size(pose, 2), size(pose, 3));

    [pose, orientationInfo] = apply_orientation_heuristic(pose);
    info.orientation = orientationInfo;
    if orientationInfo.applied
        info.orientationLabel = orientationInfo.bestLabel;
    end
    info.orientationScore = orientationInfo.score;
end

function [pose, info] = normalize_pose_format(rawPred)
    info = struct();
    info.transform = 'none';
    info.notes = '';
    info.originalClass = class(rawPred);

    if isempty(rawPred)
        pose = zeros(0, 3, 0);
        info.transform = 'empty';
        info.outputSize = size(pose);
        return;
    end

    pose = squeeze(rawPred);
    pose = double(pose);
    nd = ndims(pose);

    if nd == 2
        dims = size(pose);
        if mod(dims(2), 3) == 0
            joints = dims(2) / 3;
            pose = reshape(pose, [dims(1), 3, joints]);
            info.transform = 'reshape_flat';
        elseif mod(dims(1), 3) == 0
            joints = dims(1) / 3;
            pose = reshape(pose, [3, joints, dims(2)]);
            pose = permute(pose, [3 1 2]);
            info.transform = 'permute_flat';
        else
            error('normalize_pose_format:UnsupportedShape', ...
                'Unable to interpret pose data of size %s', mat2str(dims));
        end
    elseif nd == 3
        dims = size(pose);
        if dims(2) ~= 3
            permCandidates = [
                1 2 3;
                1 3 2;
                2 1 3;
                2 3 1;
                3 1 2;
                3 2 1];
            matched = false;
            for idx = 1:size(permCandidates,1)
                perm = permCandidates(idx,:);
                permDims = dims(perm);
                if permDims(2) == 3
                    pose = permute(pose, perm);
                    info.transform = sprintf('permute_%d%d%d', perm(1), perm(2), perm(3));
                    matched = true;
                    break;
                end
            end
            if ~matched
                error('normalize_pose_format:NoCoordinateDimension', ...
                    'Unable to align coordinate dimension for size %s', mat2str(dims));
            end
        end
    else
        dims = size(pose);
        error('normalize_pose_format:UnsupportedNDims', ...
            'Unsupported pose dimensionality %d with size %s', nd, mat2str(dims));
    end

    info.outputSize = size(pose);
end

function [pose, info] = apply_orientation_heuristic(pose)
    info = struct();
    info.applied = false;
    info.bestLabel = '';
    info.score = NaN;
    info.scores = [];
    info.warning = '';

    if isempty(pose)
        return;
    end

    if size(pose, 2) ~= 3
        error('apply_orientation_heuristic:InvalidFormat', ...
            'Pose data must have size(:,2) == 3 to represent XYZ coordinates.');
    end

    numJoints = size(pose, 3);
    idxUpper = select_joint_indices(numJoints, 'upper');
    idxLower = select_joint_indices(numJoints, 'lower');

    if isempty(idxUpper) || isempty(idxLower)
        info.warning = 'Insufficient joints for orientation heuristic';
        return;
    end

    transforms = get_orientation_transforms();
    numTransforms = size(transforms, 1);
    scores = nan(numTransforms, 1);

    x = pose(:,1,:);
    y = pose(:,2,:);
    z = pose(:,3,:);

    for t = 1:numTransforms
        candidate = transforms{t,1}(x, y, z);
        zUpper = candidate(:,3,idxUpper);
        zLower = candidate(:,3,idxLower);
        diffVals = mean(zUpper, 3, 'omitnan') - mean(zLower, 3, 'omitnan');
        scores(t) = mean(diffVals(:), 'omitnan');
    end

    info.scores = scores;

    if all(~isfinite(scores))
        info.warning = 'Orientation heuristic returned non-finite scores';
        return;
    end

    [bestScore, bestIdx] = max(scores);
    info.score = bestScore;
    info.bestLabel = strtrim(transforms{bestIdx,2});

    pose = transforms{bestIdx,1}(x, y, z);
    info.applied = true;

    if bestScore < 0
        info.warning = 'Best orientation score is negative';
    end
end

function idx = select_joint_indices(numJoints, regionType)
    switch lower(regionType)
        case 'upper'
            if numJoints >= 23
                idx = [4 5 6];
            elseif numJoints >= 14
                idx = [4 5];
            else
                idx = 1:min(3, numJoints);
            end
        case 'lower'
            if numJoints >= 23
                idx = [11 15 19 23];
            elseif numJoints >= 14
                idx = [8 10 12 14];
            else
                idx = max(numJoints-3+1,1):numJoints;
            end
        otherwise
            idx = [];
    end
    idx = idx(idx >= 1 & idx <= numJoints);
end

function transforms = get_orientation_transforms()
    transforms = {
        @(x,y,z) cat(2, x, y, z),      ' [X, Y, Z]';
        @(x,y,z) cat(2, x, -y, -z),    ' [X, -Y, -Z]';
        @(x,y,z) cat(2, x, z, -y),     ' [X, Z, -Y]';
        @(x,y,z) cat(2, x, -z, y),     ' [X, -Z, Y]';
        @(x,y,z) cat(2, y, x, -z),     ' [Y, X, -Z]';
        @(x,y,z) cat(2, y, -x, z),     ' [Y, -X, Z]';
        @(x,y,z) cat(2, z, y, -x),     ' [Z, Y, -X]';
        @(x,y,z) cat(2, z, -y, x),     ' [Z, -Y, X]';
        @(x,y,z) cat(2, -x, y, -z),    ' [-X, Y, -Z]';
        @(x,y,z) cat(2, -x, -y, z),    ' [-X, -Y, Z]';
        @(x,y,z) cat(2, -y, x, z),     ' [-Y, X, Z]';
        @(x,y,z) cat(2, -y, -x, -z),   ' [-Y, -X, -Z]';
    };
end

function [alignedSeq, info] = align_pose_sequence(poseSeq, idxStruct)
    if isempty(poseSeq)
        alignedSeq = poseSeq;
        info = struct('rotation', eye(3), 'forwardAxis', [1 0 0], ...
            'lateralAxis', [0 1 0], 'normalAxis', [0 0 1], ...
            'referenceCenter', zeros(3,1), 'referenceCenterRotated', zeros(3,1));
        return;
    end

    axesInfo = compute_alignment_axes(poseSeq, idxStruct);
    R = axesInfo.rotation;
    refCenter = axesInfo.referenceCenter;
    refCenterRot = R' * refCenter;

    alignedSeq = zeros(size(poseSeq), 'like', poseSeq);
    numFrames = size(poseSeq,1);
    for f = 1:numFrames
        frame = squeeze(poseSeq(f,:,:)); % 3 x joints
        alignedFrame = R' * (frame - refCenter) + refCenterRot;
        alignedSeq(f,:,:) = alignedFrame;
    end

    info = struct();
    info.rotation = R;
    info.forwardAxis = axesInfo.forward;
    info.lateralAxis = axesInfo.lateral;
    info.normalAxis = axesInfo.normal;
    info.referenceCenter = refCenter;
    info.referenceCenterRotated = refCenterRot;
    info.sampleFramesUsed = axesInfo.sampleCount;
end

function axesInfo = compute_alignment_axes(poseSeq, idxStruct)
    numFrames = size(poseSeq,1);
    sampleCount = min(5000, numFrames);
    sampleIdx = unique(round(linspace(1, numFrames, sampleCount)));

    forwardSum = zeros(1,3);
    lateralSum = zeros(1,3);
    normalSum = zeros(1,3);
    centerSum = zeros(1,3);
    validCount = 0;

    for ii = 1:numel(sampleIdx)
        frame = squeeze(poseSeq(sampleIdx(ii),:,:)); % 3 x joints

        frontPos = mean(frame(:, idxStruct.front), 2);
        hindPos  = mean(frame(:, idxStruct.hind), 2);
        leftPos  = mean(frame(:, idxStruct.left), 2);
        rightPos = mean(frame(:, idxStruct.right), 2);

        forwardVec = frontPos - hindPos;
        lateralVec = rightPos - leftPos;
        if norm(forwardVec) < 1e-6 || norm(lateralVec) < 1e-6
            continue;
        end

        normalVec = cross(forwardVec, lateralVec);
        if norm(normalVec) < 1e-6
            continue;
        end

        normalVec = normalVec ./ norm(normalVec);
        if normalVec(3) < 0
            normalVec = -normalVec;
            lateralVec = -lateralVec;
        end

        forwardProj = forwardVec - dot(forwardVec, normalVec) * normalVec;
        lateralProj = lateralVec - dot(lateralVec, normalVec) * normalVec;

        if norm(forwardProj) < 1e-6 || norm(lateralProj) < 1e-6
            continue;
        end

        forwardProj = forwardProj ./ norm(forwardProj);
        lateralProj = lateralProj ./ norm(lateralProj);

        forwardSum = forwardSum + forwardProj';
        lateralSum = lateralSum + lateralProj';
        normalSum = normalSum + normalVec';
        centerSum = centerSum + mean(frame(:, idxStruct.paws), 2)';
        validCount = validCount + 1;
    end

    if validCount == 0 || norm(normalSum) < 1e-6
        forward = [1 0 0];
        normal = [0 0 1];
        center = squeeze(mean(mean(poseSeq,1),3));
        center = center(:);
    else
        normal = normalSum / norm(normalSum);
        forwardRaw = forwardSum - dot(forwardSum, normal) * normal;
        if norm(forwardRaw) < 1e-6
            forward = [1 0 0];
        else
            forward = forwardRaw / norm(forwardRaw);
        end
        center = (centerSum / validCount)';
        center = center(:);
    end

    lateral = cross(normal, forward);
    if norm(lateral) < 1e-6
        lateral = [0 1 0];
    else
        lateral = lateral / norm(lateral);
    end

    forward = forward(:);
    lateral = lateral(:);
    normal = normal(:);
    R = [forward lateral normal];

    axesInfo = struct();
    axesInfo.rotation = R;
    axesInfo.forward = forward';
    axesInfo.lateral = lateral';
    axesInfo.normal = normal';
    axesInfo.referenceCenter = center;
    axesInfo.sampleCount = validCount;
end

function metadata = create_file_metadata(file_list)
    % Create metadata structure for organizing files
    metadata = struct();
    metadata.all_files = file_list;
    metadata.groups = {};
    metadata.weeks = {};
    
    for i = 1:length(file_list)
        [group, batchLabel] = extract_metadata_from_filename(file_list{i});
        metadata.groups{i} = group;
        metadata.weeks{i} = batchLabel;
    end
end

function [group, batchLabel] = extract_metadata_from_filename(filename)
    % Extract group and batch information from filename for cpfull/wt datasets
    
    [~, name, ~] = fileparts(filename);
    base = lower(name);
    
    if contains(base, 'cpfull')
        group = 'CPFULL';
    elseif contains(base, 'wt')
        group = 'WT';
    else
        group = 'UNKNOWN';
    end
    
    tokens = regexp(base, 'vid(\d+)', 'tokens', 'once');
    batchLabel = 'batch_unknown';
    if ~isempty(tokens)
        vidNum = str2double(tokens{1});
        if vidNum <= 3
            batchLabel = 'batch1';
        else
            batchLabel = 'batch2';
        end
    end
end

function label = infer_batch_label(filename)
    % Batch assignment derived from video index (vid#) for cpfull/wt datasets
    [~, batchLabel] = extract_metadata_from_filename(filename);
    if isempty(batchLabel)
        label = 'batch_unknown';
    else
        label = batchLabel;
    end
end

function [grp, id] = extract_group_id_from_filename(filename)
    % Returns group string and numeric id for cpfull/wt datasets
    [~, name, ~] = fileparts(filename);
    base = lower(name);
    
    if contains(base, 'cpfull')
        grp = 'CPFULL';
    elseif contains(base, 'wt')
        grp = 'WT';
    else
        grp = upper(name);
    end
    
    tokens = regexp(base, 'vid(\d+)', 'tokens', 'once');
    if ~isempty(tokens)
        id = str2double(tokens{1});
    else
        id = NaN;
    end
end

function [xx, density] = findPointDensity(points, sigma, gridSize, xRange)
    % Return xx as a 1D grid vector (compatible with downstream code)
    % and a 2D density map over xx-by-xx.
    
    if size(points, 1) < 10
        xx = linspace(xRange(1), xRange(2), gridSize);
        density = zeros(gridSize, gridSize);
        return;
    end
    
    % Build 1D grids
    if numel(xRange) == 2
        xx = linspace(xRange(1), xRange(2), gridSize);
        yy = xx;
    else
        xx = linspace(xRange(1), xRange(2), gridSize);
        yy = linspace(xRange(3), xRange(4), gridSize);
    end
    
    % Mesh for density computation (not returned)
    [gridX, gridY] = meshgrid(xx, yy);
    
    % Subsample for efficiency
    if size(points, 1) > 10000
        idx = randperm(size(points, 1), 10000);
        points = points(idx, :);
    end
    
    % KDE with Gaussian kernels
    density = zeros(gridSize, gridSize);
    invTwoSigma2 = 1 / (2 * sigma^2);
    normConst = 2 * pi * sigma^2;
    for i = 1:size(points, 1)
        px = points(i, 1);
        py = points(i, 2);
        dist2 = (gridX - px).^2 + (gridY - py).^2;
        density = density + exp(-dist2 * invTwoSigma2);
    end
    density = density / (size(points, 1) * normConst);
end

function combined = combineCells(cellArray, dim)
    % Combine cell array contents
    if nargin < 2
        dim = 1;
    end
    
    if isempty(cellArray)
        combined = [];
        return;
    end
    
    % Remove empty cells
    cellArray = cellArray(~cellfun(@isempty, cellArray));
    
    if isempty(cellArray)
        combined = [];
        return;
    end
    
    if dim == 1
        combined = vertcat(cellArray{:});
    else
        combined = horzcat(cellArray{:});
    end
end

function params = setRunParameters(params)
    % Set default run parameters if not provided
    
    if isempty(params)
        params = struct();
    end
    
    if ~isfield(params, 'samplingFreq')
        params.samplingFreq = 50;  % Hz
    end
    
    if ~isfield(params, 'minF')
        params.minF = 0.5;  % Hz
    end
    
    if ~isfield(params, 'maxF')
        params.maxF = 20;  % Hz
    end
    
    if ~isfield(params, 'omega0')
        params.omega0 = 5;  % Wavelet center frequency
    end
    
    if ~isfield(params, 'numPeriods')
        params.numPeriods = 5;  % Number of wavelet periods
    end
    
    if ~isfield(params, 'batchSize')
        params.batchSize = 10000;  % For processing large datasets
    end
    
    % MotionMapper specific parameters
    if ~isfield(params, 'kdNeighbors')
        params.kdNeighbors = 5;  % Number of nearest neighbors for template matching
    end
    
    if ~isfield(params, 'templateLength')
        params.templateLength = 25;  % Template length in frames
    end
    
    if ~isfield(params, 'minTemplateLength')
        params.minTemplateLength = 10;  % Minimum template length in frames
    end
    
    % Defaults needed for t-SNE projection search
    if ~isfield(params, 'sigmaTolerance')
        params.sigmaTolerance = 1e-5;  % Tolerance for sigma search in perplexity match
    end
    
    if ~isfield(params, 'maxNeighbors')
        params.maxNeighbors = 200;  % Max neighbors for sparse probability computation
    end
    
    if ~isfield(params, 'numProcessors')
        params.numProcessors = 1;  % Number of processors for parallel processing
    end
end

function create_group_comparison_plots(results, groupColors)
    % Create comprehensive group comparison plots
    
    fprintf('Creating group comparison plots...\n');
    
    % Extract unique groups and weeks
    all_groups = {};
    all_weeks = {};
    for i = 1:length(results.reembedding_metadata_all)
        if ~isempty(results.reembedding_metadata_all{i})
            group = results.reembedding_metadata_all{i}.group;
            week = results.reembedding_metadata_all{i}.week;
            
            if ~any(strcmp(all_groups, group))
                all_groups{end+1} = group;
            end
            if ~any(strcmp(all_weeks, week))
                all_weeks{end+1} = week;
            end
        end
    end
    
    % Create overview plot
    figure('Name', 'All Groups Overview', 'Position', [100 100 1500 1000]);
    
    % Plot background density
    imagesc(results.D);
    axis equal off;
    colormap(flipud(gray));
    caxis([0 max(results.D(:))*0.8]);
    hold on;
    
    % Add watershed boundaries
    scatter(results.llbwb(:,2), results.llbwb(:,1), 0.5, '.', 'k', 'MarkerEdgeAlpha', 0.3);
    
    % Plot each group with different colors
    legend_handles = [];
    legend_labels = {};
    
    for g = 1:length(all_groups)
        group = all_groups{g};
        
        % Find indices for this group
        group_indices = [];
        for i = 1:length(results.reembedding_metadata_all)
            if ~isempty(results.reembedding_metadata_all{i}) && ...
               strcmp(results.reembedding_metadata_all{i}.group, group)
                group_indices = [group_indices, i];
            end
        end
        
        if ~isempty(group_indices)
            % Collect all points for this group
            all_points = [];
            for idx = group_indices
                z = results.zEmbeddings_all{idx};
                z_img = (z + 65) * 501 / 130;  % Transform to image coordinates
                all_points = [all_points; z_img];
            end
            
            % Create density overlay
            if size(all_points, 1) > 10
                [N, xedges, yedges] = histcounts2(all_points(:,1), all_points(:,2), ...
                    linspace(1, 501, 60), linspace(1, 501, 60));
                
                % Smooth the density
                if exist('imgaussfilt', 'file')
                    N = imgaussfilt(N', 3);
                else
                    N = conv2(N', ones(5)/25, 'same');
                end
                
                % Get color for this group
                clean_group = strrep(group, '_flip', '');
                if isfield(groupColors, clean_group)
                    color = groupColors.(clean_group);
                else
                    color = rand(1, 3);  % Random color if not defined
                end
                
                % Create colored overlay
                overlay = zeros(size(N,1), size(N,2), 3);
                for c = 1:3
                    overlay(:,:,c) = color(c);
                end
                
                % Display density overlay
                h = imagesc(xedges(1:end-1), yedges(1:end-1), overlay);
                set(h, 'AlphaData', N/max(N(:))*0.6);
                
                % Add to legend
                h_legend = scatter(nan, nan, 100, color, 'filled', 's');
                legend_handles = [legend_handles, h_legend];
                legend_labels = [legend_labels, {strrep(group, '_', ' ')}];
            end
        end
    end
    
    title('All Groups - Re-embedded using SNI\_2 + week4-TBI\_3 Training');
    legend(legend_handles, legend_labels, 'Location', 'eastoutside');
end

function create_individual_mouse_plots(results, groupColors)
    % Create individual mouse density plots
    
    fprintf('Creating individual mouse plots...\n');
    
    % Group by experimental group
    group_indices = struct();
    for i = 1:length(results.reembedding_metadata_all)
        if ~isempty(results.reembedding_metadata_all{i})
            group = results.reembedding_metadata_all{i}.group;
            clean_group = strrep(group, '_flip', '');
            
            if ~isfield(group_indices, clean_group)
                group_indices.(clean_group) = [];
            end
            group_indices.(clean_group) = [group_indices.(clean_group), i];
        end
    end
    
    % Create plots for each group
    field_names = fieldnames(group_indices);
    for g = 1:length(field_names)
        group = field_names{g};
        indices = group_indices.(group);
        
        if length(indices) >= 4  % Only create plots for groups with enough samples
            figure('Name', sprintf('%s Individual Mice', group), 'Position', [100 100 1800 1000]);
            
            n_mice = length(indices);
            n_cols = min(6, n_mice);
            n_rows = ceil(n_mice / n_cols);
            
            for m = 1:length(indices)
                idx = indices(m);
                
                subplot(n_rows, n_cols, m);
                
                % Background
                imagesc(results.D);
                hold on;
                
                % Get embedding data
                z = results.zEmbeddings_all{idx};
                z_img = (z + 65) * 501 / 130;
                
                % Create density map
                if size(z_img, 1) > 10
                    [N, xedges, yedges] = histcounts2(z_img(:,1), z_img(:,2), ...
                        linspace(1, 501, 40), linspace(1, 501, 40));
                    
                    if exist('imgaussfilt', 'file')
                        N = imgaussfilt(N', 2);
                    else
                        N = conv2(N', ones(3)/9, 'same');
                    end
                    
                    % Get color
                    if isfield(groupColors, group)
                        color = groupColors.(group);
                    else
                        color = [0.5 0.5 0.5];
                    end
                    
                    % Create overlay
                    overlay = zeros(size(N,1), size(N,2), 3);
                    for c = 1:3
                        overlay(:,:,c) = color(c);
                    end
                    
                    % Display
                    if max(N(:)) > 0
                        h = imagesc(xedges(1:end-1), yedges(1:end-1), overlay);
                        set(h, 'AlphaData', N/max(N(:))*0.8);
                    end
                end
                
                % Add watershed boundaries
                scatter(results.llbwb(:,2), results.llbwb(:,1), 0.5, '.', 'k', 'MarkerEdgeAlpha', 0.3);
                
                axis equal off;
                colormap(gca, flipud(gray));
                caxis([0 max(results.D(:))*0.8]);
                
                % Create title from filename
                if ~isempty(results.reembedding_labels_all{idx})
                    mouse_name = strrep(results.reembedding_labels_all{idx}, '_', '\_');
                    title(mouse_name, 'Interpreter', 'tex', 'FontSize', 8);
                end
            end
            
            sgtitle(sprintf('%s - Individual Mouse Density Maps', group), 'FontSize', 14);
        end
    end
end

function create_temporal_analysis_plots(results, groupColors)
    % Create temporal analysis plots comparing week1 vs week4
    
    fprintf('Creating temporal analysis plots...\n');
    
    % Separate by week
    week1_indices = [];
    week4_indices = [];
    
    for i = 1:length(results.reembedding_metadata_all)
        if ~isempty(results.reembedding_metadata_all{i})
            week = results.reembedding_metadata_all{i}.week;
            if strcmp(week, 'week1')
                week1_indices = [week1_indices, i];
            elseif strcmp(week, 'week4')
                week4_indices = [week4_indices, i];
            end
        end
    end
    
    if ~isempty(week1_indices) && ~isempty(week4_indices)
        figure('Name', 'Temporal Analysis: Week1 vs Week4', 'Position', [100 100 1400 700]);
        
        % Week 1
        subplot(1, 2, 1);
        imagesc(results.D);
        hold on;
        
        % Group week1 data by experimental group
        week1_groups = struct();
        for idx = week1_indices
            if ~isempty(results.reembedding_metadata_all{idx})
                group = results.reembedding_metadata_all{idx}.group;
                clean_group = strrep(group, '_flip', '');
                
                if ~isfield(week1_groups, clean_group)
                    week1_groups.(clean_group) = [];
                end
                week1_groups.(clean_group) = [week1_groups.(clean_group), idx];
            end
        end
        
        % Plot each group
        legend_handles1 = [];
        legend_labels1 = {};
        group_names = fieldnames(week1_groups);
        
        for g = 1:length(group_names)
            group = group_names{g};
            indices = week1_groups.(group);
            
            % Collect all points
            all_points = [];
            for idx = indices
                z = results.zEmbeddings_all{idx};
                z_img = (z + 65) * 501 / 130;
                all_points = [all_points; z_img];
            end
            
            if size(all_points, 1) > 10
                [N, xedges, yedges] = histcounts2(all_points(:,1), all_points(:,2), ...
                    linspace(1, 501, 50), linspace(1, 501, 50));
                
                if exist('imgaussfilt', 'file')
                    N = imgaussfilt(N', 2);
                else
                    N = conv2(N', ones(3)/9, 'same');
                end
                
                % Get color
                if isfield(groupColors, group)
                    color = groupColors.(group);
                else
                    color = rand(1, 3);
                end
                
                % Create overlay
                overlay = zeros(size(N,1), size(N,2), 3);
                for c = 1:3
                    overlay(:,:,c) = color(c);
                end
                
                if max(N(:)) > 0
                    h = imagesc(xedges(1:end-1), yedges(1:end-1), overlay);
                    set(h, 'AlphaData', N/max(N(:))*0.6);
                end
                
                % Add to legend
                h_legend = scatter(nan, nan, 100, color, 'filled', 's');
                legend_handles1 = [legend_handles1, h_legend];
                legend_labels1 = [legend_labels1, {group}];
            end
        end
        
        scatter(results.llbwb(:,2), results.llbwb(:,1), 0.5, '.', 'k', 'MarkerEdgeAlpha', 0.3);
        axis equal off;
        colormap(gca, flipud(gray));
        caxis([0 max(results.D(:))*0.8]);
        title('Week 1');
        legend(legend_handles1, legend_labels1, 'Location', 'best');
        
        % Week 4
        subplot(1, 2, 2);
        imagesc(results.D);
        hold on;
        
        % Group week4 data
        week4_groups = struct();
        for idx = week4_indices
            if ~isempty(results.reembedding_metadata_all{idx})
                group = results.reembedding_metadata_all{idx}.group;
                clean_group = strrep(group, '_flip', '');
                
                if ~isfield(week4_groups, clean_group)
                    week4_groups.(clean_group) = [];
                end
                week4_groups.(clean_group) = [week4_groups.(clean_group), idx];
            end
        end
        
        % Plot each group
        legend_handles2 = [];
        legend_labels2 = {};
        group_names = fieldnames(week4_groups);
        
        for g = 1:length(group_names)
            group = group_names{g};
            indices = week4_groups.(group);
            
            % Collect all points
            all_points = [];
            for idx = indices
                z = results.zEmbeddings_all{idx};
                z_img = (z + 65) * 501 / 130;
                all_points = [all_points; z_img];
            end
            
            if size(all_points, 1) > 10
                [N, xedges, yedges] = histcounts2(all_points(:,1), all_points(:,2), ...
                    linspace(1, 501, 50), linspace(1, 501, 50));
                
                if exist('imgaussfilt', 'file')
                    N = imgaussfilt(N', 2);
                else
                    N = conv2(N', ones(3)/9, 'same');
                end
                
                % Get color (use week4 color scheme if available)
                week4_group_name = ['week4_' group];
                if isfield(groupColors, week4_group_name)
                    color = groupColors.(week4_group_name);
                elseif isfield(groupColors, group)
                    color = groupColors.(group) * 0.7;  % Darker version
                else
                    color = rand(1, 3);
                end
                
                % Create overlay
                overlay = zeros(size(N,1), size(N,2), 3);
                for c = 1:3
                    overlay(:,:,c) = color(c);
                end
                
                if max(N(:)) > 0
                    h = imagesc(xedges(1:end-1), yedges(1:end-1), overlay);
                    set(h, 'AlphaData', N/max(N(:))*0.6);
                end
                
                % Add to legend
                h_legend = scatter(nan, nan, 100, color, 'filled', 's');
                legend_handles2 = [legend_handles2, h_legend];
                legend_labels2 = [legend_labels2, {group}];
            end
        end
        
        scatter(results.llbwb(:,2), results.llbwb(:,1), 0.5, '.', 'k', 'MarkerEdgeAlpha', 0.3);
        axis equal off;
        colormap(gca, flipud(gray));
        caxis([0 max(results.D(:))*0.8]);
        title('Week 4');
        legend(legend_handles2, legend_labels2, 'Location', 'best');
        
        sgtitle('Temporal Analysis: Week1 vs Week4 (Re-embedded using SNI\_2 + week4-TBI\_3)', 'FontSize', 14);
    end
end

function [model, Xcorr] = fit_mnn_correction(X, batchVec, opts)
    % Train an MNN correction model and return corrected training data

    if nargin < 3 || isempty(opts)
        opts = struct();
    end
    if ~isfield(opts,'k') || isempty(opts.k), opts.k = 20; end
    if ~isfield(opts,'ndim') || isempty(opts.ndim), opts.ndim = min(50, size(X,2)); end
    if ~isfield(opts,'sigma'), opts.sigma = []; end
    if ~isfield(opts,'distance') || isempty(opts.distance), opts.distance = 'euclidean'; end
    if ~isfield(opts,'verbose') || isempty(opts.verbose), opts.verbose = true; end

    validateattributes(X, {'double'}, {'2d','nonempty'}, mfilename, 'X', 1);
    mustBeVector(batchVec);
    if ~(isscalar(opts.k) && isnumeric(opts.k) && opts.k > 0)
        error('fit_mnn_correction:invalidK','opts.k must be a positive scalar');
    end
    if ~(isscalar(opts.ndim) && isnumeric(opts.ndim) && opts.ndim > 0)
        error('fit_mnn_correction:invalidDim','opts.ndim must be a positive scalar');
    end

    ndim = min([opts.ndim, size(X,2)]);

    % Robust per-feature scaling (median/MAD) to temper outliers
    mu = median(X,1,'omitnan');
    mad0 = 1.4826*mad(X,1,1);
    mad0(mad0==0) = 1;
    Xs = (X - mu)./mad0;

    % Center for PCA explicitly (store mean for apply step)
    pcMu = mean(Xs,1);
    X0 = Xs - pcMu;

    % Economy SVD for PCA
    [~,S,V] = svd(X0,'econ'); %#ok<ASGLU>
    W = V(:,1:ndim);
    Z = X0*W;

    % Choose reference batch = largest batch by N
    [groups,~,gidx] = unique(batchVec);
    Ns = accumarray(gidx,1);
    [~,refI] = max(Ns);
    refBatch = groups(refI);
    refMask = (gidx==refI);

    X0_ref = X0(refMask,:);
    Z_ref = Z(refMask,:);

    % Kernel bandwidth (auto) based on reference kNN distances
    if isempty(opts.sigma)
        idxRef = knnsearch(Z_ref, Z_ref, 'K', max(3,opts.k), 'Distance', opts.distance);
        dd = zeros(size(Z_ref,1),1);
        for i = 1:size(Z_ref,1)
            nbr = idxRef(i,end);
            dd(i) = norm(Z_ref(i,:) - Z_ref(nbr,:));
        end
        sigma = median(dd(dd>0));
        if ~isfinite(sigma) || sigma<=0
            sigma = 1;
        end
    else
        sigma = opts.sigma;
    end
    sigma2 = sigma^2;

    % Prepare corrected matrix (centered space)
    X0_corr = zeros(size(X0));
    X0_corr(refMask,:) = X0_ref;
    order = [refI, setdiff(1:numel(groups), refI)];

    if opts.verbose
        fprintf('MNN: reference batch = %s (#%d, N=%d), k=%d, PCs=%d, sigma=%.3g\n', ...
            string(refBatch), refI, sum(refMask), opts.k, ndim, sigma);
    end

    % Sequentially merge each remaining batch into the reference
    for oi = 2:numel(order)
        b = order(oi);
        maskB = (gidx==b);
        if ~any(maskB)
            continue;
        end

        Z_b = Z(maskB,:);
        X0_b = X0(maskB,:);

        Z_ref_now = (X0_corr(refMask | ismember(gidx,order(1:oi-1)),:)) * W; %#ok<*ISMT>
        X0_ref_now = X0_corr(refMask | ismember(gidx,order(1:oi-1)),:);

        [idxB2R, ~] = knnsearch(Z_ref_now, Z_b, 'K', opts.k, 'Distance', opts.distance); %#ok<ASGLU>
        [idxR2B, ~] = knnsearch(Z_b, Z_ref_now, 'K', opts.k, 'Distance', opts.distance); %#ok<ASGLU>

        N_b = size(Z_b,1);
        offsets = zeros(N_b, size(X0_b,2));
        used = false(N_b,1);

        for i = 1:N_b
            anchors = idxB2R(i,:);
            isMutual = false(numel(anchors),1);
            for j = 1:numel(anchors)
                isMutual(j) = any(idxR2B(anchors(j),:) == i);
            end
            anchors = anchors(isMutual);
            if isempty(anchors)
                anchors = idxB2R(i,1);
            else
                used(i) = true;
            end

            D = sum((Z_ref_now(anchors,:) - Z_b(i,:)).^2,2);
            w = exp(-D/(2*sigma2));
            wsum = sum(w);
            if wsum==0
                w = ones(size(w));
                wsum = numel(w);
            end

            delta = (w'/wsum) * (X0_ref_now(anchors,:) - X0_b(i,:));
            offsets(i,:) = delta;
        end

        X0_b_corr = X0_b + offsets;
        X0_corr(maskB,:) = X0_b_corr;

        if opts.verbose
            fprintf('MNN: merged batch #%d (N=%d); anchors found for %d/%d points\n', ...
                b, sum(maskB), sum(used), N_b);
        end
    end

    % Uncenter / unscale back to original feature space
    Xcorr = (X0_corr + pcMu).*mad0 + mu;

    % Build model for future application
    model = struct();
    model.mu = mu;
    model.mad = mad0;
    model.pcMu = pcMu;
    model.W = W;
    model.ndim = ndim;
    model.k = opts.k;
    model.sigma = sigma;
    model.distance = opts.distance;
    model.refZ = (X0_corr(refMask | ismember(gidx,order),:)) * W;
    model.refX0 = X0_corr(refMask | ismember(gidx,order),:);

    if opts.verbose
        fprintf('MNN: model ready (ref points = %d)\n', size(model.refZ,1));
    end
end

function Xcorr = apply_mnn_correction(Xnew, model)
    % Apply a trained MNN correction model to new data before projection

    Xs = (Xnew - model.mu)./model.mad;
    X0 = Xs - model.pcMu;
    Z = X0 * model.W;

    k = model.k;
    sigma2 = model.sigma^2;

    idxN2R = knnsearch(model.refZ, Z, 'K', k, 'Distance', model.distance);
    idxR2N = knnsearch(Z, model.refZ, 'K', k, 'Distance', model.distance);

    M = size(Z,1);
    offsets = zeros(M, size(X0,2));

    for i = 1:M
        anchors = idxN2R(i,:);
        isMutual = false(numel(anchors),1);
        for j = 1:numel(anchors)
            isMutual(j) = any(idxR2N(anchors(j),:) == i);
        end
        anchors = anchors(isMutual);
        if isempty(anchors)
            anchors = idxN2R(i,1);
        end

        D = sum((model.refZ(anchors,:) - Z(i,:)).^2,2);
        w = exp(-D/(2*sigma2));
        wsum = sum(w);
        if wsum==0
            w = ones(size(w));
            wsum = numel(w);
        end

        delta = (w'/wsum) * (model.refX0(anchors,:) - X0(i,:));
        offsets(i,:) = delta;
    end

    X0_corr = X0 + offsets;
    Xcorr = (X0_corr + model.pcMu).*model.mad + model.mu;
end
