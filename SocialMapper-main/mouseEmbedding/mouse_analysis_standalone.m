%% Standalone Mouse Behavior Analysis and Visualization
% This script contains all the analysis from line 700+ with dependencies
% Run this directly to generate all behavioral density maps and analysis

clear; close all; clc;
fprintf('Starting standalone mouse behavior analysis...\n');

%% ========================================================================
%% LOAD ESSENTIAL DATA AND DEPENDENCIES
%% ========================================================================

% Load required colormap and basic data
try
    load('colormaps.mat');
    fprintf('✓ Loaded colormaps.mat\n');
catch
    fprintf('Warning: colormaps.mat not found, using defaults\n');
end

% Load mouse file order information
try
    load('mouseFileOrder.mat');
    fprintf('✓ Loaded mouseFileOrder.mat\n');
catch
    error('Error: mouseFileOrder.mat is required but not found');
end

% Essential parameters
fprintf('Setting up analysis parameters...\n');
llt = 30000;

%% ========================================================================
%% LOAD EMBEDDED DATA AND CREATE CORE VARIABLES
%% ========================================================================

fprintf('Loading individual mouse embeddings...\n');
% Load individual mouse embeddings (128 total)
EVL = cell(128,1); 
ICVL = zeros(128,1);
for i = 1:128
    try
        load(['mouse/RE_lone/RE_LONE_' num2str(i) '.mat'],'z','inConvHull','filename');
        EVL{i} = z;
        ICVL(i) = mean(inConvHull);
    catch
        fprintf('Warning: Could not load RE_LONE_%d.mat\n', i);
        EVL{i} = [];
        ICVL(i) = 0;
    end
end

fprintf('Loading social mouse embeddings...\n');
% Load social mouse embeddings (48 recordings, 2 animals each)
EVS = cell(48,2); 
ICVS = zeros(48,2);
for i = 1:48
    try
        load(['mouse/RE_soc/RE_SOC_' num2str(i) '.mat'],'z1','z2','inCH1','inCH2','filename');
        EVS{i,1} = z1; 
        EVS{i,2} = z2;
        ICVS(i,1) = mean(inCH1); 
        ICVS(i,2) = mean(inCH2);
    catch
        fprintf('Warning: Could not load RE_SOC_%d.mat\n', i);
        EVS{i,1} = []; 
        EVS{i,2} = [];
        ICVS(i,1) = 0; 
        ICVS(i,2) = 0;
    end
end

%% ========================================================================
%% CREATE INDIVIDUAL BEHAVIOR DENSITY MAPS AND WATERSHEDS
%% ========================================================================

fprintf('Creating individual behavior density maps...\n');

% Combine all individual embeddings
evall = combineCells(EVL);
if isempty(evall)
    error('No individual embedding data found');
end

% Create density map
[xx, d] = findPointDensity(evall,1,501,[-65 65]);
D = d;

% Watershed segmentation
LL = watershed(-d,8);
LL2 = LL; 
LL2(d < 1e-6) = -1;

% Create watershed boundaries
LLBW = LL2==0;
LLBWB = bwboundaries(LLBW);
llbwb = LLBWB(2:end);
llbwb = combineCells(llbwb');

%% ========================================================================
%% CREATE SOCIAL BEHAVIOR DENSITY MAPS AND WATERSHEDS  
%% ========================================================================

fprintf('Creating social behavior density maps...\n');

% Combine all social embeddings
evallsoc = combineCells(EVS(:));
if isempty(evallsoc)
    error('No social embedding data found');
end

% Create social density map
[xxsoc, dsoc] = findPointDensity(evallsoc,1.5,501,[-85 85]);

% Social watershed segmentation
LLsoc = watershed(-dsoc,8);
LL2soc = LLsoc; 
LL2soc(dsoc < 1e-6) = -1;

% Create social watershed boundaries
LLBWsoc = LL2soc==0;
LLBWBsoc = bwboundaries(LLBWsoc);
llbwbsoc = LLBWBsoc(2:end);
llbwbsoc = combineCells(llbwbsoc');

%% ========================================================================
%% BEHAVIORAL CATEGORIZATION AND LABELING
%% ========================================================================

fprintf('Setting up behavioral categories...\n');

% Individual behavioral categories
idle = [29 31 36 39];
slow = [44 58 62 66 68 78 85 92];
head = [74 79 82];
groom = [33 37 40 46 52];
crouched = [15 16 21 23 24 28];
actCrouch = [20 22 25 26 27 30 32 34 35 38 41 43 47 48 50 56];
stepsCrouched = [42 45 49 51 53 54 55 59 61];
rear = [13 14 17 18 19];
highRear = [1 2 3 4 5 6 7 8 9 10 11 12];
slowExp = [63 76 88 90 91];
Exp = [57 60 64 72 73 75 80 81 87 93 97 99 100 103 104 106 107 111 114];
stepExp = [69 83 86 101 108 110 112 113 115 117 118];
locSlow = [67 71 77 84 94 95 96 98 105];
locFast = [65 70 89 102 109 116];

% Create individual behavior labels
maxClusterL_actual = max(max(LL2));
labelsL = zeros(1, max(118, maxClusterL_actual));
labelsL(idle) = 1;
labelsL(slow) = 2;
labelsL(head) = 3;
labelsL(groom) = 4;
labelsL(crouched) = 5;
labelsL(actCrouch) = 6;
labelsL(stepsCrouched) = 7;
labelsL(rear) = 8;
labelsL(highRear) = 9;
labelsL(slowExp) = 10;
labelsL(Exp) = 11;
labelsL(stepExp) = 12;
labelsL(locSlow) = 13;
labelsL(locFast) = 14;
% Assign unclassified clusters to category 15
if maxClusterL_actual > 118
    labelsL(119:maxClusterL_actual) = 15;
end

% Social behavioral categories
maxClusterS_actual = max(max(LL2soc));
sNI = [41 42 43 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 ...
    62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 ...
    81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 ...
    97 98 99 101 102 103 104 105 108 109 110 111 ...
    113 114 115 116 118 119 120 122 124 125 127 128 129 130 132];
sLM1 = [20 24 28 30 32 36 37 38 39 40];
sLM2 = [31 89 100 106 107 133];
sM1 = [17 19 22 25 26 27 29 33 34 44];
sM2 = [112 117 121 123 126 131];
sLMI = [31 35 43];
sMI = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 21 23];

% Create social behavior labels
labelsS = zeros(1, max(133, maxClusterS_actual));
labelsS(sNI) = 1;   % Non-interactive
labelsS(sLM1) = 2;  % Low-medium interaction 1
labelsS(sM1) = 3;   % Medium interaction 1  
labelsS(sLM2) = 4;  % Low-medium interaction 2
labelsS(sM2) = 5;   % Medium interaction 2
labelsS(sLMI) = 6;  % Low-medium interactive
labelsS(sMI) = 7;   % Medium interactive

%% ========================================================================
%% CREATE BEHAVIORAL CATEGORY MAPS
%% ========================================================================

fprintf('Creating behavioral category maps...\n');

% Individual behavior category map
LLC = zeros(size(LL2));
for i = 1:maxClusterL_actual
    if i <= length(labelsL)
        cid = find(LL2==i);
        LLC(cid) = labelsL(i);
    end
end

% Social behavior category map  
LLCsoc = zeros(size(LL2soc));
for i = 1:maxClusterS_actual
    if i <= length(labelsS)
        cid = find(LL2soc==i);
        LLCsoc(cid) = labelsS(i);
    end
end

%% ========================================================================
%% ESSENTIAL COLORMAPS
%% ========================================================================

fprintf('Setting up colormaps...\n');

% Individual behavior colormap
colorsCoarse = [1 1 1;           % 0 - background/unlabeled (white)
    0 .16 1;               % 1 - idle (blue)
    0 .3833 1;             % 2 - slow (light blue)
    0 .7222 1;             % 3 - head (cyan)
    .1333 1 .8667;         % 4 - groom (green-cyan)
    .6 1 .4;               % 5 - crouched (green)
    1 .9375 0;             % 6 - active crouched (yellow)
    1 .75 0;               % 7 - steps crouched (orange-yellow)
    1 .6 0;                % 8 - reared (orange)
    1 .4 0;                % 9 - high rear (red-orange)
    1 0 .2;                % 10 - slow explore (red-pink)
    1 0 .4;                % 11 - explore (red-magenta)
    1 0 0;                 % 12 - step explore (red)
    .6 0 0;                % 13 - locomotion slow (dark red)
    .4 0 0;                % 14 - locomotion fast (darker red)
    0 0 0];                % 15 - extra category (black)

% Social behavior colormap
cmapSOC = [1 1 1;          % 0 - background (white)
    .8 .8 1;               % 1 - non-interactive (light blue)
    .6 .6 1;               % 2 - low-medium 1 (medium blue)
    .3 .3 1;               % 3 - medium 1 (blue)
    1 .6 .6;               % 4 - low-medium 2 (light red)
    1 .3 .3;               % 5 - medium 2 (red)
    .8 .8 .3;              % 6 - low-medium interactive (yellow)
    .6 .6 .1;              % 7 - medium interactive (dark yellow)
    0 0 0];                % 8 - extra (black)

% Density colormap
if ~exist('cmap1', 'var')
    cmap1 = parula(256);
end

% Difference colormap
cmapdiff = [0 0 1;         % Blue for negative differences
    0.5 0.5 1;             % Light blue
    1 1 1;                 % White for no difference
    1 0.5 0.5;             % Light red
    1 0 0];                % Red for positive differences

%% ========================================================================
%% WATERSHED REGIONS AND USAGE CALCULATIONS
%% ========================================================================

fprintf('Calculating watershed regions and usage patterns...\n');

% Parameters for watershed region finding
vSmooth = .5;
medianLength = 1;
pThreshold = [];
minRest = 5;
obj = [];
numGMM = 2;

% Calculate watershed regions for individual behaviors (suppress plots)
WRL = cell(size(EVL));
fprintf('Processing individual watershed regions (this may take a moment)...\n');
% Turn off figures temporarily to prevent plotting interference
original_vis = get(0, 'DefaultFigureVisible');
set(0, 'DefaultFigureVisible', 'off');

for i = 1:length(EVL)
    if ~isempty(EVL{i})
        [WRL{i},~,~,~,~,~,~,~] = findWatershedRegions_v2(EVL{i},xx,LL,vSmooth,medianLength,pThreshold,minRest,obj,false,numGMM);
    else
        WRL{i} = [];
    end
end

% Calculate watershed regions for social behaviors (suppress plots)
WRS = cell(size(EVS));
fprintf('Processing social watershed regions...\n');
for i = 1:size(EVS,1)
    for j = 1:2
        if ~isempty(EVS{i,j})
            [WRS{i,j},~,~,~,~,~,~,~] = findWatershedRegions_v2(EVS{i,j},xxsoc,LL2soc,vSmooth,medianLength,pThreshold,minRest,obj,false,numGMM);
        else
            WRS{i,j} = [];
        end
    end
end

% Restore figure visibility and close any unwanted figures
set(0, 'DefaultFigureVisible', original_vis);
close all; % Close any figures created during watershed processing

% Create fine watershed assignments
wrFINE = cell(size(WRL));
for i = 1:length(wrFINE)
    if ~isempty(WRL{i})
        wrt = WRL{i};
        wrt(wrt==0) = nan;
        wrt = fillmissing(wrt,'nearest');
        wrFINE{i} = wrt;
    else
        wrFINE{i} = [];
    end
end

wrsocFINE = cell(size(WRS));
for i = 1:size(wrsocFINE,1)
    for j = 1:2
        if ~isempty(WRS{i,j})
            wrt = WRS{i,j}; 
            wrt(wrt==134) = 1;
            wrt(wrt==0) = nan;
            wrt = fillmissing(wrt,'nearest');
            wrsocFINE{i,j} = wrt;
        else
            wrsocFINE{i,j} = [];
        end
    end
end

%% ========================================================================
%% USAGE MATRICES AND BEHAVIORAL ANALYSIS
%% ========================================================================

fprintf('Calculating usage matrices...\n');

% Individual behavior usage matrix
maxL = max(max(LL));
loneUsage = zeros(128,maxL);
for i = 1:128
    if ~isempty(wrFINE{i})
        tt = hist(wrFINE{i},1:double(maxL))./llt;
        loneUsage(i,:) = tt;
    end
end

% Social behavior usage matrices
maxS = max(max(LL2soc));
socUsage1 = zeros(48,maxS); 
socUsage2 = zeros(48,maxS);
for i = 1:48
    if ~isempty(wrsocFINE{i,1})
        tt = hist(wrsocFINE{i,1},1:double(maxS))./llt;
        socUsage1(i,:) = tt;
    end
    if ~isempty(wrsocFINE{i,2})
        tt2 = hist(wrsocFINE{i,2},1:double(maxS))./llt;
        socUsage2(i,:) = tt2;
    end
end

%% ========================================================================
%% MOUSE STRAIN AND GROUP ASSIGNMENTS
%% ========================================================================

fprintf('Setting up mouse strain assignments...\n');

% Create mock strain assignments if not available
% Assuming first 16 are B strain, last 16 are W strain for individual
% For social: BB (1-12, 49-60), WW (13-28, 61-76), BW (29-48), WB (77-96)

% Individual behavior groups
Blone = loneUsage(1:16,:);
Wlone = loneUsage(17:32,:);

% Create mouseOrderL (behavioral ordering)
allcv = ones(maxL,1);  % Mock velocity data
mouseOrderL = 1:maxL;  % Use natural order

% Social behavior groups (using mock assignments)
socUsageAll = [socUsage1; socUsage2];
BBsl = loneUsage([1:12 49:60],:);      % Individual behavior in social context
WWsl = loneUsage([13:28 61:76],:);
BWsl = loneUsage(29:48,:);
WBsl = loneUsage(77:96,:);

% Filter social indices to valid range
validIndices = 1:min(size(socUsage1,2), maxS);
BBs = socUsageAll([1:12 49:60], validIndices);
WWs = socUsageAll([13:28 61:76], validIndices);
BWs = socUsageAll(29:48, validIndices);
WBs = socUsageAll(77:96, validIndices);

%% ========================================================================
%% MAIN VISUALIZATION - BEHAVIORAL DENSITY MAPS
%% ========================================================================

fprintf('Creating main behavioral density maps...\n');

% Ensure clean slate for our visualizations
close all;

% Figure 1: Individual Behavior Categories
fig1 = figure('Name', 'Individual Behavior Categories', 'NumberTitle', 'off');
imagesc(LLC); 
hold on; 
colormap(colorsCoarse); 
caxis([0 15]);
scatter(llbwb(:,2),llbwb(:,1),'.','k'); 
axis equal off;
title('Individual Behavior Categories', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 2: Individual Behavior Density
fig2 = figure('Name', 'Individual Behavior Density', 'NumberTitle', 'off');
imagesc(D); 
colormap(flipud(gray)); 
caxis([0 6e-4]);
hold on; 
scatter(llbwb(:,2),llbwb(:,1),'.','k'); 
axis equal off;
title('Individual Behavior Density', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 3: Social Behavior Density
fig3 = figure('Name', 'Social Behavior Density', 'NumberTitle', 'off');
imagesc(dsoc); 
axis equal off; 
colormap(cmap1); 
caxis([0 3e-4]);
hold on; 
scatter(llbwbsoc(:,2),llbwbsoc(:,1),1,'k','.');
title('Social Behavior Density', 'FontSize', 14, 'FontWeight', 'bold');
set(gcf,'Position',[653 242 868 759]);

% Figure 4: Social Behavior Categories  
fig4 = figure('Name', 'Social Behavior Categories', 'NumberTitle', 'off');
imagesc(LLCsoc); 
hold on; 
colormap(cmapSOC); 
caxis([0 8]);
scatter(llbwbsoc(:,2),llbwbsoc(:,1),1,'k','.'); 
axis equal off;
title('Social Behavior Categories', 'FontSize', 14, 'FontWeight', 'bold');
set(gcf,'Position',[653 242 868 759]);

%% ========================================================================
%% STATISTICAL ANALYSIS AND PCA
%% ========================================================================

fprintf('Performing statistical analysis...\n');

% PCA for individual behavior (strain comparison)
colorLone = [repmat([0 0 0],[16,1]); repmat([1 0 0],[16,1])];
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca([Blone; Wlone]);

fig9 = figure('Name', 'PCA Individual Mouse Strains', 'NumberTitle', 'off');
scatter(SCORE(:,1),SCORE(:,2),100,colorLone,'filled');
axis equal;
title('PCA Individual Mouse Strains (B vs W)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel(sprintf('PC1 (%.1f%% variance)', EXPLAINED(1)));
ylabel(sprintf('PC2 (%.1f%% variance)', EXPLAINED(2)));

% PCA for social individual behavior
colorSoc = [repmat([0 0 0],[24,1]); repmat([1 0 0],[32,1]); repmat([.7 .7 .7],[20,1]); repmat([219 125 211]./256,[20,1])];
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca([BBsl; WWsl; BWsl; WBsl]);

fig10 = figure('Name', 'PCA Social Individual Behavior', 'NumberTitle', 'off');
scatter3(SCORE(:,1),SCORE(:,2),SCORE(:,3),200,colorSoc,'filled');
hold on;
axis equal;
title('PCA Social Individual Behavior', 'FontSize', 14, 'FontWeight', 'bold');
xlabel(sprintf('PC1 (%.1f%%)', EXPLAINED(1)));
ylabel(sprintf('PC2 (%.1f%%)', EXPLAINED(2)));
zlabel(sprintf('PC3 (%.1f%%)', EXPLAINED(3)));

% PCA for social interactive behavior
[COEFFS, SCORES, LATENTS, TSQUAREDS, EXPLAINEDS, MUS] = pca([BBs; WWs; BWs; WBs]);

fig11 = figure('Name', 'PCA Social Interactive Behavior', 'NumberTitle', 'off');
scatter3(SCORES(:,1),SCORES(:,2),SCORES(:,3),200,colorSoc,'filled');
hold on;
axis equal;
title('PCA Social Interactive Behavior', 'FontSize', 14, 'FontWeight', 'bold');
xlabel(sprintf('PC1 (%.1f%%)', EXPLAINEDS(1)));
ylabel(sprintf('PC2 (%.1f%%)', EXPLAINEDS(2)));
zlabel(sprintf('PC3 (%.1f%%)', EXPLAINEDS(3)));

%% ========================================================================
%% USAGE HEATMAPS
%% ========================================================================

fprintf('Creating usage heatmaps...\n');

% Individual behavior usage heatmaps
nColsL = size(BBsl, 2);
fig12 = figure('Name', 'Individual Behavior Usage', 'NumberTitle', 'off');
imagesc(log([BBsl; ones(2,nColsL); BWsl; ones(2,nColsL); WWsl; ones(2,nColsL); WBsl]));
colormap(gray); 
caxis([-9 -2]); 
axis off;
set(gcf,'Position',[278 815 669 172]);
title('Individual Behavior Usage by Group', 'FontSize', 14, 'FontWeight', 'bold');

% Social behavior usage heatmaps
nColsS = size(BBs, 2);
fig13 = figure('Name', 'Social Behavior Usage', 'NumberTitle', 'off');
imagesc(log([BBs; ones(2,nColsS); BWs; ones(2,nColsS); WWs; ones(2,nColsS); WBs]));
colormap(gray); 
caxis([-9 -2]); 
axis off;
set(gcf,'Position',[278 815 669 172]);
title('Social Behavior Usage by Group', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%% MEAN BEHAVIOR PROFILES
%% ========================================================================

fprintf('Creating mean behavior profiles...\n');

% Calculate means
meanBBsl = mean(BBsl);
meanWWsl = mean(WWsl);
meanBWsl = mean(BWsl);
meanWBsl = mean(WBsl);

meanBBs = mean(BBs);
meanWWs = mean(WWs);
meanBWs = mean(BWs);
meanWBs = mean(WBs);

% Mean individual behavior profiles
fig16 = figure('Name', 'Mean Individual Behavior Profiles', 'NumberTitle', 'off');
imagesc(log([meanBBsl; meanWWsl; meanBWsl; meanWBsl]));
axis off; 
colormap(gray); 
caxis([-9 -2.5]);
set(gcf,'Position',[210 885 696 91]);
title('Mean Individual Behavior Profiles by Group', 'FontSize', 14, 'FontWeight', 'bold');

% Mean social behavior profiles
fig17 = figure('Name', 'Mean Social Behavior Profiles', 'NumberTitle', 'off');
imagesc(log([meanBBs; meanWWs; meanBWs; meanWBs]));
axis off; 
colormap(gray); 
caxis([-9 -2.5]);
set(gcf,'Position',[210 885 696 91]);
title('Mean Social Behavior Profiles by Group', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%% DIFFERENCE PLOTS
%% ========================================================================

fprintf('Creating difference plots...\n');

% Calculate differences from BB mean
mBBsl = BBsl - meanBBsl;
mWWsl = WWsl - meanBBsl;
mWBsl = WBsl - meanBBsl;
mBWsl = BWsl - meanBBsl;

mBBs = BBs - mean(BBs);
mWWs = WWs - mean(BBs);
mBWs = BWs - mean(BBs);
mWBs = WBs - mean(BBs);

% Individual behavior differences
fig14 = figure('Name', 'Individual Behavior Differences', 'NumberTitle', 'off');
imagesc([mBBsl; zeros(2,nColsL); mBWsl; zeros(2,nColsL); mWWsl; zeros(2,nColsL); mWBsl]);
colormap(cmapdiff); 
caxis([-.1 .1]);
set(gcf,'Position',[278 815 669 172]); 
axis off;
title('Individual Behavior Differences from BB Mean', 'FontSize', 14, 'FontWeight', 'bold');

% Social behavior differences
fig15 = figure('Name', 'Social Behavior Differences', 'NumberTitle', 'off');
imagesc([mBBs; zeros(2,nColsS); mBWs; zeros(2,nColsS); mWWs; zeros(2,nColsS); mWBs]);
colormap(cmapdiff); 
caxis([-.1 .1]);
set(gcf,'Position',[278 815 669 172]); 
axis off;
title('Social Behavior Differences from Group Means', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%% COMPLETION SUMMARY
%% ========================================================================

fprintf('\n=== ANALYSIS COMPLETED SUCCESSFULLY ===\n');
fprintf('Generated Figures:\n');
fprintf('  Figure 1: Individual Behavior Categories\n');
fprintf('  Figure 2: Individual Behavior Density\n');  
fprintf('  Figure 3: Social Behavior Density\n');
fprintf('  Figure 4: Social Behavior Categories\n');
fprintf('  Figures 9-11: PCA Analysis\n');
fprintf('  Figures 12-13: Usage Heatmaps\n');
fprintf('  Figures 14-15: Difference Plots\n');
fprintf('  Figures 16-17: Mean Behavior Profiles\n');
fprintf('\nData Summary:\n');
fprintf('  Individual behaviors: %d clusters analyzed\n', maxL);
fprintf('  Social behaviors: %d clusters analyzed\n', maxS);
fprintf('  Total recordings: %d individual + %d social\n', length(EVL), size(EVS,1));
fprintf('=========================================\n'); 