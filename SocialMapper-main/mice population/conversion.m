%--------------------------------------------------------------------------
% Single-CSV → MAT Converter & Visualization (Simple Automatic Orientation)
%--------------------------------------------------------------------------
% This script reads a 3D keypoint CSV, automatically determines the
% correct 3D orientation using a simple physical heuristic (body above
% feet), saves the corrected data, and visualizes the result.
% (Version 2: Fixed indexing bug)
%--------------------------------------------------------------------------


%% 1) Specify your CSV file
csv_file = "C:\Users\Runda\Downloads\init_save_data_AVG (1).mat";  % <-- CHANGE THIS

[folder,name,~] = fileparts(csv_file);

%% 2) Create "mat" subfolder next to the CSV
mat_folder = fullfile(folder,'mat');
if ~exist(mat_folder,'dir')
    mkdir(mat_folder);
end

%% 3) Define Target Joints and CSV Mapping
TARGET_RAT23_JOINTS = { ...
    'Snout','EarL','EarR','SpineF','SpineM','SpineL','TailBase', ...
    'ShoulderL','ElbowL','WristL','HandL','ShoulderR','ElbowR', ...
    'WristR','HandR','HipL','KneeL','AnkleL','FootL','HipR', ...
    'KneeR','AnkleR','FootR'};

jointMap = containers.Map( ...
    {'Snout','EarL','EarR','SpineF','SpineM','SpineL','TailBase', ...
     'ShoulderL','ElbowL','HandL','ShoulderR','ElbowR','HandR', ...
     'HipL','AnkleL','FootL','HipR','AnkleR','FootR'}, ...
    {'nose','left_earend','right_earend','neck_base','back_middle', ...
     'back_end','tail_base','front_left_thai','front_left_knee', ...
     'front_left_paw','front_right_thai','front_right_knee', ...
     'front_right_paw','back_left_thai','back_left_knee', ...
     'back_left_paw','back_right_thai','back_right_knee', ...
     'back_right_paw'} ...
);

%% 4) Read Raw CSV Data
fprintf('Reading CSV file: %s\n', name);
try
    T = readtable(csv_file);
catch ME
    error('Failed to read CSV file. Check path and format. Error: %s', ME.message);
end
numFrames = height(T);
p_raw = NaN(numFrames, 3, numel(TARGET_RAT23_JOINTS));
for i = 1:numel(TARGET_RAT23_JOINTS)
    joint = TARGET_RAT23_JOINTS{i};
    if isKey(jointMap,joint)
        src = jointMap(joint);
        xcol = [src '_x']; ycol = [src '_y']; zcol = [src '_z'];
        if all(ismember({xcol,ycol,zcol}, T.Properties.VariableNames))
            p_raw(:,:,i) = [T.(xcol), T.(ycol), T.(zcol)];
        end
    end
end

%% 5) Find Best Orientation Using "Body-Above-Feet" Heuristic
fprintf('Finding best orientation based on "body-above-feet" rule...\n');

orientations = {
    @(x,y,z) [x, y, z],    ' [X, Y, Z]';
    @(x,y,z) [x, -y, -z],  ' [X, -Y, -Z]';
    @(x,y,z) [x, z, -y],   ' [X, Z, -Y]';
    @(x,y,z) [x, -z, y],   ' [X, -Z, Y]';
    @(x,y,z) [y, x, -z],   ' [Y, X, -Z]';
    @(x,y,z) [y, -x, z],   ' [Y, -X, Z]';
    @(x,y,z) [z, y, -x],   ' [Z, Y, -X]';
    @(x,y,z) [z, -y, x],   ' [Z, -Y, X]';
};

idx_upper_body = find(contains(TARGET_RAT23_JOINTS, {'SpineF','SpineM','SpineL'}));
idx_lower_body = find(contains(TARGET_RAT23_JOINTS, {'Hand','Foot'}));
upright_scores = -inf(size(orientations, 1), 1);

% Test each orientation
% CORRECTED: Use size(orientations, 1) to get the number of rows (8)
for i = 1:size(orientations, 1)
    transform_func = orientations{i,1};
    p_test = transform_func(p_raw(:,1,:), p_raw(:,2,:), p_raw(:,3,:));

    z_upper = p_test(:,3,idx_upper_body);
    z_lower = p_test(:,3,idx_lower_body);
    
    mean_z_upper = nanmean(z_upper, 3);
    mean_z_lower = nanmean(z_lower, 3);
    
    upright_scores(i) = nanmean(mean_z_upper - mean_z_lower);
end

[~, best_idx] = max(upright_scores);
best_transform = orientations{best_idx, 1};

fprintf('-> Best orientation found:%s\n', orientations{best_idx, 2});

p1 = best_transform(p_raw(:,1,:), p_raw(:,2,:), p_raw(:,3,:));

%% 6) Compute Derived Joints (Wrist, Knee)
fprintf('Computing derived joints...\n');
tempCoords = struct();
for i = 1:numel(TARGET_RAT23_JOINTS)
    jointName = TARGET_RAT23_JOINTS{i};
    % Always create the field. NaN values will propagate correctly.
    tempCoords.(jointName) = squeeze(p1(:,:,i));
end

try
    p1(:,:,strcmp('WristL',TARGET_RAT23_JOINTS)) = 0.25*tempCoords.ElbowL + 0.75*tempCoords.HandL;
    p1(:,:,strcmp('WristR',TARGET_RAT23_JOINTS)) = 0.25*tempCoords.ElbowR + 0.75*tempCoords.HandR;
    p1(:,:,strcmp('KneeL',TARGET_RAT23_JOINTS))  = 0.5*(tempCoords.HipL + tempCoords.AnkleL);
    p1(:,:,strcmp('KneeR',TARGET_RAT23_JOINTS))  = 0.5*(tempCoords.HipR + tempCoords.AnkleR);
catch ME
    warning('Could not compute all derived joints. Check if source joints are present. Error: %s', ME.message);
end

%% 7) Save to .mat in the "mat" folder
outFile = fullfile(mat_folder, [name '.mat']);
%save(outFile, 'p1');
fprintf('Saved auto-oriented MAT file: %s\n', outFile);

%% 8) Visualize with Keypoint3DAnimator
fprintf('Launching final animation...\n');

skeleton.color = zeros(23,3);
skeleton.joints_idx = [ ...
    1 2; 1 3; 2 3; 1 4; 4 5; 5 6; 6 7; ...
    4 8; 8 9; 9 10; 10 11; ...
    4 12; 12 13; 13 14; 14 15; ...
    6 16; 16 17; 17 18; 18 19; ...
    6 20; 20 21; 21 22; 22 23];

figure('Name','Final Animation (Auto-Oriented)','NumberTitle','off', 'Position', [100 100 800 600]);
Keypoint3DAnimator(p1, skeleton, 'MarkerSize', 15, 'LineWidth', 2);
axis equal off;
view(-30, 40);
camproj perspective;
title('Full Animation with Automatically Corrected Orientation');