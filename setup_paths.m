% =========================================================================
% THz UM-MIMO Near-Field Channel Estimation - Path Initialization
% =========================================================================
% Run this script once upon opening MATLAB to add all project subdirectories
% to your MATLAB path.
% =========================================================================

projRoot = fileparts(mfilename('fullpath'));
if isempty(projRoot), projRoot = pwd; end

addpath(genpath(fullfile(projRoot, 'channel_simulator')));
addpath(genpath(fullfile(projRoot, 'algorithms')));
addpath(genpath(fullfile(projRoot, 'simulations')));
addpath(genpath(fullfile(projRoot, 'plotting')));
addpath(genpath(fullfile(projRoot, 'results')));

disp('=================================================================');
disp('THz UM-MIMO Near-Field Channel Estimation Environment Initialized');
disp('  - channel_simulator/ (TeraMIMO, Molecular Absorption, Utilities)');
disp('  - algorithms/        (OMP, SOMP, OMP-DR, DR Extraction)');
disp('  - simulations/       (Thesis simulation runners)');
disp('  - plotting/          (Camera-ready figure plotters)');
disp('  - results/           (Data workspaces, CSVs, Figures)');
disp('=================================================================');
