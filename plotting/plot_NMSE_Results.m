%% Dynamic Project Path Setup
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
projRoot = fileparts(thisDir);
if exist(fullfile(projRoot, 'setup_paths.m'), 'file')
    run(fullfile(projRoot, 'setup_paths.m'));
end

% =========================================================================
% Script: plot_NMSE_Results.m
% Thesis Reference: Chapter 4, Section 4.4.1, Figure 4.2
% Generates publication-ready NMSE (dB) vs Transmit SNR (-20 dB to 20 dB)
% Compares: OMP, SOMP, OMP-DR, SOMP-DR, and Proposed Deep CS (DCS)
% =========================================================================
clear; clc; close all;

% 1. Load MATLAB Simulation Workspace
mat_candidates = {
    'NMSE_Run3(-20dbto30).mat', ...
    fullfile('data', 'NMSE_Run3(-20dbto30).mat'), ...
    fullfile('..', 'Results', 'data', 'NMSE_Run3(-20dbto30).mat'), ...
    fullfile('Results', 'data', 'NMSE_Run3(-20dbto30).mat')
};

loaded = false;
for i = 1:numel(mat_candidates)
    if exist(mat_candidates{i}, 'file')
        fprintf('Loading NMSE simulation workspace: %s\n', mat_candidates{i});
        load(mat_candidates{i});
        loaded = true;
        break;
    end
end

if ~loaded
    error('NMSE simulation workspace not found. Please verify Results/data/NMSE_Run3(-20dbto30).mat');
end

% 2. DCS Benchmark Data (Thesis Table 4.2 / Section 4.4.1)
SNR_dcs     = [-20, -15, -10, -5, 0, 5, 10, 15, 20];
NMSE_dcs_dB = [-7.8301, -10.8097, -13.0024, -14.0258, -14.3457, -14.3936, -14.4444, -14.4665, -14.5358];

% 3. Define common visual properties
lw = 2; lw_dcs = 3; ms = 8;
color_OMP     = 'k';                        
color_SOMP    = 'b';                        
color_OMP_DR  = 'm';                        
color_SOMP_DR = 'r';                        
color_DCS     = [0.8500 0.3250 0.0980];     

% 4. Render Figure
figure('Color', [1 1 1], 'Name', 'Figure 4.2 - NMSE vs SNR', 'Position', [100 100 750 550]);

if exist('NMSE_OMP', 'var') && ndims(NMSE_OMP) >= 2
    plot(SNR_dB, 10*log10(mean(NMSE_OMP, [2 3])), 'k+--', 'LineWidth', lw, 'MarkerSize', ms); hold on;
    plot(SNR_dB, 10*log10(mean(NMSE_SOMP, [2 3])), 'bx--', 'LineWidth', lw, 'MarkerSize', ms);
    plot(SNR_dB, 10*log10(mean(NMSE_OMP_DR, [2 3])), 'mo--', 'LineWidth', lw, 'MarkerSize', ms);
    plot(SNR_dB, 10*log10(mean(NMSE_SOMP_DR, [2 3])), 'r*--', 'LineWidth', lw, 'MarkerSize', ms);
else
    plot(SNR_dB, NMSE_OMP, 'k+--', 'LineWidth', lw, 'MarkerSize', ms); hold on;
    plot(SNR_dB, NMSE_SOMP, 'bx--', 'LineWidth', lw, 'MarkerSize', ms);
    plot(SNR_dB, NMSE_OMP_DR, 'mo--', 'LineWidth', lw, 'MarkerSize', ms);
    plot(SNR_dB, NMSE_SOMP_DR, 'r*--', 'LineWidth', lw, 'MarkerSize', ms);
end

% Overlay DCS curve
plot(SNR_dcs, NMSE_dcs_dB, 'p-', 'Color', color_DCS, 'LineWidth', lw_dcs, ...
     'MarkerFaceColor', color_DCS, 'MarkerSize', 10);

xlabel('Transmit SNR [dB]', 'FontSize', 13, 'Interpreter', 'latex');
ylabel('NMSE [dB]', 'FontSize', 13, 'Interpreter', 'latex');
title('{\bf Figure 4.2:} Normalized Mean Squared Error vs. Transmit SNR', 'FontSize', 14, 'Interpreter', 'latex');

lgd = legend('OMP', 'SOMP', 'OMP-DR', 'SOMP-DR', 'Deep CS (Proposed)', ...
             'Location', 'southwest', 'FontSize', 11, 'Interpreter', 'latex');
grid on; box on;
xlim([-20 20]);
ax = gca; ax.TickLabelInterpreter = 'latex'; ax.FontSize = 12;

saveas(gcf, fullfile(fileparts(mfilename('fullpath')), 'NMSE_vs_SNR_CameraReady.png'));
fprintf('Figure 4.2 saved successfully.\n');
