%% Dynamic Project Path Setup
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
projRoot = fileparts(thisDir);
if exist(fullfile(projRoot, 'setup_paths.m'), 'file')
    run(fullfile(projRoot, 'setup_paths.m'));
end

% =========================================================================
% Script: plot_EAR_Results.m
% Thesis Reference: Chapter 4, Section 4.4.2, Figure 4.3
% Generates publication-ready Effective Achievable Rate (EAR) vs Measurements
% (a) Figure 4.3(a): Infinite Coherence Time
% (b) Figure 4.3(b): Finite Coherence Time (T_coh = 512 symbols)
% =========================================================================
clear; clc; close all;

% 1. Load MATLAB Simulation Workspace
mat_candidates = {
    'EAR_Run2 (good).mat', ...
    fullfile('data', 'EAR_Run2 (good).mat'), ...
    fullfile('..', 'Results', 'data', 'EAR_Run2 (good).mat'), ...
    fullfile('Results', 'data', 'EAR_Run2 (good).mat')
};

loaded = false;
for i = 1:numel(mat_candidates)
    if exist(mat_candidates{i}, 'file')
        fprintf('Loading EAR simulation workspace: %s\n', mat_candidates{i});
        load(mat_candidates{i});
        loaded = true;
        break;
    end
end

if ~loaded
    error('EAR simulation workspace not found.');
end

% 2. Load DCS CSV Data
csv_candidates = {
    'DCS_EAR_vs_Measurements_SNR24_dl100.csv', ...
    fullfile('..', 'python_dcs', 'data', 'DCS_EAR_vs_Measurements_SNR24_dl100.csv'), ...
    fullfile('python_dcs', 'data', 'DCS_EAR_vs_Measurements_SNR24_dl100.csv')
};

dcs_loaded = false;
for i = 1:numel(csv_candidates)
    if exist(csv_candidates{i}, 'file')
        dcs_data = readmatrix(csv_candidates{i});
        Np_dcs   = dcs_data(:, 1);
        EAR_inf  = dcs_data(:, 2);
        EAR_512  = dcs_data(:, 4);
        dcs_loaded = true;
        break;
    end
end

if ~dcs_loaded
    error('DCS EAR CSV file not found.');
end

M_T_meas = sum(M_T_measMAT);
lw = 2; lw_dcs = 3; ms = 8;
color_OMP     = 'k';                        
color_SOMP    = 'b';                        
color_OMP_DR  = 'm';                        
color_SOMP_DR = 'r';                        
color_DCS     = [0.8500 0.3250 0.0980];     

% ---------------------------------------------------------
% Figure 4.3(a): Infinite Coherence Time
% ---------------------------------------------------------
figure('Color', [1 1 1], 'Name', 'Figure 4.3(a) - EAR Infinite Coherence', 'Position', [100 100 700 500]);
plot(M_T_meas, mean(R_eff_PCI_Inf, [2 3]), 'g+-', 'LineWidth', 2.5, 'MarkerSize', ms); hold on;
plot(M_T_meas, mean(R_eff_OMP_Inf, [2 3]), 'ko-', 'LineWidth', lw, 'MarkerSize', ms);
plot(M_T_meas, mean(R_eff_SOMP_Inf, [2 3]), 'bx-', 'LineWidth', lw, 'MarkerSize', ms);
plot(M_T_meas, mean(R_eff_OMP_DR_Inf, [2 3]), 'ms-', 'LineWidth', lw, 'MarkerSize', ms);
plot(M_T_meas, mean(R_eff_SOMP_DR_Inf, [2 3]), 'rd-', 'LineWidth', lw, 'MarkerSize', ms);
plot(Np_dcs, EAR_inf, 'p-', 'Color', color_DCS, 'LineWidth', lw_dcs, ...
     'MarkerFaceColor', color_DCS, 'MarkerSize', 10);

xlabel('Number of Pilot Measurements ($M_{\mathrm{T}}^{\mathrm{tr}}$)', 'FontSize', 13, 'Interpreter', 'latex');
ylabel('Effective Achievable Rate [bits/s/Hz]', 'FontSize', 13, 'Interpreter', 'latex');
title('{\bf Figure 4.3(a):} Effective Achievable Rate ($T_{\mathrm{coh}} = \infty$)', 'FontSize', 14, 'Interpreter', 'latex');
legend('Perfect CSI', 'OMP', 'SOMP', 'OMP-DR', 'SOMP-DR', 'Deep CS (Proposed)', ...
       'Location', 'southeast', 'FontSize', 10, 'Interpreter', 'latex');
grid on; box on;
ax = gca; ax.TickLabelInterpreter = 'latex'; ax.FontSize = 12;
saveas(gcf, fullfile(fileparts(mfilename('fullpath')), 'Fig4_3a_EAR_Infinite.png'));

% ---------------------------------------------------------
% Figure 4.3(b): Finite Coherence Time (T_coh = 512 symbols)
% ---------------------------------------------------------
figure('Color', [1 1 1], 'Name', 'Figure 4.3(b) - EAR Finite Coherence (512)', 'Position', [150 150 700 500]);
plot(M_T_meas, mean(R_eff_PCI_Tcoh1, [2 3]), 'g+-', 'LineWidth', 2.5, 'MarkerSize', ms); hold on;
plot(M_T_meas, mean(R_eff_OMP_Tcoh1, [2 3]), 'ko-', 'LineWidth', lw, 'MarkerSize', ms);
plot(M_T_meas, mean(R_eff_SOMP_Tcoh1, [2 3]), 'bx-', 'LineWidth', lw, 'MarkerSize', ms);
plot(M_T_meas, mean(R_eff_OMP_DR_Tcoh1, [2 3]), 'ms-', 'LineWidth', lw, 'MarkerSize', ms);
plot(M_T_meas, mean(R_eff_SOMP_DR_Tcoh1, [2 3]), 'rd-', 'LineWidth', lw, 'MarkerSize', ms);
plot(Np_dcs, EAR_512, 'p-', 'Color', color_DCS, 'LineWidth', lw_dcs, ...
     'MarkerFaceColor', color_DCS, 'MarkerSize', 10);

xlabel('Number of Pilot Measurements ($M_{\mathrm{T}}^{\mathrm{tr}}$)', 'FontSize', 13, 'Interpreter', 'latex');
ylabel('Effective Achievable Rate [bits/s/Hz]', 'FontSize', 13, 'Interpreter', 'latex');
title('{\bf Figure 4.3(b):} Effective Achievable Rate ($T_{\mathrm{coh}} = 512$ symbols)', 'FontSize', 14, 'Interpreter', 'latex');
legend('Perfect CSI', 'OMP', 'SOMP', 'OMP-DR', 'SOMP-DR', 'Deep CS (Proposed)', ...
       'Location', 'southwest', 'FontSize', 10, 'Interpreter', 'latex');
grid on; box on;
ax = gca; ax.TickLabelInterpreter = 'latex'; ax.FontSize = 12;
saveas(gcf, fullfile(fileparts(mfilename('fullpath')), 'Fig4_3b_EAR_512.png'));
fprintf('Figure 4.3 saved successfully.\n');
