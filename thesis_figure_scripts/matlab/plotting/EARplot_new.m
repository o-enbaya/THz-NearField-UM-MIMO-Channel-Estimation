% =========================================================================
% Script: plot_EAR_Results.m
% Purpose: Generates publication-ready EAR vs Measurements figures 
%          comparing Deep CS (Proposed) against OMP/SOMP baselines.
% =========================================================================
clear; clc; close all; 

% 1. Load MATLAB Simulation Workspace
% CHANGE 'EAR_Results.mat' to whatever you named your saved workspace file!
load('EAR_Run2 (good).mat'); 

% 2. Load DCS CSV data 
dcs_data = readmatrix('DCS_EAR_vs_Measurements_SNR24_dl100.csv'); 
Np_dcs   = dcs_data(:, 1);
EAR_inf  = dcs_data(:, 2);
EAR_512  = dcs_data(:, 4);

% X-axis for conventional MATLAB algorithms
M_T_meas = sum(M_T_measMAT);

% 3. Define common visual properties
lw = 2;       % Line width for algorithms
ms = 8;       % Marker size

% =========================================================================
% FIGURE 1: T_coh = Infinity 
% =========================================================================
figure('Color', [1,1,1], 'Name', 'T_coh = Infinity');
hold on; box on; 

% Perfect CSI (Solid thick black line to match P-CSI style)
plot(M_T_meas, R_eff_PCI, '-k', 'LineWidth', 2.5);

% Conventional algorithms 
plot(M_T_meas, R_eff_OMP(3,:), '-s', 'Color', 'r', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'r');

plot(M_T_meas, R_eff_SOMP(3,:), '-d', 'Color', 'g', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'g');

plot(M_T_meas, R_eff_OMP_DR(3,:), '-*', 'Color', 'b', 'LineWidth', lw, ...
    'MarkerSize', ms);

plot(M_T_meas, R_eff_SOMP_DR(3,:), '->', 'Color', 'c', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'c');

% Proposed DCS (Magenta Pentagram)
plot(Np_dcs, EAR_inf, '-p', 'Color', 'm', 'LineWidth', lw, ...
    'MarkerSize', ms+1, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'm');

% Formatting
xlabel('Number of Measurements ($M_{\mathrm{T}}^{\textrm{tr}}$)', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('Effective Achievable Rate ($R_{\textrm{eff}}$) [b/s/Hz]', 'FontSize', 14, 'Interpreter', 'latex');
title('\textbf{Spectral Efficiency vs. Measurements ($T_{\textrm{coh}} = \infty$)}', 'Interpreter', 'latex', 'FontSize', 15);

xlim([10 160]); ylim([5 23]);
ax = gca;
ax.FontSize = 13;
ax.TickLabelInterpreter = 'latex';
grid on; grid minor;
ax.GridAlpha = 0.4;
ax.MinorGridAlpha = 0.5;
ax.MinorGridLineStyle = ':'; 

legend('Perfect CSI', 'OMP', 'SOMP', 'OMP-DR', 'SOMP-DR', 'Deep CS (Proposed)', ...
    'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 12);


% =========================================================================
% FIGURE 2: T_coh = 512 
% =========================================================================
figure('Color', [1,1,1], 'Name', 'T_coh = 512');
hold on; box on;

% Perfect CSI
plot(M_T_meas, R_eff_PCI, '-k', 'LineWidth', 2.5);

% Conventional algorithms
plot(M_T_meas, R_eff_OMP(1,:), '-s', 'Color', 'r', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'r');

plot(M_T_meas, R_eff_SOMP(1,:), '-d', 'Color', 'g', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'g');

plot(M_T_meas, R_eff_OMP_DR(1,:), '-*', 'Color', 'b', 'LineWidth', lw, ...
    'MarkerSize', ms);

plot(M_T_meas, R_eff_SOMP_DR(1,:), '->', 'Color', 'c', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'c');

% Proposed DCS
plot(Np_dcs, EAR_512, '-p', 'Color', 'm', 'LineWidth', lw, ...
    'MarkerSize', ms+1, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'm');

% Formatting
xlabel('Number of Measurements ($M_{\mathrm{T}}^{\textrm{tr}}$)', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('Effective Achievable Rate ($R_{\textrm{eff}}$) [b/s/Hz]', 'FontSize', 14, 'Interpreter', 'latex');
title('\textbf{Spectral Efficiency vs. Measurements ($T_{\textrm{coh}} = 512$)}', 'Interpreter', 'latex', 'FontSize', 15);

xlim([10 160]); ylim([5 23]);
ax = gca;
ax.FontSize = 13;
ax.TickLabelInterpreter = 'latex';
grid on; grid minor;
ax.GridAlpha = 0.4;
ax.MinorGridAlpha = 0.5;
ax.MinorGridLineStyle = ':'; 

legend('Perfect CSI', 'OMP', 'SOMP', 'OMP-DR', 'SOMP-DR', 'Deep CS (Proposed)', ...
    'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 12);