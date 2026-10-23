% =========================================================================
% Script: plot_NMSE_vs_Rho.m
% Purpose: Generates publication-ready NMSE vs Compression Ratio (\rho) 
%          figure, comparing Deep CS against OMP/SOMP baselines.
%          X-axis constrained to exact first/last data points.
% =========================================================================
clear; clc; close all; 

% 1. Load MATLAB Simulation Workspace
load('NMSe_RHo_SnR10.mat'); 

% 2. Load DCS CSV data 
% ASSUMPTION: Column 1 is rho, Column 2 is NMSE (dB)
dcs_data = readmatrix('DCS_NMSE_vs_Rho_SNR10.csv'); 
rho_dcs     = dcs_data(:, 1);
NMSE_dcs_dB = dcs_data(:, 2);

% 3. Calculate Compression Ratio for conventional algorithms
N_t = Q_T * Qbar_T; % Total number of transmit antennas
rho_array = sum(M_T_measMAT) / N_t;

% 4. Average and convert conventional MATLAB data to dB
avg_NMSE_OMP_dB     = 10 * log10(mean(NMSE_OMP, [2 3]));     avg_NMSE_OMP_dB     = avg_NMSE_OMP_dB(:);
avg_NMSE_SOMP_dB    = 10 * log10(mean(NMSE_SOMP, [2 3]));    avg_NMSE_SOMP_dB    = avg_NMSE_SOMP_dB(:);
avg_NMSE_OMP_DR_dB  = 10 * log10(mean(NMSE_OMP_DR, [2 3]));  avg_NMSE_OMP_DR_dB  = avg_NMSE_OMP_DR_dB(:);
avg_NMSE_SOMP_DR_dB = 10 * log10(mean(NMSE_SOMP_DR, [2 3])); avg_NMSE_SOMP_DR_dB = avg_NMSE_SOMP_DR_dB(:);

% 5. Define common visual properties
lw = 2;       % Line width
ms = 8;       % Marker size

% 6. Create the Figure
figure('Color', [1,1,1], 'Name', 'NMSE vs Compression Ratio (\rho)');
hold on; box on; 

% 7. Plot conventional algorithms
% Red Square 
plot(rho_array, avg_NMSE_OMP_dB, '-s', 'Color', 'r', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'r');

% Green Diamond 
plot(rho_array, avg_NMSE_SOMP_dB, '-d', 'Color', 'g', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'g');

% Blue Asterisk 
plot(rho_array, avg_NMSE_OMP_DR_dB, '-*', 'Color', 'b', 'LineWidth', lw, ...
    'MarkerSize', ms);

% Cyan Right-Pointing Triangle 
plot(rho_array, avg_NMSE_SOMP_DR_dB, '->', 'Color', 'c', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'c');

% 8. Plot Proposed DCS (Magenta Pentagram)
plot(rho_dcs, NMSE_dcs_dB, '-p', 'Color', 'm', 'LineWidth', lw, ...
    'MarkerSize', ms+1, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'm');

% 9. Formatting
xlabel('Compression Ratio ($\rho = N_p / N_t$)', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('NMSE [dB]', 'FontSize', 14, 'Interpreter', 'latex');
title('\textbf{NMSE vs. Compression Ratio ($\rho$)}', 'Interpreter', 'latex', 'FontSize', 15);

% Axes and Grid styling 
ax = gca;
ax.FontSize = 13;
ax.TickLabelInterpreter = 'latex';
grid on; 
grid minor;
ax.GridAlpha = 0.4;
ax.MinorGridAlpha = 0.5;
ax.MinorGridLineStyle = ':'; 

% --- CHANGED: Strict X-axis limits, padded Y-axis limits ---
% 1. Set X-axis exactly to the first and last data points
min_rho = min([rho_array(:); rho_dcs(:)]);
max_rho = max([rho_array(:); rho_dcs(:)]);
xlim([min_rho, max_rho]);

% 2. Add padding ONLY to the Y-axis so the top/bottom markers don't get clipped
y_all_data = [avg_NMSE_OMP_dB(:); avg_NMSE_SOMP_dB(:); avg_NMSE_OMP_DR_dB(:); avg_NMSE_SOMP_DR_dB(:); NMSE_dcs_dB(:)];
min_y = min(y_all_data);
max_y = max(y_all_data);
y_range = max_y - min_y;

y_padding = 0.05 * y_range; % 5% padding
ylim([min_y - y_padding, max_y + y_padding]);

% 10. Legend
legend('OMP', 'SOMP', 'OMP-DR', 'SOMP-DR', 'Deep CS (Proposed)', ...
    'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 12);