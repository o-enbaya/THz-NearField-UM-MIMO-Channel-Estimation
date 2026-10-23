% =========================================================================
% Script: plot_NMSE_Results.m
% Purpose: Generates publication-ready NMSE (dB) vs SNR figure
%          comparing Deep CS (20 Steps) against OMP/SOMP baselines.
%          Deep CS is plotted in Magenta to distinguish it.
% =========================================================================
clear; clc; close all; 

% 1. Load MATLAB Simulation Workspace
% CHANGE 'NMSE_Results.mat' to whatever you named your saved workspace file!
load('NMSE_Run4(using override block and edited source file).mat'); 

% 2. Input DCS Data (Already in dB)
SNR_dcs = [-20, -15, -10, -5, 0, 5, 10, 15, 20];
NMSE_dcs_dB = [-7.8301, -10.8097, -13.0024, -14.0258, -14.3457, -14.3936, -14.4444, -14.4665, -14.5358];

% 3. Define common visual properties
lw = 2;       % Line width for all lines
ms = 8;       % Marker size for all lines

% 4. Create the Figure
figure('Color', [1,1,1], 'Name', 'NMSE vs SNR');
hold on; box on; 

% 5. AVERAGE AND CONVERT MATLAB DATA TO dB
avg_NMSE_OMP_dB     = 10 * log10(mean(NMSE_OMP, [2 3]));     avg_NMSE_OMP_dB     = avg_NMSE_OMP_dB(:);
avg_NMSE_SOMP_dB    = 10 * log10(mean(NMSE_SOMP, [2 3]));    avg_NMSE_SOMP_dB    = avg_NMSE_SOMP_dB(:);
avg_NMSE_OMP_DR_dB  = 10 * log10(mean(NMSE_OMP_DR, [2 3]));  avg_NMSE_OMP_DR_dB  = avg_NMSE_OMP_DR_dB(:);
avg_NMSE_SOMP_DR_dB = 10 * log10(mean(NMSE_SOMP_DR, [2 3])); avg_NMSE_SOMP_DR_dB = avg_NMSE_SOMP_DR_dB(:);

% 6. Plot conventional algorithms 
% Red Square 
plot(SNR_dB, avg_NMSE_OMP_dB, '-s', 'Color', 'r', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'r');

% Green Diamond 
plot(SNR_dB, avg_NMSE_SOMP_dB, '-d', 'Color', 'g', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'g');

% Blue Asterisk 
plot(SNR_dB, avg_NMSE_OMP_DR_dB, '-*', 'Color', 'b', 'LineWidth', lw, ...
    'MarkerSize', ms);

% Cyan Right-Pointing Triangle 
plot(SNR_dB, avg_NMSE_SOMP_DR_dB, '->', 'Color', 'c', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'c');

% 7. Plot Proposed DCS (COLOR CHANGED TO MAGENTA)
% Using a pentagram ('p') marker to distinguish it clearly from the others.
% The color is now magenta ('m') instead of black.
plot(SNR_dcs, NMSE_dcs_dB, '-p', 'Color', 'm', 'LineWidth', lw, ...
    'MarkerSize', ms+1, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'm');

% 8. Formatting
xlabel('SNR [dB]', 'FontSize', 14);
ylabel('NMSE [dB]', 'FontSize', 14);

% Axes and Grid styling 
ax = gca;
ax.FontSize = 13;
ax.TickLabelInterpreter = 'latex';
grid on; 
grid minor;
ax.GridAlpha = 0.4;
ax.MinorGridAlpha = 0.5;
ax.MinorGridLineStyle = ':'; 
xlim([-20 20]); 
ylim([-18 15]); 

% 9. Legend (Location remains Top Right)
legend('OMP', 'SOMP', 'OMP-DR', 'SOMP-DR', 'Deep CS (Proposed)', ...
    'Location', 'northeast', 'FontSize', 12);