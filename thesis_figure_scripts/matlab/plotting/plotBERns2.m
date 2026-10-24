% =========================================================================
% Script: plot_BER_Ns2_Final.m
% Purpose: Standalone BER vs SNR plot for Ns = 2
%          - Uses exact extracted data from BERNS2pc.fig
%          - Uses first-script curve styling
%          - Uses Ns=2 axis settings
% =========================================================================
clear; clc; close all;

%% 1. SNR Vector
SNR_dB = -20:5:30;

%% 2. Exact extracted Ns=2 data
BER_OMP = [ ...
    0.50060903, 0.49996506, 0.49735423, 0.48073083, 0.44677516, ...
    0.38682109, 0.32654752, 0.19335563, 0.19185803, 0.15591054, 0.10835663];

BER_SOMP = [ ...
    0.49984026, 0.50099840, 0.49311601, 0.46394768, 0.42814497, ...
    0.37450579, 0.31961861, 0.18387580, 0.19871206, 0.15122804, 0.10387380];

BER_OMP_DR = [ ...
    0.50088359, 0.50066893, 0.49839756, 0.48513878, 0.45335463, ...
    0.38595248, 0.28290735, 0.18764976, 0.19310603, 0.14017073, 0.08068590];

BER_SOMP_DR = [ ...
    0.49864217, 0.49912141, 0.49529752, 0.46576977, 0.43653155, ...
    0.38785942, 0.27359225, 0.18283746, 0.19008087, 0.14004593, 0.07463059];

BER_Perf_CSI = [ ...
    0.48509830, 0.47383790, 0.44831720, 0.40320340, 0.33674320, ...
    0.26070390, 0.18445740, 0.12295380, 0.06613069, 0.01975938, 0.00102087];

BER_Deep_CS = [ ...
    0.48623500, 0.47415380, 0.45015220, 0.40661190, 0.34222290, ...
    0.26961510, 0.19786140, 0.13946190, 0.08037140, 0.02783147, 0.00189097];

%% 3. Log-plot protection
plot_floor = 1e-6;

BER_OMP_plot      = max(BER_OMP,      plot_floor);
BER_SOMP_plot     = max(BER_SOMP,     plot_floor);
BER_OMP_DR_plot   = max(BER_OMP_DR,   plot_floor);
BER_SOMP_DR_plot  = max(BER_SOMP_DR,  plot_floor);
BER_Perf_CSI_plot = max(BER_Perf_CSI, plot_floor);
BER_Deep_CS_plot  = max(BER_Deep_CS,  plot_floor);

%% 4. Visual properties
lw = 2;
ms = 8;

%% 5. Create figure
figure('color',[1,1,1], 'Name', 'BER vs SNR Final');
hold on; box on;

% --- Classical Results: first-script styling ---
semilogy(SNR_dB, BER_OMP_plot, '-s', ...
    'Color', 'r', 'LineWidth', lw, ...
    'MarkerSize', ms, ...
    'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'r');

semilogy(SNR_dB, BER_SOMP_plot, '-d', ...
    'Color', 'g', 'LineWidth', lw, ...
    'MarkerSize', ms, ...
    'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'g');

semilogy(SNR_dB, BER_OMP_DR_plot, '-*', ...
    'Color', 'b', 'LineWidth', lw, ...
    'MarkerSize', ms);

semilogy(SNR_dB, BER_SOMP_DR_plot, '->', ...
    'Color', 'c', 'LineWidth', lw, ...
    'MarkerSize', ms, ...
    'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'c');

% --- Perfect CSI and Deep CS ---
semilogy(SNR_dB, BER_Perf_CSI_plot, '-k', ...
    'LineWidth', 2.5);

semilogy(SNR_dB, BER_Deep_CS_plot, '-p', ...
    'Color', 'm', 'LineWidth', lw, ...
    'MarkerSize', ms+2, ...
    'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'm');

%% 6. Formatting
grid on;
grid minor;

xlabel('Signal-to-Noise Ratio (SNR) [dB]', 'FontSize', 12, 'Interpreter', 'latex');
ylabel('Bit Error Rate (BER)', 'FontSize', 12, 'Interpreter', 'latex');


ax = gca;
ax.YScale = 'log';
ax.FontSize = 11;
ax.TickLabelInterpreter = 'latex';
ax.GridAlpha = 0.4;
ax.MinorGridAlpha = 0.5;
ax.MinorGridLineStyle = ':';

ylim([1e-6 1]);
xlim([-20 30]);

legend('OMP', 'SOMP', 'OMP-DR', 'SOMP-DR', 'Perfect CSI', 'Deep CS (Proposed)', ...
    'Location', 'southwest', 'Interpreter', 'latex');