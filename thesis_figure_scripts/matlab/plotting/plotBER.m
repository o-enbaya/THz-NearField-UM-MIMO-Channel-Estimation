% =========================================================================
% Script: plot_BER_Final_NoFlatTail.m
% Purpose: BER vs SNR plot with:
%          - classical curve styling from the first script
%          - log-scale BER axis
%          - Perfect CSI and Deep CS shown only up to 10 dB
% =========================================================================
clear; clc; close all;

%% 1. SNR Vector
SNR_dB = -20:5:30;

%% 2. Exact extracted data from BERns1E100.fig
BER_OMP = [0.500131789, 0.499616613, 0.500256589, 0.493111022, 0.452706669, ...
           0.371052316, 0.229630591, 0.0916922923, 0.0141643371, ...
           9.68450479e-05, 3.09504792e-05];

BER_SOMP = [0.500112819, 0.498775958, 0.499707468, 0.482706669, 0.430260583, ...
            0.358100040, 0.211927915, 0.0816224042, 0.0118360623, ...
            8.18690096e-05, 1e-8];

BER_OMP_DR = [0.49982628, 0.49994908, 0.50037340, 0.49517173, 0.45106230, ...
              0.36327776, 0.21272664, 0.07847344, 0.01504692, ...
              0.00202975, 0.00062001];

BER_SOMP_DR = [0.49975539, 0.49937999, 0.50069089, 0.48118610, 0.42228834, ...
               0.34079273, 0.18702576, 0.06673822, 0.01309305, ...
               0.00066893, 1e-8];

BER_Perf_CSI = [3.369778e-01, 2.428135e-01, 1.297754e-01, 4.763478e-02, ...
                5.785743e-03, 5.491214e-05, 0, 0, 0, 0, 0];

BER_Deep_CS = [3.395607e-01, 2.436621e-01, 1.302516e-01, 4.762780e-02, ...
               6.125200e-03, 7.288338e-05, 0, 0, 0, 0, 0];

%% 3. Replace zeros only for log plotting
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
figure('Color',[1 1 1], 'Name','BER vs SNR Final');
hold on;
box on;

% --- Classical Results: styling from first script ---
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

% --- Perfect CSI and Deep CS only up to 10 dB ---
idx_upto_10 = SNR_dB <= 10;

semilogy(SNR_dB(idx_upto_10), BER_Perf_CSI_plot(idx_upto_10), '-k', ...
    'LineWidth', 2.5);

semilogy(SNR_dB(idx_upto_10), BER_Deep_CS_plot(idx_upto_10), '-p', ...
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

xlim([-20 30]);
ylim([1e-6 1]);

legend('OMP', 'SOMP', 'OMP-DR', 'SOMP-DR', 'Perfect CSI', 'Deep CS (Proposed)', ...
    'Location', 'southwest', 'Interpreter', 'latex');