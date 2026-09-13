%% Dynamic Project Path Setup
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
projRoot = fileparts(thisDir);
if exist(fullfile(projRoot, 'setup_paths.m'), 'file')
    run(fullfile(projRoot, 'setup_paths.m'));
end

%% =========================================================================
% RUNTIME PLOTTING SCRIPT
% Standalone plotting script for OMP / SOMP / OMP-DR / SOMP-DR / DCS
% =========================================================================
clc; clear; close all;

%% -----------------------------
% 1) Runtime data
% -----------------------------
% SNR points
SNR_dB = [-20 -15 -10 -5 0 5 10 15 20];

% Classical methods
% These are your measured average runtimes per full realization
runtime_OMP_avg     = 6.450301;
runtime_SOMP_avg    = 0.766663;
runtime_OMP_DR_avg  = 7.527417;
runtime_SOMP_DR_avg = 0.745208;

% Since your classical runtime summary was reported as one average value,
% we draw them as constant lines across SNR
runtime_OMP_snr     = runtime_OMP_avg     * ones(size(SNR_dB));
runtime_SOMP_snr    = runtime_SOMP_avg    * ones(size(SNR_dB));
runtime_OMP_DR_snr  = runtime_OMP_DR_avg  * ones(size(SNR_dB));
runtime_SOMP_DR_snr = runtime_SOMP_DR_avg * ones(size(SNR_dB));

% DCS runtime per sample from Python
runtime_DCS_snr = [0.41225653 0.35450487 0.35327713 0.35411432 0.35690381 ...
                   0.35955017 0.36082901 0.36133298 0.36205174];

runtime_DCS_avg = mean(runtime_DCS_snr);

%% -----------------------------
% 2) Runtime vs SNR
% -----------------------------
figure('Color',[1 1 1]);
plot(SNR_dB, runtime_OMP_snr, 'ko-','LineWidth',2,'MarkerSize',8); hold on;
plot(SNR_dB, runtime_SOMP_snr, 'bx-','LineWidth',2,'MarkerSize',8);
plot(SNR_dB, runtime_OMP_DR_snr, 'ms-','LineWidth',2,'MarkerSize',8);
plot(SNR_dB, runtime_SOMP_DR_snr, 'rd-','LineWidth',2,'MarkerSize',8);
plot(SNR_dB, runtime_DCS_snr, 'g^-','LineWidth',2,'MarkerSize',8);

xlabel('SNR (dB)','FontSize',13);
ylabel('Runtime per Channel Estimate (s)','FontSize',13);
title('Measured Runtime vs SNR','FontSize',13);
legend('OMP','SOMP','OMP-DR','SOMP-DR','DCS','Location','best');
grid on;
box off;

%% -----------------------------
% 3) Average runtime bar chart
% -----------------------------
runtime_bar = [runtime_OMP_avg, runtime_SOMP_avg, runtime_OMP_DR_avg, runtime_SOMP_DR_avg, runtime_DCS_avg];
labels = {'OMP','SOMP','OMP-DR','SOMP-DR','DCS'};

figure('Color',[1 1 1]);
bar(runtime_bar);
set(gca,'XTick',1:numel(labels),'XTickLabel',labels);
ylabel('Average Runtime per Channel Estimate (s)','FontSize',13);
title('Average Runtime Comparison','FontSize',13);
grid on;
box off;

for i = 1:numel(runtime_bar)
    text(i, runtime_bar(i) + 0.08, sprintf('%.3f', runtime_bar(i)), ...
        'HorizontalAlignment','center','FontSize',11);
end

%% -----------------------------
% 4) Normalized runtime bar chart
% Normalize to DCS
% -----------------------------
runtime_norm_dcs = runtime_bar / runtime_DCS_avg;

figure('Color',[1 1 1]);
bar(runtime_norm_dcs);
set(gca,'XTick',1:numel(labels),'XTickLabel',labels);
ylabel('Normalized Runtime (relative to DCS)','FontSize',13);
title('Normalized Runtime Comparison','FontSize',13);
grid on;
box off;

for i = 1:numel(runtime_norm_dcs)
    text(i, runtime_norm_dcs(i) + 0.15, sprintf('%.2f', runtime_norm_dcs(i)), ...
        'HorizontalAlignment','center','FontSize',11);
end

%% -----------------------------
% 5) Zoomed runtime bar chart for faster methods
% -----------------------------
runtime_fast = [runtime_SOMP_avg, runtime_SOMP_DR_avg, runtime_DCS_avg];
labels_fast = {'SOMP','SOMP-DR','DCS'};

figure('Color',[1 1 1]);
bar(runtime_fast);
set(gca,'XTick',1:numel(labels_fast),'XTickLabel',labels_fast);
ylabel('Average Runtime (s)','FontSize',13);
title('Zoomed Runtime Comparison for Faster Methods','FontSize',13);
grid on;
box off;

for i = 1:numel(runtime_fast)
    text(i, runtime_fast(i) + 0.015, sprintf('%.3f', runtime_fast(i)), ...
        'HorizontalAlignment','center','FontSize',11);
end

%% -----------------------------
% 6) Optional log-scale runtime bar chart
% -----------------------------
figure('Color',[1 1 1]);
bar(runtime_bar);
set(gca,'XTick',1:numel(labels),'XTickLabel',labels);
set(gca,'YScale','log');
ylabel('Average Runtime (s, log scale)','FontSize',13);
title('Average Runtime Comparison (Log Scale)','FontSize',13);
grid on;
box off;

%% -----------------------------
% 7) Print summary
% -----------------------------
fprintf('\n===== Runtime Summary =====\n');
fprintf('OMP avg runtime     = %.6f s\n', runtime_OMP_avg);
fprintf('SOMP avg runtime    = %.6f s\n', runtime_SOMP_avg);
fprintf('OMP-DR avg runtime  = %.6f s\n', runtime_OMP_DR_avg);
fprintf('SOMP-DR avg runtime = %.6f s\n', runtime_SOMP_DR_avg);
fprintf('DCS avg runtime     = %.6f s\n', runtime_DCS_avg);

fprintf('\n===== Runtime Relative to DCS =====\n');
fprintf('OMP     = %.2f x DCS\n', runtime_OMP_avg / runtime_DCS_avg);
fprintf('SOMP    = %.2f x DCS\n', runtime_SOMP_avg / runtime_DCS_avg);
fprintf('OMP-DR  = %.2f x DCS\n', runtime_OMP_DR_avg / runtime_DCS_avg);
fprintf('SOMP-DR = %.2f x DCS\n', runtime_SOMP_DR_avg / runtime_DCS_avg);
fprintf('DCS     = 1.00 x DCS\n');