% =========================================================================
% Script: plot_BarCharts.m
% Purpose: Recreates the Inference Complexity and Reduction bar charts
% =========================================================================
clear; clc; close all;

%% ---------------------------------------------------------
% Figure 1: Inference Complexity Relative to DCS
% ---------------------------------------------------------
figure('Color', [1 1 1], 'Name', 'Inference Complexity Relative to DCS');

% Data
algorithms = {'DCS', 'OMP', 'SOMP', 'OMP-DR', 'SOMP-DR'};
complexity_vals = [1.00, 5.12, 5.13, 5.01, 5.07];

% Create Bar Chart
b1 = bar(complexity_vals, 'FaceColor', [0 0.4470 0.7410], 'EdgeColor', 'k');

% Formatting
set(gca, 'XTickLabel', algorithms, 'FontSize', 12);
ylabel('Normalized Complexity (relative to DCS)', 'FontSize', 14);


% Add value labels on top of bars
for i = 1:length(complexity_vals)
    text(i, complexity_vals(i), sprintf('%.2f', complexity_vals(i)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 12);
end

% Grid and Axes styling
box off;
set(gca, 'YGrid', 'on', 'XGrid', 'off', 'GridAlpha', 0.3);
ylim([0 6]); % Give headroom for text labels


%% ---------------------------------------------------------
% Figure 2: Complexity Reduction Achieved by DR
% ---------------------------------------------------------
figure('Color', [1 1 1], 'Name', 'Complexity Reduction Achieved by DR');

% Data
reduction_labels = {'OMP-DR vs OMP', 'SOMP-DR vs SOMP'};
reduction_vals = [2.06, 1.13];

% Create Bar Chart (Slightly narrower bars for 2-item chart looks better)
b2 = bar(reduction_vals, 'FaceColor', [0 0.4470 0.7410], 'EdgeColor', 'k', 'BarWidth', 0.6);

% Formatting
set(gca, 'XTickLabel', reduction_labels, 'FontSize', 12);
ylabel('Complexity Reduction (%)', 'FontSize', 14);


% Add value labels with '%' sign on top of bars
for i = 1:length(reduction_vals)
    text(i, reduction_vals(i), sprintf('%.2f%%', reduction_vals(i)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 12);
end

% Grid and Axes styling
box off;
set(gca, 'YGrid', 'on', 'XGrid', 'off', 'GridAlpha', 0.3);
ylim([0 2.5]); % Match Y-axis limits from screenshot