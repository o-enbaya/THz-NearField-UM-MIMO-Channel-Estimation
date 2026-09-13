%% Dynamic Project Path Setup
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
projRoot = fileparts(thisDir);
if exist(fullfile(projRoot, 'setup_paths.m'), 'file')
    run(fullfile(projRoot, 'setup_paths.m'));
end

% =========================================================================
% Script: plot_NMSE_vs_Rho.m
% Thesis Reference: Chapter 4, Section 4.4.3, Figure 4.4
% Purpose: Generates publication-ready NMSE vs. Compression Ratio (rho = Np/Nt)
%          at fixed Transmit SNR = 10 dB.
% =========================================================================
clear; clc; close all;

% 1. Load MATLAB Simulation Workspace
mat_candidates = {
    'NMSe_RHo_SnR10.mat', ...
    fullfile('data', 'NMSe_RHo_SnR10.mat'), ...
    fullfile('..', 'Results', 'data', 'NMSe_RHo_SnR10.mat'), ...
    fullfile('Results', 'data', 'NMSe_RHo_SnR10.mat')
};

loaded = false;
for i = 1:numel(mat_candidates)
    if exist(mat_candidates{i}, 'file')
        fprintf('Loading NMSE vs Rho workspace: %s\n', mat_candidates{i});
        load(mat_candidates{i});
        loaded = true;
        break;
    end
end

if ~loaded
    error('NMSe_RHo_SnR10.mat not found in Results/data.');
end

% 2. Load DCS CSV Data
csv_candidates = {
    'DCS_NMSE_vs_Rho_SNR10.csv', ...
    fullfile('..', 'python_dcs', 'data', 'DCS_NMSE_vs_Rho_SNR10.csv'), ...
    fullfile('python_dcs', 'data', 'DCS_NMSE_vs_Rho_SNR10.csv')
};

dcs_loaded = false;
for i = 1:numel(csv_candidates)
    if exist(csv_candidates{i}, 'file')
        dcs_table = readtable(csv_candidates{i});
        rho_dcs  = dcs_table{:, 1};
        nmse_dcs = dcs_table{:, 2};
        dcs_loaded = true;
        break;
    end
end

if ~dcs_loaded
    error('DCS_NMSE_vs_Rho_SNR10.csv not found.');
end

% Compression Ratio vector rho = M_T / Nt (Nt = 64 per subarray, or total)
if exist('M_T_Beam', 'var')
    rho_vec = M_T_Beam / 64;
else
    rho_vec = linspace(0.05, 1.0, size(NMSE_OMP, 1));
end

lw = 2; lw_dcs = 3; ms = 8;
color_DCS = [0.8500 0.3250 0.0980];

figure('Color', [1 1 1], 'Name', 'Figure 4.4 - NMSE vs Rho', 'Position', [100 100 750 520]);

plot(rho_vec, 10*log10(mean(NMSE_OMP, [2 3])), 'ko-', 'LineWidth', lw, 'MarkerSize', ms); hold on;
plot(rho_vec, 10*log10(mean(NMSE_SOMP, [2 3])), 'bx-', 'LineWidth', lw, 'MarkerSize', ms);
plot(rho_vec, 10*log10(mean(NMSE_OMP_DR, [2 3])), 'ms-', 'LineWidth', lw, 'MarkerSize', ms);
plot(rho_vec, 10*log10(mean(NMSE_SOMP_DR, [2 3])), 'rd-', 'LineWidth', lw, 'MarkerSize', ms);

plot(rho_dcs, nmse_dcs, 'p-', 'Color', color_DCS, 'LineWidth', lw_dcs, ...
     'MarkerFaceColor', color_DCS, 'MarkerSize', 10);

xlabel('Compression Ratio ($\rho = N_p / N_t$)', 'FontSize', 13, 'Interpreter', 'latex');
ylabel('NMSE [dB]', 'FontSize', 13, 'Interpreter', 'latex');
title('{\bf Figure 4.4:} NMSE vs. Compression Ratio ($\rho$) at SNR = 10 dB', 'FontSize', 14, 'Interpreter', 'latex');
legend('OMP', 'SOMP', 'OMP-DR', 'SOMP-DR', 'Deep CS (Proposed)', ...
       'Location', 'northeast', 'FontSize', 11, 'Interpreter', 'latex');
grid on; box on;
xlim([0.05 1.0]);
ylim([-18 0]);
ax = gca; ax.TickLabelInterpreter = 'latex'; ax.FontSize = 12;

saveas(gcf, fullfile(fileparts(mfilename('fullpath')), 'Fig4_4_NMSE_vs_Rho.png'));
fprintf('Figure 4.4 saved successfully.\n');
