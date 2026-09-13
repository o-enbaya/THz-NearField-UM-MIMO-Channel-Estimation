%% Dynamic Project Path Setup
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
projRoot = fileparts(thisDir);
if exist(fullfile(projRoot, 'setup_paths.m'), 'file')
    run(fullfile(projRoot, 'setup_paths.m'));
end

%% =========================================================================
% FINAL COMPLEXITY + TRADEOFF PLOTS
% Paste after NMSE computation and DR-statistics printing
% =========================================================================

%% -----------------------------
% 1) Complexity parameters
% -----------------------------
% Classical side (from your MATLAB setup)
K_comp  = K;                              % = 8
L_comp  = Lbar;                           % = 10
Np_comp = Qbar_T * Qbar_R;                % = 64*8 = 512
G_comp  = size(Abar_T,2) * size(Abar_R,2);% = 8192

% Use measured reduced dictionary averages if available
if exist('G_OMP_DR_all','var') && ~isempty(G_OMP_DR_all)
    G_OMP_DR_avg = mean(G_OMP_DR_all);
else
    G_OMP_DR_avg = 8023.32;   % fallback from your measured run
end

if exist('G_SOMP_DR_all','var') && ~isempty(G_SOMP_DR_all)
    G_SOMP_DR_avg = mean(G_SOMP_DR_all);
else
    G_SOMP_DR_avg = 8098.91;  % fallback from your measured run
end

% DCS side (from your DCS script)
T_dcs   = 20;
Ns_dcs  = 4;
Np_dcs  = 100;
Nt_dcs  = 256;
Nr_dcs  = 32;

%% -----------------------------
% 2) Per-estimate inference complexity
% -----------------------------
C_DCS     = T_dcs * Ns_dcs * Np_dcs * Nt_dcs * Nr_dcs;
C_OMP     = K_comp * (L_comp * Np_comp * G_comp + L_comp^3);
C_SOMP    = L_comp * K_comp * Np_comp * G_comp + K_comp * Np_comp * L_comp^2 + L_comp^3;
C_OMP_DR  = K_comp * (L_comp * Np_comp * G_OMP_DR_avg + L_comp^3);
C_SOMP_DR = L_comp * K_comp * Np_comp * G_SOMP_DR_avg + K_comp * Np_comp * L_comp^2 + L_comp^3;

labels_all = {'DCS','OMP','SOMP','OMP-DR','SOMP-DR'};
C_all      = [C_DCS, C_OMP, C_SOMP, C_OMP_DR, C_SOMP_DR];
C_norm_dcs = C_all / C_DCS;

labels_class = {'OMP','SOMP','OMP-DR','SOMP-DR'};
C_class      = [C_OMP, C_SOMP, C_OMP_DR, C_SOMP_DR];
C_norm_omp   = C_class / C_OMP;

%% -----------------------------
% 3) NMSE summary values
% -----------------------------
% Classical methods: average NMSE over all tested SNRs
nmse_omp_avg_db    = 10*log10(mean(NMSE_OMP(:)));
nmse_somp_avg_db   = 10*log10(mean(NMSE_SOMP(:)));
nmse_ompdr_avg_db  = 10*log10(mean(NMSE_OMP_DR(:)));
nmse_sompdr_avg_db = 10*log10(mean(NMSE_SOMP_DR(:)));

% Classical methods: NMSE at a chosen SNR point
target_snr = 0;   % change to 5 or 10 if you prefer
[~, idx_target_snr] = min(abs(SNR_dB - target_snr));

nmse_omp_pt_db    = 10*log10(mean(NMSE_OMP(idx_target_snr,:,:),    [2 3]));
nmse_somp_pt_db   = 10*log10(mean(NMSE_SOMP(idx_target_snr,:,:),   [2 3]));
nmse_ompdr_pt_db  = 10*log10(mean(NMSE_OMP_DR(idx_target_snr,:,:), [2 3]));
nmse_sompdr_pt_db = 10*log10(mean(NMSE_SOMP_DR(idx_target_snr,:,:),[2 3]));

% -----------------------------
% IMPORTANT:
% Replace these two values with your actual DCS NMSE results
% -----------------------------
nmse_dcs_avg_db = -15.0;  % <-- replace with actual DCS average NMSE over SNR range
nmse_dcs_pt_db  = -12.0;  % <-- replace with actual DCS NMSE at target_snr

nmse_all_avg_db = [nmse_dcs_avg_db, nmse_omp_avg_db, nmse_somp_avg_db, nmse_ompdr_avg_db, nmse_sompdr_avg_db];
nmse_all_pt_db  = [nmse_dcs_pt_db,  nmse_omp_pt_db,  nmse_somp_pt_db,  nmse_ompdr_pt_db,  nmse_sompdr_pt_db];

nmse_class_avg_db = [nmse_omp_avg_db, nmse_somp_avg_db, nmse_ompdr_avg_db, nmse_sompdr_avg_db];
nmse_class_pt_db  = [nmse_omp_pt_db,  nmse_somp_pt_db,  nmse_ompdr_pt_db,  nmse_sompdr_pt_db];

%% -----------------------------
% 4) Print summary
% -----------------------------
fprintf('\n============================================================\n');
fprintf('FINAL COMPLEXITY + TRADEOFF SUMMARY\n');
fprintf('============================================================\n');
fprintf('Per-estimate inference complexity:\n');
fprintf('DCS      = %.4f\n', C_DCS);
fprintf('OMP      = %.4f\n', C_OMP);
fprintf('SOMP     = %.4f\n', C_SOMP);
fprintf('OMP-DR   = %.4f\n', C_OMP_DR);
fprintf('SOMP-DR  = %.4f\n', C_SOMP_DR);
fprintf('\nNormalized to DCS:\n');
fprintf('DCS      = %.4f\n', C_norm_dcs(1));
fprintf('OMP      = %.4f\n', C_norm_dcs(2));
fprintf('SOMP     = %.4f\n', C_norm_dcs(3));
fprintf('OMP-DR   = %.4f\n', C_norm_dcs(4));
fprintf('SOMP-DR  = %.4f\n', C_norm_dcs(5));
fprintf('\nAverage NMSE over all SNRs (dB):\n');
fprintf('DCS      = %.4f\n', nmse_all_avg_db(1));
fprintf('OMP      = %.4f\n', nmse_all_avg_db(2));
fprintf('SOMP     = %.4f\n', nmse_all_avg_db(3));
fprintf('OMP-DR   = %.4f\n', nmse_all_avg_db(4));
fprintf('SOMP-DR  = %.4f\n', nmse_all_avg_db(5));
fprintf('============================================================\n');

%% -----------------------------
% 5) Complexity bar chart (all methods)
% -----------------------------
figure('Color',[1 1 1]);
bar(C_norm_dcs);
set(gca,'XTick',1:numel(labels_all),'XTickLabel',labels_all);
ylabel('Normalized Inference Complexity (relative to DCS)','FontSize',13);
title('Inference Complexity Relative to DCS','FontSize',13);
grid on; box off;
ylim([0 1.15*max(C_norm_dcs)]);

for i = 1:numel(C_norm_dcs)
    text(i, C_norm_dcs(i)+0.05*max(C_norm_dcs), sprintf('%.2f', C_norm_dcs(i)), ...
        'HorizontalAlignment','center','FontSize',11);
end

%% -----------------------------
% 6) Zoomed complexity bar chart (classical only)
% -----------------------------
figure('Color',[1 1 1]);
bar(C_norm_omp);
set(gca,'XTick',1:numel(labels_class),'XTickLabel',labels_class);
ylabel('Normalized Inference Complexity (relative to OMP)','FontSize',13);
title('Zoomed Complexity Comparison for Classical Methods','FontSize',13);
grid on; box off;
ylim([0.97 1.005]);

for i = 1:numel(C_norm_omp)
    text(i, C_norm_omp(i)+0.001, sprintf('%.3f', C_norm_omp(i)), ...
        'HorizontalAlignment','center','FontSize',11);
end

%% -----------------------------
% 7) Full tradeoff plot: average NMSE over SNR range
% -----------------------------
figure('Color',[1 1 1]);
scatter(C_norm_dcs, nmse_all_avg_db, 100, 'filled'); hold on;
grid on; box off;

for i = 1:numel(labels_all)
    text(C_norm_dcs(i)+0.04, nmse_all_avg_db(i), labels_all{i}, 'FontSize',11);
end

xlabel('Normalized Inference Complexity (relative to DCS)','FontSize',13);
ylabel('Average NMSE over SNR range (dB)','FontSize',13);
title('Average Complexity-Performance Tradeoff','FontSize',13);
xlim([0.9, 1.05*max(C_norm_dcs)]);

%% -----------------------------
% 8) Zoomed tradeoff plot: classical methods only
% -----------------------------
figure('Color',[1 1 1]);
scatter(C_norm_omp, nmse_class_avg_db, 100, 'filled'); hold on;
grid on; box off;

for i = 1:numel(labels_class)
    text(C_norm_omp(i)+0.002, nmse_class_avg_db(i), labels_class{i}, 'FontSize',11);
end

xlabel('Normalized Inference Complexity (relative to OMP)','FontSize',13);
ylabel('Average NMSE over SNR range (dB)','FontSize',13);
title('Zoomed Complexity-Performance Tradeoff for Classical Methods','FontSize',13);
xlim([0.975, 1.005]);

%% -----------------------------
% 9) Optional tradeoff plot at one SNR only
% -----------------------------
figure('Color',[1 1 1]);
scatter(C_norm_dcs, nmse_all_pt_db, 100, 'filled'); hold on;
grid on; box off;

for i = 1:numel(labels_all)
    text(C_norm_dcs(i)+0.04, nmse_all_pt_db(i), labels_all{i}, 'FontSize',11);
end

xlabel('Normalized Inference Complexity (relative to DCS)','FontSize',13);
ylabel(sprintf('NMSE at %d dB SNR (dB)', target_snr),'FontSize',13);
title(sprintf('Complexity-Performance Tradeoff at %d dB', target_snr),'FontSize',13);
xlim([0.9, 1.05*max(C_norm_dcs)]);

%% -----------------------------
% 10) Percentage complexity reduction by DR
% -----------------------------
red_ompdr_pct  = (1 - C_OMP_DR/C_OMP) * 100;
red_sompdr_pct = (1 - C_SOMP_DR/C_SOMP) * 100;

figure('Color',[1 1 1]);
bar([red_ompdr_pct, red_sompdr_pct]);
set(gca,'XTick',1:2,'XTickLabel',{'OMP-DR vs OMP','SOMP-DR vs SOMP'});
ylabel('Complexity Reduction (%)','FontSize',13);
title('Complexity Reduction Achieved by DR','FontSize',13);
grid on; box off;
ylim([0, max([red_ompdr_pct, red_sompdr_pct]) + 0.5]);

text(1, red_ompdr_pct+0.05, sprintf('%.2f%%', red_ompdr_pct), ...
    'HorizontalAlignment','center','FontSize',11);
text(2, red_sompdr_pct+0.05, sprintf('%.2f%%', red_sompdr_pct), ...
    'HorizontalAlignment','center','FontSize',11);