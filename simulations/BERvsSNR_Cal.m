%% Dynamic Project Path Setup
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
projRoot = fileparts(thisDir);
if exist(fullfile(projRoot, 'setup_paths.m'), 'file')
    run(fullfile(projRoot, 'setup_paths.m'));
end

% =========================================================================
% Script: Simulate_BER_vs_SNR_Final.m
% Purpose: Generates mathematically correct NMSE and BER vs SNR plots
%          Uses native TeraMIMO physics, proper MMSE scaling, and a 
%          high-speed data frame loop for perfectly smooth BER curves.
%          INCLUDES STRICT FAIR PILOT-CALIBRATION (SHARED NOISE).
% =========================================================================
clc; clear; close all;

%% 1. Add Paths
path(pathdef); addpath(pwd);
cd Algorithms; addpath(genpath(pwd)); cd ..; cd TeraMIMO_Channel; addpath(genpath(pwd)); cd ..; cd Molecular_Absorption;addpath(genpath(pwd)); cd ..; cd Utilities; addpath(genpath(pwd)); cd ..;

%% 2. Initialize Channel Parameters & Physics Overrides
p_ch = generate_channel_param_TIV();
p_ch = update_channel_param_TIV(p_ch);
K_abs = get_Abs_Coef(p_ch);

%% 3. Hardware & Memory Overrides
get_array_trans_param;
num_multG = 2; Q_T_quant = 2; Q_R_quant = 2;
Qbar_R_h = 4; Qbar_R_v = 2; Qbar_R = 8;
if exist('p_ch','var')
    p_ch.BW = 10e9; p_ch.d_TR = 1.0; p_ch.dist = 1.0;
    p_ch.Nsub_c = 8; p_ch.Mar = 4; p_ch.Nar = 2; 
    p_ch = get_FrequencyRng_param(p_ch); 
    K_abs = get_Abs_Coef(p_ch);
end
B_sys = 10e9; K = 8; Q_R = 4;

%% 4. Main Simulation Parameters
E = 1;          % Number of channel realizations
N_frames = 100; % Data frames per channel 
Lbar = 10;      % Sparsity parameter for OMP/SOMP

Tx_power_tot_dBm = -19.26:5:33.74;
G_hv_TR_OVS = 2; 

Ns_train = 1; 
Ns_data = 4; 
Compression_RatioTx = 1; Compression_RatioRx = 1;
Sel_SAs_IndTx = 1:Q_T; Sel_SAs_IndRx = 1:Q_R;
Num_of_UsedSATx = length(Sel_SAs_IndTx); Num_of_UsedSARx = length(Sel_SAs_IndRx);
M_T_measMAT = zeros(Q_T,1); M_R_measMAT = zeros(Q_R,1); 
M_T_measMAT(Sel_SAs_IndTx,1) = Compression_RatioTx*Qbar_T; 
M_R_measMAT(Sel_SAs_IndRx,1) = Compression_RatioRx*Qbar_R;

% >>> STRICT FAIRNESS: 32-Symbol Pilot Configuration (Matches Python) <<<
N_PIL = 32;                     
PILOT_REP = N_PIL / Ns_data;
S_PIL = repmat(eye(Ns_data), 1, PILOT_REP);

%% 5. Grid & Dictionaries 
get_grid_quant_param;
get_TxRxDict;
if exist('Tx_Dict','var'), Tx_Dict = single(Tx_Dict); end
if exist('Rx_Dict','var'), Rx_Dict = single(Rx_Dict); end

%% 6. Setup SNR and Result Arrays
get_SNR_THzChannel;

NMSE_OMP = zeros(SNR_len,E,K); NMSE_SOMP = zeros(SNR_len,E,K); 
BER_Perfect = zeros(SNR_len,E,K);
BER_OMP     = zeros(SNR_len,E,K); BER_SOMP    = zeros(SNR_len,E,K); 
BER_OMP_DR  = zeros(SNR_len,E,K); BER_SOMP_DR = zeros(SNR_len,E,K); 

%% 7. Main Estimation & BER Loop
tic;
for indx_snr = 1:SNR_len
    
    noise_pwr = N0_sc; 
    Tx_pwr = Tx_power_tot(indx_snr)/Ns_train;
    tx_scale = sqrt(Tx_pwr / Ns_data);
    
    for indx_iter = 1:E
        disp(['Simulating SNR Step: ', num2str(indx_snr), '/', num2str(SNR_len), ' | Iteration: ', num2str(indx_iter)]);
        
        CH_Response = channel_TIV(p_ch, K_abs);
        H_AoSA = cell2mat(CH_Response.H);        
        
        get_proposed_HSPM_estimation;
        
        for indx_subc = 1:K
            H_true_sub = cell2mat(H_SP_UM(:,:,indx_subc));
            H_omp_sub  = cell2mat(H_Est_OMP_UM(:,:,indx_subc));
            H_somp_sub = cell2mat(H_Est_SOMP_UM(:,:,indx_subc));
            H_omp_dr_sub  = cell2mat(H_Est_OMP_DR_UM(:,:,indx_subc));
            H_somp_dr_sub = cell2mat(H_Est_SOMP_DR_UM(:,:,indx_subc));
            
            NMSE_OMP(indx_snr,indx_iter,indx_subc) = norm(H_omp_sub-H_true_sub,'fro')^2/norm(H_true_sub,'fro')^2;
            NMSE_SOMP(indx_snr,indx_iter,indx_subc) = norm(H_somp_sub-H_true_sub,'fro')^2/norm(H_true_sub,'fro')^2;
            
            M = 16; 
            num_bits = Ns_data * log2(M);

            % -----------------------------------------------------------
            % 1. Compute SVD and Combiners
            % -----------------------------------------------------------
            [U_true, ~, V_true] = svd(H_true_sub, 'econ');
            [U_omp, ~, V_omp]   = svd(H_omp_sub, 'econ');
            [U_somp, ~, V_somp] = svd(H_somp_sub, 'econ');
            [U_omp_dr, ~, V_omp_dr]   = svd(H_omp_dr_sub, 'econ');
            [U_somp_dr, ~, V_somp_dr] = svd(H_somp_dr_sub, 'econ');

            F_true = V_true(:, 1:Ns_data); W_true = U_true(:, 1:Ns_data);
            F_omp  = V_omp(:, 1:Ns_data);  W_omp  = U_omp(:, 1:Ns_data);
            F_somp = V_somp(:, 1:Ns_data); W_somp = U_somp(:, 1:Ns_data);
            F_omp_dr  = V_omp_dr(:, 1:Ns_data);  W_omp_dr  = U_omp_dr(:, 1:Ns_data);
            F_somp_dr = V_somp_dr(:, 1:Ns_data); W_somp_dr = U_somp_dr(:, 1:Ns_data);

            % -----------------------------------------------------------
            % 2. Build Pilot-Calibrated MMSE Equalizers (SHARED NOISE)
            % -----------------------------------------------------------
            [Nr_tot, ~] = size(H_true_sub);

            % Perfect CSI Baseline
            H_eff_true_tx = tx_scale * (W_true' * H_true_sub * F_true);
            Eq_true = (H_eff_true_tx' * H_eff_true_tx + noise_pwr * eye(Ns_data)) \ H_eff_true_tx';

            % >>> Generate ONE shared pilot noise matrix <<<
            n_pil = sqrt(noise_pwr/2) * (randn(Nr_tot, N_PIL) + 1i*randn(Nr_tot, N_PIL));

            % 1) OMP Pilot Calibration
            Y_pil_omp = W_omp' * (H_true_sub * (tx_scale * F_omp * S_PIL) + n_pil);
            H_eff_hat_omp = (Y_pil_omp * S_PIL') / PILOT_REP;
            Eq_omp = (H_eff_hat_omp' * H_eff_hat_omp + noise_pwr * eye(Ns_data)) \ H_eff_hat_omp';

            % 2) SOMP Pilot Calibration
            Y_pil_somp = W_somp' * (H_true_sub * (tx_scale * F_somp * S_PIL) + n_pil);
            H_eff_hat_somp = (Y_pil_somp * S_PIL') / PILOT_REP;
            Eq_somp = (H_eff_hat_somp' * H_eff_hat_somp + noise_pwr * eye(Ns_data)) \ H_eff_hat_somp';

            % 3) OMP-DR Pilot Calibration
            Y_pil_omp_dr = W_omp_dr' * (H_true_sub * (tx_scale * F_omp_dr * S_PIL) + n_pil);
            H_eff_hat_omp_dr = (Y_pil_omp_dr * S_PIL') / PILOT_REP;
            Eq_omp_dr = (H_eff_hat_omp_dr' * H_eff_hat_omp_dr + noise_pwr * eye(Ns_data)) \ H_eff_hat_omp_dr';

            % 4) SOMP-DR Pilot Calibration
            Y_pil_somp_dr = W_somp_dr' * (H_true_sub * (tx_scale * F_somp_dr * S_PIL) + n_pil);
            H_eff_hat_somp_dr = (Y_pil_somp_dr * S_PIL') / PILOT_REP;
            Eq_somp_dr = (H_eff_hat_somp_dr' * H_eff_hat_somp_dr + noise_pwr * eye(Ns_data)) \ H_eff_hat_somp_dr';
            
            % Reset error counters 
            err_true = 0; err_omp = 0; err_somp = 0; err_omp_dr = 0; err_somp_dr = 0;
            
            % -----------------------------------------------------------
            % 3. High-Speed Data Frame Loop
            % -----------------------------------------------------------
            for frm = 1:N_frames
                tx_bits = randi([0 1], num_bits, 1);
                s = qammod(tx_bits, M, 'InputType', 'bit', 'UnitAveragePower', true);
                x = tx_scale * s; 

                n_data = sqrt(noise_pwr/2) * (randn(Nr_tot,1) + 1i*randn(Nr_tot,1));

                % 1) Perfect CSI
                rx_sig_true = W_true' * (H_true_sub * F_true * x + n_data);
                rx_bits_true = qamdemod(Eq_true * rx_sig_true, M, 'OutputType','bit', 'UnitAveragePower',true);
                err_true = err_true + sum(tx_bits ~= rx_bits_true(:));

                % 2) OMP
                rx_sig_omp = W_omp' * (H_true_sub * F_omp * x + n_data);
                rx_bits_omp = qamdemod(Eq_omp * rx_sig_omp, M, 'OutputType','bit', 'UnitAveragePower',true);
                err_omp = err_omp + sum(tx_bits ~= rx_bits_omp(:));

                % 3) SOMP
                rx_sig_somp = W_somp' * (H_true_sub * F_somp * x + n_data);
                rx_bits_somp = qamdemod(Eq_somp * rx_sig_somp, M, 'OutputType','bit', 'UnitAveragePower',true);
                err_somp = err_somp + sum(tx_bits ~= rx_bits_somp(:));

                % 4) OMP-DR
                rx_sig_omp_dr = W_omp_dr' * (H_true_sub * F_omp_dr * x + n_data);
                rx_bits_omp_dr = qamdemod(Eq_omp_dr * rx_sig_omp_dr, M, 'OutputType','bit', 'UnitAveragePower',true);
                err_omp_dr = err_omp_dr + sum(tx_bits ~= rx_bits_omp_dr(:));

                % 5) SOMP-DR
                rx_sig_somp_dr = W_somp_dr' * (H_true_sub * F_somp_dr * x + n_data);
                rx_bits_somp_dr = qamdemod(Eq_somp_dr * rx_sig_somp_dr, M, 'OutputType','bit', 'UnitAveragePower',true);
                err_somp_dr = err_somp_dr + sum(tx_bits ~= rx_bits_somp_dr(:));
            end
            
            % -----------------------------------------------------------
            % 4. Final Average BER per Subcarrier
            % -----------------------------------------------------------
            total_bits_sent = N_frames * num_bits;
            BER_Perfect(indx_snr,indx_iter,indx_subc) = err_true / total_bits_sent;
            BER_OMP(indx_snr,indx_iter,indx_subc)     = err_omp / total_bits_sent;
            BER_SOMP(indx_snr,indx_iter,indx_subc)    = err_somp / total_bits_sent;
            BER_OMP_DR(indx_snr,indx_iter,indx_subc)  = err_omp_dr / total_bits_sent;
            BER_SOMP_DR(indx_snr,indx_iter,indx_subc) = err_somp_dr / total_bits_sent;
        end
    end
end
Sim_Duration = toc;
disp(['Simulation Finished in ', num2str(Sim_Duration), ' seconds.']);

%% 8. Export Data to CSV for Unified Python Plotting
mean_BER_Perfect = squeeze(mean(BER_Perfect, [2 3]));
mean_BER_OMP     = squeeze(mean(BER_OMP, [2 3]));
mean_BER_SOMP    = squeeze(mean(BER_SOMP, [2 3]));
mean_BER_OMP_DR  = squeeze(mean(BER_OMP_DR, [2 3]));
mean_BER_SOMP_DR = squeeze(mean(BER_SOMP_DR, [2 3]));

matlab_results = table(SNR_dB', mean_BER_Perfect, mean_BER_OMP, mean_BER_SOMP, mean_BER_OMP_DR, mean_BER_SOMP_DR, ...
    'VariableNames', {'SNR_dB', 'BER_Perfect', 'BER_OMP_Cal', 'BER_SOMP_Cal', 'BER_OMP_DR_Cal', 'BER_SOMP_DR_Cal'});
writetable(matlab_results, 'Classical_Baselines_Calibrated.csv');
disp('Classical baselines saved to Classical_Baselines_Calibrated.csv');
%% 8. Plot BER vs SNR Results
figure('color',[1,1,1], 'Name', 'BER vs SNR Final');

% Safe mean function to prevent log(0) crashes at high SNR
safe_mean = @(x) max(squeeze(mean(x, [2 3])), 1e-12);

% Plot Classical Calibrated Baselines
semilogy(SNR_dB, safe_mean(BER_Perfect), 'g+-', 'linewidth', 2.5, 'MarkerSize', 8); hold on;
semilogy(SNR_dB, safe_mean(BER_OMP), 'ko-', 'linewidth', 2, 'MarkerSize', 8);
semilogy(SNR_dB, safe_mean(BER_SOMP), 'bx-', 'linewidth', 2, 'MarkerSize', 8);
semilogy(SNR_dB, safe_mean(BER_OMP_DR), 'ms-', 'linewidth', 2, 'MarkerSize', 8);
semilogy(SNR_dB, safe_mean(BER_SOMP_DR), 'rd-', 'linewidth', 2, 'MarkerSize', 8);

% Name of the CSV generated by your final Python script
csv_filename = 'DCS_BER_vs_SNR_Deployable.csv'; 

if exist(csv_filename, 'file')
    dcs_table = readtable(csv_filename);
    
    % Ensure we pull the exact Calibrated DCS column, not the Perfect baseline
    snr_dcs = dcs_table.SNR_dB;
    ber_dcs_cal = max(dcs_table.BER_DCS_Cal, 1e-12); 
    
    semilogy(snr_dcs, ber_dcs_cal, 'p-', 'Color', [0.8500 0.3250 0.0980], ...
        'linewidth', 3, 'MarkerFaceColor', [0.8500 0.3250 0.0980], 'MarkerSize', 10);
        
    le = legend('Perfect CSI', 'OMP (Calibrated)', 'SOMP (Calibrated)', ...
        'OMP-DR (Calibrated)', 'SOMP-DR (Calibrated)', 'Deep CS (Proposed)');
else
    disp(['Note: ' csv_filename ' not found. Run Python script to overlay AI results.']);
    le = legend('Perfect CSI', 'OMP (Calibrated)', 'SOMP (Calibrated)', ...
        'OMP-DR (Calibrated)', 'SOMP-DR (Calibrated)');
end

% Formatting to match your exact IEEE/LaTeX specifications
xlabel('Signal-to-Noise Ratio (SNR) [dB]', 'FontSize', 14, 'Interpreter', 'latex', 'FontWeight', 'bold');
ylabel('Bit Error Rate (BER)', 'FontSize', 14, 'Interpreter', 'latex', 'FontWeight', 'bold');
title('\textbf{System BER Performance vs. SNR}', 'Interpreter', 'latex', 'FontSize', 15);

le.Interpreter = 'latex'; 
le.Location = 'southwest'; 
le.FontSize = 12;
ax = gca; 
ax.TickLabelInterpreter = 'latex'; 
ax.FontSize = 13;
box off; grid on; axis tight;
ylim([1e-5 1]);

% Save the final figure
saveas(gcf, 'Final_BER_Plot_LaTeX.png');