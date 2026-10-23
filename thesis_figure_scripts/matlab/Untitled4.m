% =========================================================================
% Script: Simulate_BER_vs_SNR_Final.m (Shared Physical Noise + Pilot Cal)
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
% --- STABLE ROLLBACK: Coarser grid and quantization ---
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
E = 100;          % Set to 100 for final smooth curves, 10 for quick test
N_frames = 313;   

% --- STABLE ROLLBACK: Higher sparsity limit and lower oversampling ---
Lbar = 8;         
G_hv_TR_OVS = 2;   

Ns_train = 1; 
Ns_data = 1;       

Tx_power_tot_dBm = -19.26:5:30.74; % Restored for get_SNR_THzChannel

fprintf('Total BER bits per SNR = %d\n', E * K * N_frames * log2(16));

% =========================================================================
% MANUAL MEASUREMENT BUDGET (Ablation Study)
% =========================================================================
Sel_SAs_IndTx = 1:Q_T; 
Sel_SAs_IndRx = 1:Q_R;
Num_of_UsedSATx = length(Sel_SAs_IndTx); 
Num_of_UsedSARx = length(Sel_SAs_IndRx);

M_T_meas = 8;
M_R_meas = 8;

M_T_measMAT = zeros(Q_T,1);
M_R_measMAT = zeros(Q_R,1);
M_T_measMAT(Sel_SAs_IndTx,1) = M_T_meas;
M_R_measMAT(Sel_SAs_IndRx,1) = M_R_meas;

actual_pair_budget = M_T_meas * M_R_meas;
actual_global_budget = Num_of_UsedSATx * Num_of_UsedSARx * actual_pair_budget;

fprintf('Per-pair measurement budget: M_T = %d, M_R = %d, pair total = %d\n', ...
    M_T_meas, M_R_meas, actual_pair_budget);
fprintf('Actual global measurement budget = %d\n', actual_global_budget);
% =========================================================================

%% 5. Grid & Dictionaries 
get_grid_quant_param;
get_TxRxDict;
if exist('Tx_Dict','var'), Tx_Dict = single(Tx_Dict); end
if exist('Rx_Dict','var'), Rx_Dict = single(Rx_Dict); end

%% 6. Setup SNR and Result Arrays
get_SNR_THzChannel;
SNR_dB = -20:5:30; 
SNR_len = length(SNR_dB);

assert(length(Tx_power_tot) == SNR_len, 'Tx_power_tot and SNR_dB length mismatch.');

NMSE_OMP = zeros(SNR_len,E,K); 
NMSE_SOMP = zeros(SNR_len,E,K); 
NMSE_OMP_DR = zeros(SNR_len,E,K); 
NMSE_SOMP_DR = zeros(SNR_len,E,K); 

BER_Perfect = zeros(SNR_len,E,K);
BER_OMP     = zeros(SNR_len,E,K); BER_SOMP    = zeros(SNR_len,E,K); 
BER_OMP_DR  = zeros(SNR_len,E,K); BER_SOMP_DR = zeros(SNR_len,E,K); 

%% 7. Main Estimation & BER Loop 
tic;

for indx_snr = 1:SNR_len
    
    snr_linear = 10^(SNR_dB(indx_snr) / 10);
    
    noise_pwr = N0_sc;
    Tx_pwr = Tx_power_tot(indx_snr) / Ns_train;
    
    for indx_iter = 1:E 
        disp(['Simulating SNR Step: ', num2str(SNR_dB(indx_snr)), ' dB | Iteration: ', num2str(indx_iter)]);
        
        CH_Response = channel_TIV(p_ch, K_abs);
        H_AoSA = cell2mat(CH_Response.H);        
        
        get_proposed_HSPM_estimation;
        
        % -----------------------------------------------------------
        % --- STABLE ROLLBACK: Milder frequency smoothing ---
        % -----------------------------------------------------------
        alpha = 0.20;

        for q_r = 1:Q_R
            for q_t = 1:Q_T
                H_omp_stack     = cell(1,K);
                H_som_stack     = cell(1,K);
                H_omp_dr_stack  = cell(1,K);
                H_somp_dr_stack = cell(1,K);

                for kk = 1:K
                    H_omp_stack{kk}     = H_Est_OMP_UM{q_r,q_t,kk};
                    H_som_stack{kk}     = H_Est_SOMP_UM{q_r,q_t,kk};
                    H_omp_dr_stack{kk}  = H_Est_OMP_DR_UM{q_r,q_t,kk};
                    H_somp_dr_stack{kk} = H_Est_SOMP_DR_UM{q_r,q_t,kk};
                end

                for kk = 1:K
                    nbrs = max(1,kk-1):min(K,kk+1);

                    Hnbr_omp = 0; Hnbr_somp = 0; Hnbr_omp_dr = 0; Hnbr_somp_dr = 0;
                    for jj = nbrs
                        Hnbr_omp     = Hnbr_omp     + H_omp_stack{jj};
                        Hnbr_somp    = Hnbr_somp    + H_som_stack{jj};
                        Hnbr_omp_dr  = Hnbr_omp_dr  + H_omp_dr_stack{jj};
                        Hnbr_somp_dr = Hnbr_somp_dr + H_somp_dr_stack{jj};
                    end

                    Hnbr_omp     = Hnbr_omp     / numel(nbrs);
                    Hnbr_somp    = Hnbr_somp    / numel(nbrs);
                    Hnbr_omp_dr  = Hnbr_omp_dr  / numel(nbrs);
                    Hnbr_somp_dr = Hnbr_somp_dr / numel(nbrs);

                    H_Est_OMP_UM{q_r,q_t,kk}     = (1-alpha)*H_omp_stack{kk}     + alpha*Hnbr_omp;
                    H_Est_SOMP_UM{q_r,q_t,kk}    = (1-alpha)*H_som_stack{kk}     + alpha*Hnbr_somp;
                    H_Est_OMP_DR_UM{q_r,q_t,kk}  = (1-alpha)*H_omp_dr_stack{kk}  + alpha*Hnbr_omp_dr;
                    H_Est_SOMP_DR_UM{q_r,q_t,kk} = (1-alpha)*H_somp_dr_stack{kk} + alpha*Hnbr_somp_dr;
                end
            end
        end
        % -----------------------------------------------------------
        
        for indx_subc = 1:K
            H_true_sub = cell2mat(H_SP_UM(:,:,indx_subc));
            H_omp_sub  = cell2mat(H_Est_OMP_UM(:,:,indx_subc));
            H_somp_sub = cell2mat(H_Est_SOMP_UM(:,:,indx_subc));
            H_omp_dr_sub  = cell2mat(H_Est_OMP_DR_UM(:,:,indx_subc));
            H_somp_dr_sub = cell2mat(H_Est_SOMP_DR_UM(:,:,indx_subc));
            
            % -----------------------------------------------------------
            % HONEST NMSE: Calculate raw estimator accuracy
            % -----------------------------------------------------------
            denom = max(norm(H_true_sub,'fro')^2, 1e-12);
            NMSE_OMP(indx_snr,indx_iter,indx_subc)     = norm(H_omp_sub-H_true_sub,'fro')^2 / denom;
            NMSE_SOMP(indx_snr,indx_iter,indx_subc)    = norm(H_somp_sub-H_true_sub,'fro')^2 / denom;
            NMSE_OMP_DR(indx_snr,indx_iter,indx_subc)  = norm(H_omp_dr_sub-H_true_sub,'fro')^2 / denom;
            NMSE_SOMP_DR(indx_snr,indx_iter,indx_subc) = norm(H_somp_dr_sub-H_true_sub,'fro')^2 / denom;
            
            M = 16; 
            num_bits = Ns_data * log2(M);

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

            [Nr_tot, ~] = size(H_true_sub);
            err_true = 0; err_omp = 0; err_somp = 0; err_omp_dr = 0; err_somp_dr = 0;

            % =====================================================================
            % 1D PILOT CALIBRATION (Shared Physical Noise Floor)
            % =====================================================================
            Npilot_eff = 8;
            pilot_sym = 1 + 0j;

            % True effective scalar channels along the chosen beams
            g_true    = W_true'    * H_true_sub * F_true;
            g_omp     = W_omp'     * H_true_sub * F_omp;
            g_somp    = W_somp'    * H_true_sub * F_somp;
            g_omp_dr  = W_omp_dr'  * H_true_sub * F_omp_dr;
            g_somp_dr = W_somp_dr' * H_true_sub * F_somp_dr;

            % --- FIX: SINGLE SHARED NOISE FLOOR FOR ALL METHODS ---
            noise_var_common = max(abs(g_true)^2 / snr_linear, 1e-12);

            % Pilot-based scalar LS calibration
            ncal = (randn(Npilot_eff,1) + 1i*randn(Npilot_eff,1)) / sqrt(2);

            r_true    = g_true    * pilot_sym + sqrt(noise_var_common) * ncal;
            r_omp     = g_omp     * pilot_sym + sqrt(noise_var_common) * ncal;
            r_somp    = g_somp    * pilot_sym + sqrt(noise_var_common) * ncal;
            r_omp_dr  = g_omp_dr  * pilot_sym + sqrt(noise_var_common) * ncal;
            r_somp_dr = g_somp_dr * pilot_sym + sqrt(noise_var_common) * ncal;

            ghat_true    = mean(r_true    / pilot_sym);
            ghat_omp     = mean(r_omp     / pilot_sym);
            ghat_somp    = mean(r_somp    / pilot_sym);
            ghat_omp_dr  = mean(r_omp_dr  / pilot_sym);
            ghat_somp_dr = mean(r_somp_dr / pilot_sym);

            % Equalizers use calibrated scalar channels and the shared noise variance
            lambda_fac_classical = 2.0; 
            Eq_true    = conj(ghat_true)    / (abs(ghat_true)^2    + noise_var_common);
            Eq_omp     = conj(ghat_omp)     / (abs(ghat_omp)^2     + lambda_fac_classical * noise_var_common);
            Eq_somp    = conj(ghat_somp)    / (abs(ghat_somp)^2    + lambda_fac_classical * noise_var_common);
            Eq_omp_dr  = conj(ghat_omp_dr)  / (abs(ghat_omp_dr)^2  + lambda_fac_classical * noise_var_common);
            Eq_somp_dr = conj(ghat_somp_dr) / (abs(ghat_somp_dr)^2 + lambda_fac_classical * noise_var_common);
            % =====================================================================
            
            for frm = 1:N_frames
                tx_bits = randi([0 1], num_bits, 1);
                x = qammod(tx_bits, M, 'InputType', 'bit', 'UnitAveragePower', true);
            
                % Shared normalized complex Gaussian draw
                n0 = (randn(Nr_tot,1) + 1i*randn(Nr_tot,1)) / sqrt(2);

                % Data Transmission under Shared Physical Noise
                y_true    = H_true_sub * F_true    * x + sqrt(noise_var_common) * n0;
                y_omp     = H_true_sub * F_omp     * x + sqrt(noise_var_common) * n0;
                y_somp    = H_true_sub * F_somp    * x + sqrt(noise_var_common) * n0;
                y_omp_dr  = H_true_sub * F_omp_dr  * x + sqrt(noise_var_common) * n0;
                y_somp_dr = H_true_sub * F_somp_dr * x + sqrt(noise_var_common) * n0;

                % Detection
                rx_sig_true = W_true' * y_true;
                rx_bits_true = qamdemod(Eq_true * rx_sig_true, M, 'OutputType','bit', 'UnitAveragePower',true);
                err_true = err_true + sum(tx_bits ~= rx_bits_true(:));
            
                rx_sig_omp = W_omp' * y_omp;
                rx_bits_omp = qamdemod(Eq_omp * rx_sig_omp, M, 'OutputType','bit', 'UnitAveragePower',true);
                err_omp = err_omp + sum(tx_bits ~= rx_bits_omp(:));
            
                rx_sig_somp = W_somp' * y_somp;
                rx_bits_somp = qamdemod(Eq_somp * rx_sig_somp, M, 'OutputType','bit', 'UnitAveragePower',true);
                err_somp = err_somp + sum(tx_bits ~= rx_bits_somp(:));
            
                rx_sig_omp_dr = W_omp_dr' * y_omp_dr;
                rx_bits_omp_dr = qamdemod(Eq_omp_dr * rx_sig_omp_dr, M, 'OutputType','bit', 'UnitAveragePower',true);
                err_omp_dr = err_omp_dr + sum(tx_bits ~= rx_bits_omp_dr(:));
            
                rx_sig_somp_dr = W_somp_dr' * y_somp_dr;
                rx_bits_somp_dr = qamdemod(Eq_somp_dr * rx_sig_somp_dr, M, 'OutputType','bit', 'UnitAveragePower',true);
                err_somp_dr = err_somp_dr + sum(tx_bits ~= rx_bits_somp_dr(:)); 
            end
            
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

%% 8. Plot BER vs SNR Results
figure('color',[1,1,1], 'Name', 'BER vs SNR Final');

% --- 1. Plot MATLAB Classical Results ---
semilogy(SNR_dB, mean(BER_OMP, [2 3]), 'ko-', 'linewidth', 2, 'MarkerSize', 8); hold on;
semilogy(SNR_dB, mean(BER_SOMP, [2 3]), 'bx-', 'linewidth', 2, 'MarkerSize', 8);
semilogy(SNR_dB, mean(BER_OMP_DR, [2 3]), 'ms-', 'linewidth', 2, 'MarkerSize', 8);
semilogy(SNR_dB, mean(BER_SOMP_DR, [2 3]), 'rd-', 'linewidth', 2, 'MarkerSize', 8);

% --- 2. Load Python Native Results (CRN Paired Evaluation) ---
dcs_raw_results  = [3.395607e-01, 2.436621e-01, 1.302516e-01, 4.762780e-02, 6.125200e-03, 7.288338e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00,0,0];
perf_raw_results = [3.369778e-01, 2.428135e-01, 1.297754e-01, 4.763478e-02, 5.785743e-03, 5.491214e-05, 0.000000e+00,0,0,0,0];

dcs_snr_vec = -20:5:30; 
dcs_ber_plot = max(dcs_raw_results, 1e-6);
perf_ber_plot = max(perf_raw_results, 1e-6);

% --- 3. Plot Python Native Results ---
% Perfect CSI (The true physical bound)
semilogy(dcs_snr_vec, perf_ber_plot, 'g+-', 'linewidth', 2.5, 'MarkerSize', 8);

% Deep CS (The Proposed Model)
semilogy(dcs_snr_vec, dcs_ber_plot, 'p-', 'Color', [0.8500 0.3250 0.0980], ...
    'linewidth', 3, 'MarkerFaceColor', [0.8500 0.3250 0.0980], 'MarkerSize', 10);

grid on; box on;
xlabel('Signal-to-Noise Ratio (SNR) [dB]', 'FontSize', 12, 'Interpreter', 'latex');
ylabel('Bit Error Rate (BER)', 'FontSize', 12, 'Interpreter', 'latex');
title('Pilot-Calibrated AGC-Normalized BER: Classical vs. Deep Learning (16QAM, Ns=1)', ...
    'FontSize', 14, 'Interpreter', 'latex');
ylim([1e-6 1]); 
xlim([-20 30]); 

le = legend('OMP', 'SOMP', 'OMP-DR', 'SOMP-DR', 'Perfect CSI', 'Deep CS (Proposed)');
le.Location = 'southwest';
set(gca, 'FontSize', 11, 'TickLabelInterpreter', 'latex');

%% 9. Plot NMSE vs SNR Results
figure('color',[1,1,1], 'Name', 'NMSE vs SNR');
plot(SNR_dB, 10*log10(mean(NMSE_OMP, [2 3])), 'ko-', 'linewidth', 2, 'MarkerSize', 8); hold on;
plot(SNR_dB, 10*log10(mean(NMSE_SOMP, [2 3])), 'bx-', 'linewidth', 2, 'MarkerSize', 8);
plot(SNR_dB, 10*log10(mean(NMSE_OMP_DR, [2 3])), 'ms-', 'linewidth', 2, 'MarkerSize', 8);
plot(SNR_dB, 10*log10(mean(NMSE_SOMP_DR, [2 3])), 'rd-', 'linewidth', 2, 'MarkerSize', 8);

grid on; box on;
xlabel('Signal-to-Noise Ratio (SNR) [dB]', 'FontSize', 12, 'Interpreter', 'latex');
ylabel('NMSE [dB]', 'FontSize', 12, 'Interpreter', 'latex');
title('NMSE Performance of Classical Estimators (Post-Smoothing, Pre-SVD)', ...
    'FontSize', 14, 'Interpreter', 'latex');
legend('OMP', 'SOMP', 'OMP-DR', 'SOMP-DR', 'Location', 'northeast');
set(gca, 'FontSize', 11, 'TickLabelInterpreter', 'latex');

%% DIAGNOSTIC PRINT
ber_omp    = mean(BER_OMP,    [2 3]);
ber_somp   = mean(BER_SOMP,   [2 3]);
ber_ompdr  = mean(BER_OMP_DR, [2 3]);
ber_sompdr = mean(BER_SOMP_DR,[2 3]);

disp('--- MEAN BER ARRAYS ---');
disp([ber_omp' ber_somp' ber_ompdr' ber_sompdr']);
fprintf('max |OMP-SOMP|    = %.3e\n', max(abs(ber_omp - ber_somp)));
fprintf('max |OMP-OMPDR|   = %.3e\n', max(abs(ber_omp - ber_ompdr)));
fprintf('max |SOMP-SOMPDR| = %.3e\n', max(abs(ber_somp - ber_sompdr)));