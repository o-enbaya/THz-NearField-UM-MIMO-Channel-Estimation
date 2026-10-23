clc; clear; close all;

%% Add Paths
path(pathdef); addpath(pwd);
cd Algorithms; addpath(genpath(pwd)); cd ..;
cd TeraMIMO_Channel; addpath(genpath(pwd)); cd ..;
cd Molecular_Absorption; addpath(genpath(pwd)); cd ..;
cd Utilities; addpath(genpath(pwd)); cd ..;

%% Initialize Channel Parameters
p_ch = generate_channel_param_TIV();
p_ch = update_channel_param_TIV(p_ch);

%% Calculation of Absorption Coefficient
K_abs = get_Abs_Coef(p_ch);

%% Parameters related to the transmitter and receiver uniform linear/planar arrays
get_array_trans_param;

% =========================================================================
% FIXES
% =========================================================================
num_multG = 2;
Q_T_quant = 2;
Q_R_quant = 2;

Qbar_R_h = 4;
Qbar_R_v = 2;
Qbar_R   = 8;

if exist('p_ch','var')
    p_ch.BW   = 10e9;
    p_ch.d_TR = 1.0;
    p_ch.dist = 1.0;
    p_ch.Nsub_c = 8;
    p_ch.Mar = 4;
    p_ch.Nar = 2;
    p_ch = get_FrequencyRng_param(p_ch);
    K_abs = get_Abs_Coef(p_ch);
end

B_sys = 10e9;
K     = 8;

Q_R     = 4;
Q_R_tot = 32;
% =========================================================================

%% Main Simulation Parameters
E = 1;                     % number of Monte Carlo trials
Lbar = 10;                   % sparsity level estimate
Tx_power_tot_dBm = -19.26:5:33.74;
DNR_dB = 0;
G_hv_TR_OVS = 2;
num_multG = 2;
Q_T_quant = 2;
Q_R_quant = 2;

%% Measurements/Beams during training/data transmission
Ns_train = 1;
Ns_data  = min(Q_T,Q_R);
N_RF_T   = Ns_data;
N_RF_R   = Ns_data;

M_T_Beam = Qbar_T;
M_R_Beam = Qbar_R;

Compression_RatioTx = 1;
Compression_RatioRx = 1;

Sel_SAs_IndTx = 1:Q_T;
Sel_SAs_IndRx = 1:Q_R;

Num_of_UsedSATx = length(Sel_SAs_IndTx);
Num_of_UsedSARx = length(Sel_SAs_IndRx);

M_T_measMAT = zeros(Q_T,1);
M_R_measMAT = zeros(Q_R,1);

M_T_measMAT(Sel_SAs_IndTx,1) = Compression_RatioTx * M_T_Beam;
M_R_measMAT(Sel_SAs_IndRx,1) = Compression_RatioRx * M_R_Beam;

%% Grid / Quantization
get_grid_quant_param;

%% Construct Tx/Rx Dictionaries
get_TxRxDict;

fprintf('K = %d\n', K);
fprintf('L = %d\n', Lbar);
fprintf('G_T = %d\n', size(Abar_T,2));
fprintf('G_R = %d\n', size(Abar_R,2));
fprintf('G   = %d\n', size(Abar_T,2)*size(Abar_R,2));
fprintf('Qbar_T = %d\n', Qbar_T);
fprintf('Qbar_R = %d\n', Qbar_R);
fprintf('Np = %d\n', Qbar_T * Qbar_R);

if exist('Tx_Dict','var'), Tx_Dict = single(Tx_Dict); end
if exist('Rx_Dict','var'), Rx_Dict = single(Rx_Dict); end
if exist('A_T_all','var'), A_T_all = single(A_T_all); end
if exist('A_R_all','var'), A_R_all = single(A_R_all); end

%% Define SNR
get_SNR_THzChannel;

%% Initialize NMSE Arrays
NMSE_OMP     = zeros(SNR_len,E,K);
NMSE_OMP_DR  = zeros(SNR_len,E,K);
NMSE_SOMP    = zeros(SNR_len,E,K);
NMSE_SOMP_DR = zeros(SNR_len,E,K);

%% DR reduced-dictionary trackers
G_OMP_DR_all     = [];
G_SOMP_DR_all    = [];
G_OMP_DR_T_all   = [];
G_OMP_DR_R_all   = [];
G_SOMP_DR_T_all  = [];
G_SOMP_DR_R_all  = [];

%% Runtime trackers (seconds per FULL realization)
runtime_OMP     = zeros(SNR_len,1);
runtime_SOMP    = zeros(SNR_len,1);
runtime_OMP_DR  = zeros(SNR_len,1);
runtime_SOMP_DR = zeros(SNR_len,1);

%% Main Loop
tic;
for indx_snr = 1:SNR_len
    noise_pwr = N0_sc;
    Tx_pwr = Tx_power_tot(indx_snr)/Ns_train;

    for indx_iter = 1:E
        disp([indx_snr indx_iter])

        % Generate THz channel
        CH_Response = channel_TIV(p_ch, K_abs);
        H_AoSA = cell2mat(CH_Response.H);

        % reset per-realization method timers
        t_OMP_thiscall     = 0;
        t_SOMP_thiscall    = 0;
        t_OMP_DR_thiscall  = 0;
        t_SOMP_DR_thiscall = 0;

        % clear DR size variables from previous realization
        clear G_OMP_DR_subc G_OMP_DR_T_subc G_OMP_DR_R_subc
        clear G_SOMP_DR_val G_SOMP_DR_T_val G_SOMP_DR_R_val

        % Run estimators
        get_proposed_HSPM_estimation;

        % accumulate runtime (per full realization)
        runtime_OMP(indx_snr)     = runtime_OMP(indx_snr)     + t_OMP_thiscall;
        runtime_SOMP(indx_snr)    = runtime_SOMP(indx_snr)    + t_SOMP_thiscall;
        runtime_OMP_DR(indx_snr)  = runtime_OMP_DR(indx_snr)  + t_OMP_DR_thiscall;
        runtime_SOMP_DR(indx_snr) = runtime_SOMP_DR(indx_snr) + t_SOMP_DR_thiscall;

        % collect DR sizes
        if exist('G_OMP_DR_subc','var') && ~isempty(G_OMP_DR_subc)
            G_OMP_DR_all = [G_OMP_DR_all, G_OMP_DR_subc(:).'];
        end
        if exist('G_OMP_DR_T_subc','var') && ~isempty(G_OMP_DR_T_subc)
            G_OMP_DR_T_all = [G_OMP_DR_T_all, G_OMP_DR_T_subc(:).'];
        end
        if exist('G_OMP_DR_R_subc','var') && ~isempty(G_OMP_DR_R_subc)
            G_OMP_DR_R_all = [G_OMP_DR_R_all, G_OMP_DR_R_subc(:).'];
        end

        if exist('G_SOMP_DR_val','var') && ~isempty(G_SOMP_DR_val)
            G_SOMP_DR_all = [G_SOMP_DR_all, G_SOMP_DR_val];
        end
        if exist('G_SOMP_DR_T_val','var') && ~isempty(G_SOMP_DR_T_val)
            G_SOMP_DR_T_all = [G_SOMP_DR_T_all, G_SOMP_DR_T_val];
        end
        if exist('G_SOMP_DR_R_val','var') && ~isempty(G_SOMP_DR_R_val)
            G_SOMP_DR_R_all = [G_SOMP_DR_R_all, G_SOMP_DR_R_val];
        end

        % NMSE
        for indx_subc = 1:K
            NMSE_OMP(indx_snr,indx_iter,indx_subc) = ...
                norm(cell2mat(H_Est_OMP_UM(:,:,indx_subc)) - cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2 / ...
                norm(cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2;

            NMSE_OMP_DR(indx_snr,indx_iter,indx_subc) = ...
                norm(cell2mat(H_Est_OMP_DR_UM(:,:,indx_subc)) - cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2 / ...
                norm(cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2;

            NMSE_SOMP(indx_snr,indx_iter,indx_subc) = ...
                norm(cell2mat(H_Est_SOMP_UM(:,:,indx_subc)) - cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2 / ...
                norm(cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2;

            NMSE_SOMP_DR(indx_snr,indx_iter,indx_subc) = ...
                norm(cell2mat(H_Est_SOMP_DR_UM(:,:,indx_subc)) - cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2 / ...
                norm(cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2;
        end
    end
end

%% Total simulation time
Sim_Duration = toc;
hr   = floor(Sim_Duration/3600);
mint = floor((Sim_Duration - hr*3600)/60);
sec  = Sim_Duration - hr*3600 - mint*60;
fprintf('The simulation time is: %d hr %d min %f sec\n',hr,mint,sec);

%% DR stats
G_full = size(Abar_T,2) * size(Abar_R,2);

fprintf('\n===== Reduced Dictionary Statistics =====\n');
fprintf('Full dictionary size G = %d\n', G_full);

if ~isempty(G_OMP_DR_all)
    G_OMP_DR_avg = mean(G_OMP_DR_all);
    fprintf('Avg OMP-DR reduced size      = %.2f\n', G_OMP_DR_avg);
    fprintf('OMP-DR reduction factor      = %.2f x\n', G_full / G_OMP_DR_avg);
end

if ~isempty(G_SOMP_DR_all)
    G_SOMP_DR_avg = mean(G_SOMP_DR_all);
    fprintf('Avg SOMP-DR reduced size     = %.2f\n', G_SOMP_DR_avg);
    fprintf('SOMP-DR reduction factor     = %.2f x\n', G_full / G_SOMP_DR_avg);
end

if ~isempty(G_OMP_DR_T_all), fprintf('Avg OMP-DR Tx reduced size   = %.2f\n', mean(G_OMP_DR_T_all)); end
if ~isempty(G_OMP_DR_R_all), fprintf('Avg OMP-DR Rx reduced size   = %.2f\n', mean(G_OMP_DR_R_all)); end
if ~isempty(G_SOMP_DR_T_all), fprintf('Avg SOMP-DR Tx reduced size  = %.2f\n', mean(G_SOMP_DR_T_all)); end
if ~isempty(G_SOMP_DR_R_all), fprintf('Avg SOMP-DR Rx reduced size  = %.2f\n', mean(G_SOMP_DR_R_all)); end

%% Average runtime per full realization
avg_runtime_OMP     = runtime_OMP / E;
avg_runtime_SOMP    = runtime_SOMP / E;
avg_runtime_OMP_DR  = runtime_OMP_DR / E;
avg_runtime_SOMP_DR = runtime_SOMP_DR / E;

fprintf('\n===== Runtime Summary (seconds per full realization) =====\n');
fprintf('OMP avg runtime     = %.6f s\n', mean(avg_runtime_OMP));
fprintf('SOMP avg runtime    = %.6f s\n', mean(avg_runtime_SOMP));
fprintf('OMP-DR avg runtime  = %.6f s\n', mean(avg_runtime_OMP_DR));
fprintf('SOMP-DR avg runtime = %.6f s\n', mean(avg_runtime_SOMP_DR));

%% Plot NMSE vs SNR
figure('color',[1,1,1]);
plot(SNR_dB,10*log10(mean(NMSE_OMP,[2 3])),'ko-','linewidth',2,'MarkerSize',10); hold on;
plot(SNR_dB,10*log10(mean(NMSE_SOMP,[2 3])),'bx-','linewidth',2,'MarkerSize',10);
plot(SNR_dB,10*log10(mean(NMSE_OMP_DR,[2 3])),'ms-','linewidth',2,'MarkerSize',10);
plot(SNR_dB,10*log10(mean(NMSE_SOMP_DR,[2 3])),'rd-','linewidth',2,'MarkerSize',10);
xlabel('SNR (dB)','FontSize',14,'Interpreter','latex');
ylabel('NMSE (dB)','FontSize',14,'Interpreter','latex');
le = legend('OMP','SOMP','OMP-DR','SOMP-DR');
le.Interpreter='latex';
le.Location='northeast';
le.FontSize=14;
ax = gca;
ax.TickLabelInterpreter='latex';
ax.FontSize=14;
box off; grid on;

%% Plot runtime vs SNR
figure('Color',[1 1 1]);
plot(SNR_dB, avg_runtime_OMP, 'ko-','LineWidth',2,'MarkerSize',8); hold on;
plot(SNR_dB, avg_runtime_SOMP, 'bx-','LineWidth',2,'MarkerSize',8);
plot(SNR_dB, avg_runtime_OMP_DR, 'ms-','LineWidth',2,'MarkerSize',8);
plot(SNR_dB, avg_runtime_SOMP_DR, 'rd-','LineWidth',2,'MarkerSize',8);
xlabel('SNR (dB)','FontSize',13);
ylabel('Average Runtime per Full Realization (s)','FontSize',13);
title('Runtime vs SNR for Classical Methods','FontSize',13);
legend('OMP','SOMP','OMP-DR','SOMP-DR','Location','best');
grid on; box off;

%% Runtime bar chart
runtime_bar = [mean(avg_runtime_OMP), mean(avg_runtime_SOMP), ...
               mean(avg_runtime_OMP_DR), mean(avg_runtime_SOMP_DR)];

figure('Color',[1 1 1]);
bar(runtime_bar);
set(gca,'XTick',1:4,'XTickLabel',{'OMP','SOMP','OMP-DR','SOMP-DR'});
ylabel('Average Runtime per Full Realization (s)','FontSize',13);
title('Average Runtime Comparison for Classical Methods','FontSize',13);
grid on; box off;