%% Dynamic Project Path Setup
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
projRoot = fileparts(thisDir);
if exist(fullfile(projRoot, 'setup_paths.m'), 'file')
    run(fullfile(projRoot, 'setup_paths.m'));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Custom Script: NMSE vs. Compression Ratio (\rho)
% Based strictly on the NMSEvsSNR.m template
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clc;clear;close all;
%% Add Paths
path(pathdef); addpath(pwd);
cd Algorithms; addpath(genpath(pwd)); cd ..; cd TeraMIMO_Channel; addpath(genpath(pwd)); cd ..; cd Molecular_Absorption;addpath(genpath(pwd)); cd ..; cd Utilities; addpath(genpath(pwd)); cd ..;
%% Initialize Channel Parameters
p_ch = generate_channel_param_TIV();
p_ch = update_channel_param_TIV(p_ch);
%% Calculation of Absorption Coefficient 
K_abs = get_Abs_Coef(p_ch);
%% Parameters related to the transmitter and receiver uniform linear/planar arrays
get_array_trans_param;
%% Main Simulation Parameters
E = 1; % Number of trials/runs/iterations
Lbar = 10; % Estimation of the sparsity level value 

% --- DYNAMIC SNR SETUP (Calculated safely inside main script) ---
SNR_target = 40; % Type your target SNR here (e.g., 40, 20, or 10)

% 1. Pre-calculate Noise and Path Loss based on current p_ch settings
tmp_K = p_ch.Nsub_c;
tmp_N0_dBm = -173.8 + 10*log10(p_ch.BW);
tmp_lambda = p_ch.c / p_ch.Fc;
tmp_PL = ((4*pi*p_ch.d_tx_rx)/tmp_lambda)^p_ch.PLE * exp(K_abs(ceil(tmp_K/2),1)*p_ch.d_tx_rx);
tmp_PL_dB = 10*log10(tmp_PL);

% 2. Solve for the exact Transmit Power required
tmp_const = tmp_N0_dBm + 10*log10(tmp_K);
Tx_power_tot_dBm = SNR_target + tmp_PL_dB + tmp_const;
G_hv_TR_OVS = 2; % Dictionary oversampling
num_multG = 2; % Grid oversampling
Q_T_quant = 2; % Number of Tx PSs quantization bits
Q_R_quant = 2; % Number of Rx PSs quantization bits

%% Measurements/Beams during the training/data transmission phase
Ns_train = 1; % Number of spatial streams during training
Ns_data = min(Q_T,Q_R); % The number of spatial streams during data transmission Ns
N_RF_T = Ns_data; % Number of Tx SAs
N_RF_R = Ns_data; % Number of Rx SAs

% --- CHANGED: Array of Measurements (Sweeping N_p) ---
M_T_Beam = [4:4:40 48 56 Qbar_T]; % Training beams sweep
M_R_Beam = repmat(Qbar_R, 1, length(M_T_Beam)); % Fixes Rx to max physical antennas
Compression_RatioTx = 1; Compression_RatioRx = 1;
Sel_SAs_IndTx = 1:Q_T; Sel_SAs_IndRx = 1:Q_R;
Num_of_UsedSATx = length(Sel_SAs_IndTx); Num_of_UsedSARx = length(Sel_SAs_IndRx);

% Setup dynamic measurement tracking
MeasTx_vec = length(M_T_Beam);
MeasRx_vec = length(M_R_Beam);
M_T_measMAT = zeros(Q_T,MeasTx_vec); 
M_R_measMAT = zeros(Q_R,MeasRx_vec); 
M_T_measMAT(Sel_SAs_IndTx,:) =  repmat(Compression_RatioTx*M_T_Beam,Num_of_UsedSATx,1); 
M_R_measMAT(Sel_SAs_IndRx,:) =  repmat(Compression_RatioRx*M_R_Beam,Num_of_UsedSARx,1); 

%% Parameters related to the GRID of OMP/SOMP algorithm and Quantization of Phase
get_grid_quant_param;
%% Construct the Tx and Rx Dictionaries
get_TxRxDict;
%% Define the SNR
get_SNR_THzChannel;

%% Initialize Arrays for Results: NMSE
% --- CHANGED: Storage matrices sized by MeasTx_vec instead of SNR_len ---
NMSE_OMP = zeros(MeasTx_vec,E,K); 
NMSE_OMP_DR = zeros(MeasTx_vec,E,K); 
NMSE_SOMP = zeros(MeasTx_vec,E,K); 
NMSE_SOMP_DR = zeros(MeasTx_vec,E,K);

%% Main Loop
tic;
noise_pwr = N0_sc; % sigma^2
Tx_pwr = Tx_power_tot/Ns_train; % Fixed Tx Power

% --- CHANGED: Loop through measurements instead of SNR ---
for indx_meas = 1:MeasTx_vec
    % Provide the measurement index to the estimation script
    Tx_meas_indx = indx_meas;
    Rx_meas_indx = indx_meas;
    
    for indx_iter = 1:E
        % Display the current iteration
        [indx_meas indx_iter]
        
        % AoSAs THz channel generation
        CH_Response = channel_TIV(p_ch, K_abs);
        H_AoSA = cell2mat(CH_Response.H);        
        
        % Proposed and Conventional Estimation Estimation Methods
        get_proposed_HSPM_estimation;
        
        % Performance Evaluation Metrics: NMSE Computations for the UM-MIMO
        for indx_subc = 1:K
            NMSE_OMP(indx_meas,indx_iter,indx_subc) = norm(cell2mat(H_Est_OMP_UM(:,:,indx_subc))-cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2/norm(cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2;
            NMSE_OMP_DR(indx_meas,indx_iter,indx_subc) = norm(cell2mat(H_Est_OMP_DR_UM(:,:,indx_subc))-cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2/norm(cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2;
            NMSE_SOMP(indx_meas,indx_iter,indx_subc) = norm(cell2mat(H_Est_SOMP_UM(:,:,indx_subc))-cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2/norm(cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2;
            NMSE_SOMP_DR(indx_meas,indx_iter,indx_subc) = norm(cell2mat(H_Est_SOMP_DR_UM(:,:,indx_subc))-cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2/norm(cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2;
        end
    end
end
% Keep track of simulation time
Sim_Duration = toc;
hr = floor(Sim_Duration/3600);mint = floor((Sim_Duration - hr*3600)/60);sec = Sim_Duration - hr*3600 - mint*60;
fprintf('The simulation time is: %d hr %d min %f sec\n',hr,mint,sec);

%% Plot Results 
% --- CHANGED: Calculate the Compression Ratio for the X-axis ---
N_t = Q_T * Qbar_T; % Total number of transmit antennas
rho_array = sum(M_T_measMAT) / N_t;

% NMSE versus Compression Ratio (\rho)
figure('color',[1,1,1]);
plot(rho_array,10*log10(mean(NMSE_OMP,[2 3])),'ko-','linewidth',2,'MarkerSize',10);hold on;
plot(rho_array,10*log10(mean(NMSE_SOMP,[2 3])),'bx-','linewidth',2,'MarkerSize',10);
plot(rho_array,10*log10(mean(NMSE_OMP_DR,[2 3])),'ms-','linewidth',2,'MarkerSize',10);
plot(rho_array,10*log10(mean(NMSE_SOMP_DR,[2 3])),'rd-','linewidth',2,'MarkerSize',10);
xlabel('$\rho = N_p / N_t$','FontSize',14,'Interpreter','latex');
ylabel('NMSE (dB)','FontSize',14,'Interpreter','latex');
le=legend('OMP','SOMP','OMP-DR','SOMP-DR');le.Interpreter='latex';le.Location='northeast';le.FontSize=14;
ax=gca;ax.TickLabelInterpreter='latex';ax.FontSize=14;
box off;
grid on;
xlim([min(rho_array) max(rho_array)]);
axis tight;
axlims=axis;
x_range=axlims(2)-axlims(1);y_range=axlims(4)-axlims(3);loseness=5;
axis([axlims(1)-loseness/1e2*x_range axlims(2)+loseness/1e2*x_range axlims(3)-loseness/1e2*y_range axlims(4)+loseness/1e2*y_range]);