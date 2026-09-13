%% Dynamic Project Path Setup
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
projRoot = fileparts(thisDir);
if exist(fullfile(projRoot, 'setup_paths.m'), 'file')
    run(fullfile(projRoot, 'setup_paths.m'));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Modified NMSE & AR vs DISTANCE for Classical OMP/SOMP/DR Algorithms
% Matched EXACTLY to DCS parameters: L=4, E=100, M_T=25 (39% pilots), 64x8 AEs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;
%% Add Paths
path(pathdef); addpath(pwd);
cd Algorithms; addpath(genpath(pwd)); cd ..; 
cd TeraMIMO_Channel; addpath(genpath(pwd)); cd ..; 
cd Molecular_Absorption; addpath(genpath(pwd)); cd ..; 
cd Utilities; addpath(genpath(pwd)); cd ..;

%% Initialize Channel Parameters & Enforce DCS Hardware
p_ch = generate_channel_param_TIV();

% DCS Overrides
p_ch.L = 4;
p_ch.Nsub_c = 8;
p_ch.BW = 10e9;

p_ch.Mat = 64; p_ch.Nat = 1;
p_ch.Mar = 8;  p_ch.Nar = 1;

p_ch = update_channel_param_TIV(p_ch);
if exist('get_FrequencyRng_param', 'file') == 2
    p_ch = get_FrequencyRng_param(p_ch);
end
K_abs = get_Abs_Coef(p_ch);

get_array_trans_param;

% =========================================================================
% STRICT VERIFICATION CHECKS
% =========================================================================
assert(p_ch.Nsub_c == 8, 'Nsub_c override failed.');
assert(abs(p_ch.BW - 10e9) < 1, 'BW override failed.');
assert(p_ch.Tx_AoSA.Qbardim(1) == 64 && p_ch.Tx_AoSA.Qbardim(2) == 1, 'Tx AE-per-SA is not 64x1.');
assert(p_ch.Rx_AoSA.Qbardim(1) == 8  && p_ch.Rx_AoSA.Qbardim(2) == 1, 'Rx AE-per-SA is not 8x1.');

%% Main Simulation Parameters
E = 100;                 % DCS Iteration Match
Lbar = 10; 
RequiredSNR = 10;         % Fixed Rx SNR


% Strictly Near-Field Distances (0.4m to 1.5m, in steps of 0.1m)
Dist_Vec = 0.4:0.2:1.6; 
DistVec_Len = length(Dist_Vec);
DistVec_Len = length(Dist_Vec);

G_hv_TR_OVS = 2;         % <--- FIX 1: Dictionary Oversampling Factor
num_multG = 2; 
Q_T_quant = 2; 
Q_R_quant = 2; 

%% Measurements/Beams (DCS Pilot Match)
Ns_train = 1; 
Ns_data = min(Q_T,Q_R); 

% FIXED: 25 pilots per 64-element subarray (Matches DCS 39% compression)
M_T_Beam = 25; 
M_R_Beam = Qbar_R; 

Compression_RatioTx = 1; Compression_RatioRx = 1;
Sel_SAs_IndTx = 1:Q_T; Sel_SAs_IndRx = 1:Q_R;
Num_of_UsedSATx = length(Sel_SAs_IndTx); 
Num_of_UsedSARx = length(Sel_SAs_IndRx); 

M_T_measMAT = zeros(Q_T,1); M_R_measMAT = zeros(Q_R,1);
M_T_measMAT(Sel_SAs_IndTx,1) = Compression_RatioTx*M_T_Beam; 
M_R_measMAT(Sel_SAs_IndRx,1) = Compression_RatioRx*M_R_Beam; 

%% Construct the Tx and Rx Dictionaries
get_grid_quant_param;
get_TxRxDict;

% RAM FIX: Convert huge dictionaries to Single Precision
if exist('Tx_Dict','var'), Tx_Dict = single(Tx_Dict); end
if exist('Rx_Dict','var'), Rx_Dict = single(Rx_Dict); end
if exist('A_T_all','var'), A_T_all = single(A_T_all); end
if exist('A_R_all','var'), A_R_all = single(A_R_all); end

K = p_ch.Nsub_c;
B_sys = p_ch.BW;

%% Initialize Arrays for Results
% NMSE Arrays
NMSE_OMP = zeros(DistVec_Len, E, K, 'single'); 
NMSE_OMP_DR = zeros(DistVec_Len, E, K, 'single'); 
NMSE_SOMP = zeros(DistVec_Len, E, K, 'single'); 
NMSE_SOMP_DR = zeros(DistVec_Len, E, K, 'single');

% Achievable Rate Arrays
AR_PCSI = zeros(DistVec_Len, E, K, 'single');
AR_OMP = zeros(DistVec_Len, E, K, 'single');
AR_OMP_DR = zeros(DistVec_Len, E, K, 'single');
AR_SOMP = zeros(DistVec_Len, E, K, 'single');
AR_SOMP_DR = zeros(DistVec_Len, E, K, 'single');

%% Main Loop
tic;
for indx_dist = 1:DistVec_Len
    
    % Protect against internal workspace dependencies in helper scripts
    indx_snr = indx_dist; 
    
    % 1. PHYSICALLY MOVE THE RECEIVER
    p_ch.d_TR = Dist_Vec(indx_dist);
    p_ch.dist = Dist_Vec(indx_dist);
    p_ch.positionRx = [Dist_Vec(indx_dist); 0; 0];
    p_ch = update_channel_param_TIV(p_ch);
    
    % 2. DYNAMICALLY ADJUST POWER FOR THIS SPECIFIC DISTANCE
    lambda  = p_ch.lambda_c0;
    PLE_val = p_ch.PLE;
    d_TxRx3D = Dist_Vec(indx_dist);
    PL = ((4*pi*d_TxRx3D)./lambda).^PLE_val .* exp(K_abs(ceil(K/2),1)*d_TxRx3D);
    PL_dB = 10*log10(PL);
    
    N0_dBm = -173.8 + 10*log10(B_sys);
    N0_sc = 10.^((N0_dBm-30)/10);
    Tx_power_tot_dBm = 10*log10(K) + RequiredSNR + N0_dBm + PL_dB;
    Tx_power_tot = 10.^((Tx_power_tot_dBm-30)/10);
    
    noise_pwr = N0_sc; 
    Tx_pwr = Tx_power_tot/Ns_train;
    DataTx_pwr = Tx_pwr; % Required for AR Calculation
    
    for indx_iter = 1:E
        fprintf('Calculating Distance: %.1f m | Iteration: %d/%d\n', Dist_Vec(indx_dist), indx_iter, E);
        
        % AoSAs THz channel generation
        CH_Response = channel_TIV(p_ch, K_abs);
        H_AoSA = cell2mat(CH_Response.H);        
        
        % Initialize timer variables required by get_proposed_HSPM_estimation
        t_OMP_thiscall = 0; t_OMP_DR_thiscall = 0;
        t_SOMP_thiscall = 0; t_SOMP_DR_thiscall = 0;
        
        % Proposed and Conventional Estimation Methods
        get_proposed_HSPM_estimation;
        
        % ---------------------------------------------------------
        % NMSE Computations
        % ---------------------------------------------------------
        for indx_subc = 1:K
            NMSE_OMP(indx_dist,indx_iter,indx_subc) = norm(cell2mat(H_Est_OMP_UM(:,:,indx_subc))-cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2/norm(cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2;
            NMSE_OMP_DR(indx_dist,indx_iter,indx_subc) = norm(cell2mat(H_Est_OMP_DR_UM(:,:,indx_subc))-cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2/norm(cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2;
            NMSE_SOMP(indx_dist,indx_iter,indx_subc) = norm(cell2mat(H_Est_SOMP_UM(:,:,indx_subc))-cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2/norm(cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2;
            NMSE_SOMP_DR(indx_dist,indx_iter,indx_subc) = norm(cell2mat(H_Est_SOMP_DR_UM(:,:,indx_subc))-cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2/norm(cell2mat(H_SP_UM(:,:,indx_subc)),'fro')^2;
        end
        
      % ---------------------------------------------------------
        % Achievable Rate (AR) Computations
        % ---------------------------------------------------------
        H_AoSA_Cell = CH_Response.H;
        
        % 1. Perfect CSI Baseline Generation (ADAPTIVE SVD BACKUP)
        F_hat_bmf_PCSI = cell(Q_R, Q_T, K); W_hat_cmb_PCSI = cell(Q_R, Q_T, K);
        F_hat_bmf_PCSI(:) = {complex(zeros(Qbar_T,1))}; W_hat_cmb_PCSI(:) = {complex(zeros(Qbar_R,1))};
        
        for indx_UsedSATx = 1:Num_of_UsedSATx
            [qth,qtv] = ind2sub([Q_T_h Q_T_v],Sel_SAs_IndTx(indx_UsedSATx));
            for indx_UsedSARx = 1:Num_of_UsedSARx
                [qrh,qrv] = ind2sub([Q_R_h Q_R_v],Sel_SAs_IndRx(indx_UsedSARx));
                for indx_subc = 1:K
                    
                    % Check if Spherical Distances exist. If not, use Optimal SVD.
                    if isfield(CH_Response, 'Dist_LoS')
                        Dist_SWM_Ch = CH_Response.Dist_LoS{(qrv-1)*Q_R_h+qrh,(qtv-1)*Q_T_h+qth,indx_subc};
                        F_focus = get_NearField_ARV(Qbar_T, Dist_SWM_Ch(floor(Qbar_R/2),:).', Dist_Vec(indx_dist), p_ch.lambda_kc(indx_subc));
                        W_focus = get_NearField_ARV(Qbar_R, Dist_SWM_Ch(:,floor(Qbar_T)/2), Dist_Vec(indx_dist), p_ch.lambda_kc(indx_subc));
                        F_hat_bmf_PCSI{indx_UsedSARx,indx_UsedSATx,indx_subc} = conj(F_focus);
                        W_hat_cmb_PCSI{indx_UsedSARx,indx_UsedSATx,indx_subc} = W_focus;
                    else
                        % Optimal Perfect CSI via SVD (Works natively for Far-Field PWM)
                        H_exact = H_AoSA_Cell{(qrv-1)*Q_R_h+qrh,(qtv-1)*Q_T_h+qth,indx_subc};
                        [U_svd, ~, V_svd] = svd(H_exact);
                        F_hat_bmf_PCSI{indx_UsedSARx,indx_UsedSATx,indx_subc} = V_svd(:,1); 
                        W_hat_cmb_PCSI{indx_UsedSARx,indx_UsedSATx,indx_subc} = U_svd(:,1); 
                    end
                    
                end
            end
        end
        
        % FIX: Passed '1' instead of 'DataTx_pwr' to prevent double-power scaling
        AR_PCSI(indx_dist,indx_iter,:) = get_AchievableRate(p_ch, K, Tx_pwr, 1, noise_pwr, H_AoSA_Cell, H_AoSA_Cell, F_hat_bmf_PCSI, W_hat_cmb_PCSI);
        
        % 2. OMP & OMP-DR (3D Cell format)
        F_hat_bmf_OMP = cellfun(@conj, F_Sel_OMP, 'UniformOutput', false);
        AR_OMP(indx_dist,indx_iter,:) = get_AchievableRate(p_ch, K, Tx_pwr, 1, noise_pwr, H_Est_OMP_UM, H_AoSA_Cell, F_hat_bmf_OMP, W_Sel_OMP);
        
        F_hat_bmf_OMP_DR = cellfun(@conj, F_Sel_OMP_DR, 'UniformOutput', false);
        AR_OMP_DR(indx_dist,indx_iter,:) = get_AchievableRate(p_ch, K, Tx_pwr, 1, noise_pwr, H_Est_OMP_DR_UM, H_AoSA_Cell, F_hat_bmf_OMP_DR, W_Sel_OMP_DR);
        
        % 3. SOMP & SOMP-DR (2D Cell format mapped to 3D for subcarriers)
        F_hat_bmf_SOMP = repmat(cellfun(@conj, F_Sel_SOMP, 'UniformOutput', false), [1 1 K]);
        W_hat_cmb_SOMP = repmat(W_Sel_SOMP, [1 1 K]);
        AR_SOMP(indx_dist,indx_iter,:) = get_AchievableRate(p_ch, K, Tx_pwr, 1, noise_pwr, H_Est_SOMP_UM, H_AoSA_Cell, F_hat_bmf_SOMP, W_hat_cmb_SOMP);
        
        F_hat_bmf_SOMP_DR = repmat(cellfun(@conj, F_Sel_SOMP_DR, 'UniformOutput', false), [1 1 K]);
        W_hat_cmb_SOMP_DR = repmat(W_Sel_SOMP_DR, [1 1 K]);
        AR_SOMP_DR(indx_dist,indx_iter,:) = get_AchievableRate(p_ch, K, Tx_pwr, 1, noise_pwr, H_Est_SOMP_DR_UM, H_AoSA_Cell, F_hat_bmf_SOMP_DR, W_hat_cmb_SOMP_DR);
    end
end

% Keep track of simulation time
Sim_Duration = toc;
hr = floor(Sim_Duration/3600);mint = floor((Sim_Duration - hr*3600)/60);sec = Sim_Duration - hr*3600 - mint*60;
fprintf('The simulation time is: %d hr %d min %f sec\n',hr,mint,sec);

%% Plot Results 

% ---------------------------------------------------------
% 1. Plot NMSE vs Distance
% ---------------------------------------------------------
figure('color',[1,1,1]);
plot(Dist_Vec, 10*log10(mean(NMSE_OMP,[2 3])), 'ko-','linewidth',2,'MarkerSize',10); hold on;
plot(Dist_Vec, 10*log10(mean(NMSE_SOMP,[2 3])), 'bx-','linewidth',2,'MarkerSize',10);
plot(Dist_Vec, 10*log10(mean(NMSE_OMP_DR,[2 3])), 'ms-','linewidth',2,'MarkerSize',10);
plot(Dist_Vec, 10*log10(mean(NMSE_SOMP_DR,[2 3])), 'rd-','linewidth',2,'MarkerSize',10);

xlabel('Distance (m)','FontSize',14,'Interpreter','latex');
ylabel('NMSE (dB)','FontSize',14,'Interpreter','latex');
le = legend('OMP','SOMP','OMP-DR','SOMP-DR');
le.Interpreter = 'latex'; le.Location = 'northeast'; le.FontSize = 14;

ax = gca; ax.TickLabelInterpreter = 'latex'; ax.FontSize = 14;
box off; grid on;
xlim([0.4 1.6]); % PERFECTLY ZOOMED X-AXIS FOR NEAR-FIELD

% ---------------------------------------------------------
% 2. Plot Achievable Rate vs Distance
% ---------------------------------------------------------
figure('color',[1,1,1]);
plot(Dist_Vec, mean(AR_PCSI,[2 3]), '-pentagram', 'Color', [0 0 0.5], 'linewidth', 2, 'MarkerSize', 10); hold on;
plot(Dist_Vec, mean(AR_OMP,[2 3]), 'ko-','linewidth',2,'MarkerSize',10); hold on;
plot(Dist_Vec, mean(AR_SOMP,[2 3]), 'bx-','linewidth',2,'MarkerSize',10);
plot(Dist_Vec, mean(AR_OMP_DR,[2 3]), 'ms-','linewidth',2,'MarkerSize',10);
plot(Dist_Vec, mean(AR_SOMP_DR,[2 3]), 'rd-','linewidth',2,'MarkerSize',10);

xlabel('Distance (m)', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('Achievable Rate (bits/sec/Hz)', 'FontSize', 14, 'Interpreter', 'latex');

% Custom legend to match formatting
hplt = zeros(5,1);
hplt(1) = plot(NaN,NaN,'-pentagram','Color',[0 0 0.5],'linewidth',2,'MarkerSize',7);
hplt(2) = plot(NaN,NaN,'ko-','linewidth',2,'MarkerSize',7);
hplt(3) = plot(NaN,NaN,'bx-','linewidth',2,'MarkerSize',7);
hplt(4) = plot(NaN,NaN,'ms-','linewidth',2,'MarkerSize',7);
hplt(5) = plot(NaN,NaN,'rd-','linewidth',2,'MarkerSize',7);

L_lngd = legend(hplt, 'PCSI', 'OMP', 'SOMP', 'OMP-DR', 'SOMP-DR');
L_lngd.Interpreter = 'latex'; L_lngd.Location = 'northeast'; L_lngd.FontSize = 10;

ax = gca; ax.TickLabelInterpreter = 'latex'; ax.FontSize = 14;
box off; grid on;
xlim([0.4 1.6]); % PERFECTLY ZOOMED X-AXIS FOR NEAR-FIELD