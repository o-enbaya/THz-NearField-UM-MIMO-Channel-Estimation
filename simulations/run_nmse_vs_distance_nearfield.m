%% Dynamic Project Path Setup
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
projRoot = fileparts(thisDir);
if exist(fullfile(projRoot, 'setup_paths.m'), 'file')
    run(fullfile(projRoot, 'setup_paths.m'));
end

clear; clc;
clc; clear; close all;

% Dynamic workspace path resolution
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
addpath(genpath(thisDir));
% or:
% =========================================================
% NMSE vs Distance dataset generation (near-field only)
% One fixed distance per run
% =========================================================

% ---------- User settings ----------
n_ch       = 5000;          % number of channel realizations per distance
d_list     = 0.4:0.2:1.6;   % near-field sweep
delta      = 0.0005;        % AE spacing
Delta      = 0.01;          % SA spacing
type       = 'Multipath';   % 'LoS', 'Multipath', or 'Multipath+LoS'
max_angle  = pi/6;          % Rx rotation limit

% ---------- Output bookkeeping ----------
generated_files = cell(length(d_list), 1);

if ~exist('data', 'dir')
    mkdir('data');
end

fprintf('=========================================================\n');
fprintf('Generating NMSE-vs-distance datasets\n');
fprintf('Distances: ');
fprintf('%.2f ', d_list);
fprintf('\n=========================================================\n');

for ii = 1:length(d_list)
    d = d_list(ii);

    fprintf('\n[%d/%d] Generating fixed-distance dataset at d = %.2f m\n', ...
        ii, length(d_list), d);

    % Generate one dataset at one fixed distance
    genreate_thz_channel(n_ch, d, delta, Delta, type, max_angle);

    % Reconstruct expected filename from generator format
    % Note: this matches the near-field generator I gave earlier.
    % If you changed the filename pattern, update this line accordingly.
    tmp_p = struct();
    tmp_p.Nsub_c = 8;

    % Dimensions from your generator setup:
    % Tx total AEs = (Mt*Nt)*(Mat*Nat) = 4*64 = 256
    % Rx total AEs = (Mr*Nr)*(Mar*Nar) = 4*8  = 32
    N_t = 256;
    N_r = 32;

    d_str = sprintf('%g', d);

    generated_files{ii} = sprintf( ...
        'data/channel-r%dt%dk%d-n%dd%sdelta%gDelta%gtheta%g-nearU-%s.mat', ...
        N_r, N_t, tmp_p.Nsub_c, n_ch, d_str, delta, Delta, max_angle, type);
end

% Save manifest for later NMSE evaluation
manifest_nmse = struct();
manifest_nmse.experiment = 'NMSE_vs_distance_nearfield';
manifest_nmse.n_ch = n_ch;
manifest_nmse.d_list = d_list;
manifest_nmse.delta = delta;
manifest_nmse.Delta = Delta;
manifest_nmse.type = type;
manifest_nmse.max_angle = max_angle;
manifest_nmse.generated_files = generated_files;

save('data/manifest_nmse_vs_distance_nearfield.mat', 'manifest_nmse');

fprintf('\nDone.\n');
fprintf('Saved manifest: data/manifest_nmse_vs_distance_nearfield.mat\n');