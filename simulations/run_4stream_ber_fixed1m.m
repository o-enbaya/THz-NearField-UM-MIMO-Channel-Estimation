%% Dynamic Project Path Setup
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
projRoot = fileparts(thisDir);
if exist(fullfile(projRoot, 'setup_paths.m'), 'file')
    run(fullfile(projRoot, 'setup_paths.m'));
end

clear; clc;

% =========================================================
% 4-stream BER dataset generation at fixed d = 1 m
% =========================================================

% ---------- User settings ----------
n_ch_raw         = 20000;   % generate a large raw dataset first
d_fixed          = 1.0;     % fixed distance = 1 m
delta            = 0.0005;  % AE spacing
Delta            = 0.01;    % SA spacing
type             = 'Multipath';
max_angle        = pi/6;

% 4-stream subset rule
use_rank_filter  = true;    % true = create filtered 4-stream subset
rank_tol_4thmode = 1e-2;    % minimum normalized 4th singular mode

if ~exist('data', 'dir')
    mkdir('data');
end

fprintf('=========================================================\n');
fprintf('Generating fixed-1m dataset for 4-stream BER\n');
fprintf('=========================================================\n');

% ---------- Step 1: Generate raw fixed-1m dataset ----------
genreate_thz_channel(n_ch_raw, d_fixed, delta, Delta, type, max_angle);

% Expected filename from generator
N_t = 256;
N_r = 32;
K = 8;
d_str = sprintf('%g', d_fixed);

raw_file = sprintf( ...
    'data/channel-r%dt%dk%d-n%dd%sdelta%gDelta%gtheta%g-nearU-%s.mat', ...
    N_r, N_t, K, n_ch_raw, d_str, delta, Delta, max_angle, type);

fprintf('\nRaw dataset file:\n%s\n', raw_file);

% ---------- Step 2: Optional 4-stream filtering ----------
if use_rank_filter
    fprintf('\nLoading raw dataset and extracting 4-stream-capable subset...\n');
    S = load(raw_file);

    % Keep samples that are rank-4 across all subcarriers
    % and whose 4th normalized singular value is not too weak.
    idx4 = find(S.sample_min_rank >= 4 & S.sample_mode_worst_rel(:,4) >= rank_tol_4thmode);

    fprintf('Total raw samples     : %d\n', size(S.H, 4));
    fprintf('4-stream subset count : %d\n', numel(idx4));

    H = S.H(:,:,:,idx4);
    d_samples = S.d_samples(idx4);
    angle_samples = S.angle_samples(:,idx4);
    euler_samples = S.euler_samples(:,idx4);

    singular_values = S.singular_values(:,:,idx4);
    singular_values_rel = S.singular_values_rel(:,:,idx4);
    numerical_rank = S.numerical_rank(:,idx4);
    sample_min_rank = S.sample_min_rank(idx4);
    sample_avg_rank = S.sample_avg_rank(idx4);
    sample_mode_worst_rel = S.sample_mode_worst_rel(idx4,:);

    meta = S.meta;
    meta.filtered_for_4stream = true;
    meta.rank_tol_4thmode = rank_tol_4thmode;
    meta.original_file = raw_file;

    subset_file = sprintf( ...
        'data/channel-r%dt%dk%d-n%d-d%gm-4stream-delta%gDelta%gtheta%g-%s.mat', ...
        N_r, N_t, K, numel(idx4), d_fixed, delta, Delta, max_angle, type);

    save(subset_file, ...
        'H', ...
        'd_samples', ...
        'angle_samples', ...
        'euler_samples', ...
        'singular_values', ...
        'singular_values_rel', ...
        'numerical_rank', ...
        'sample_min_rank', ...
        'sample_avg_rank', ...
        'sample_mode_worst_rel', ...
        'meta', ...
        '-v7.3');

    fprintf('\nSaved 4-stream subset file:\n%s\n', subset_file);
else
    fprintf('\nRank filtering disabled. Use raw fixed-1m dataset directly for BER.\n');
end

% ---------- Save manifest ----------
manifest_ber = struct();
manifest_ber.experiment = 'BER_4stream_fixed_1m';
manifest_ber.n_ch_raw = n_ch_raw;
manifest_ber.d_fixed = d_fixed;
manifest_ber.delta = delta;
manifest_ber.Delta = Delta;
manifest_ber.type = type;
manifest_ber.max_angle = max_angle;
manifest_ber.use_rank_filter = use_rank_filter;
manifest_ber.rank_tol_4thmode = rank_tol_4thmode;
manifest_ber.raw_file = raw_file;

if use_rank_filter
    manifest_ber.subset_file = subset_file;
    manifest_ber.n_subset = numel(idx4);
end

save('data/manifest_ber_4stream_fixed1m.mat', 'manifest_ber');

fprintf('\nDone.\n');
fprintf('Saved manifest: data/manifest_ber_4stream_fixed1m.mat\n');