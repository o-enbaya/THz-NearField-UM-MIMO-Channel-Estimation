%% Dynamic Project Path Setup
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
projRoot = fileparts(thisDir);
if exist(fullfile(projRoot, 'setup_paths.m'), 'file')
    run(fullfile(projRoot, 'setup_paths.m'));
end

function genreate_thz_channel_universal(n_ch, d, delta, Delta, type, max_angle)
% =========================================================================
% One-file universal dataset generator with PARFOR + progress updates
%
% d:
%   scalar        -> fixed distance
%   [d_min d_max] -> random distance per realization (continuous range)
%   [d1 d2 d3...] -> uses a shuffled near-balanced assignment over exact
%                    discrete distances (perfectly balanced if n_ch is a
%                    multiple of the number of distances)
%
% Saves one .mat file containing:
%   H            : complex channel [Nr, Nt, K, n_ch]
%   d_samples    : distance used for each realization
%   angle_samples: random Rx angle shift per realization
%   meta         : metadata
% =========================================================================

tGlobal = tic;

%% -------------------- Input checks --------------------
if nargin < 6
    error('Usage: genreate_thz_channel_universal(n_ch, d, delta, Delta, type, max_angle)');
end

if ~isscalar(n_ch) || n_ch <= 0 || floor(n_ch) ~= n_ch
    error('n_ch must be a positive integer.');
end

if isscalar(d)
    if d <= 0
        error('Distance must be positive.');
    end
    d_mode = 'fixed';
    d_base = d;
    d_in = d;

elseif numel(d) == 2
    d = sort(d(:).');
    if d(1) <= 0
        error('Distance range must be positive.');
    end
    d_mode = 'range';
    d_base = d(1);
    d_in = d;

elseif numel(d) > 2
    d = sort(d(:).');
    if any(d <= 0)
        error('All distances must be positive.');
    end
    d_mode = 'discrete';
    d_base = d(1);
    d_in = d;

else
    error('d must be a scalar, a 2-element range [min max], or an array of discrete distances.');
end

%% -------------------- Parameters --------------------
p.channelType = type;

p.Fc = 0.3e12;
p.BW = 0.01e12;
p.Nsub_c = 2^3;
p.Nsub_b = 2^0;

% SAs
p.Mt = 2;  p.Nt = 2;
p.Mr = 2;  p.Nr = 2;

p.DeltaMt = Delta;
p.DeltaNt = Delta;
p.DeltaMr = Delta;
p.DeltaNr = Delta;

% AEs
p.Mat = 8;  p.Nat = 8;
p.Mar = 4;  p.Nar = 2;

p.deltaMt = delta;
p.deltaNt = delta;
p.deltaMr = delta;
p.deltaNr = delta;

% Wave model
p.WaveModelSA = 'Sphere';
p.WaveModelAE = 'Sphere';

% Geometry
p.positionTx = [0; 0; 0];
p.eulerTx    = [0; 0; 0];
p.positionRx = [d_base; 0; 0];
p.eulerRx    = [pi; 0; 0];

% Update channel parameters
p = update_channel_param_TIV(p);

%% -------------------- Absorption --------------------
K_abs = compute_Abs_Coef(p);

%% -------------------- Dimensions --------------------
K   = p.Nsub_c;
N_t = p.Qt * p.Qat;
N_r = p.Qr * p.Qar;

% Preallocation
H = complex(zeros(N_r, N_t, K, n_ch, 'single'));
d_samples = zeros(n_ch, 1, 'single');
angle_samples = zeros(3, n_ch, 'single');

% Filename string generation
if strcmp(d_mode, 'fixed')
    d_str = sprintf('%g', d_base);
elseif strcmp(d_mode, 'range')
    d_str = sprintf('%gto%g', d(1), d(2));
elseif strcmp(d_mode, 'discrete')
    if numel(d_in) >= 2
        diffs = diff(d_in);
        if all(abs(diffs - diffs(1)) < 1e-12)
            d_step = diffs(1);
            d_str = sprintf('%g_to_%g_step%g_discrete', min(d_in), max(d_in), d_step);
        else
            d_str = sprintf('%g_to_%g_%dpts_discrete', min(d_in), max(d_in), numel(d_in));
        end
    else
        d_str = sprintf('%g_discrete', d_in(1));
    end
end

fprintf('============================================================\n');
fprintf('Generating THz dataset\n');
fprintf('n_ch=%d, d=%s, delta=%g, Delta=%g, type=%s, angle=%g\n', ...
    n_ch, d_str, delta, Delta, type, max_angle);
fprintf('Array size: Nr=%d, Nt=%d, K=%d\n', N_r, N_t, K);
fprintf('============================================================\n');

%% -------------------- Start parallel pool --------------------
pool = gcp('nocreate');
if isempty(pool)
    c = parcluster('local');
    maxWorkers = c.NumWorkers;

    suggestedWorkers = min(maxWorkers, max(1, floor(feature('numcores') * 0.75)));
    pool = parpool('local', suggestedWorkers);
end

fprintf('Using %d parallel workers\n', pool.NumWorkers);

%% -------------------- Progress setup --------------------
D = parallel.pool.DataQueue;
progressCount = 0;
tLoop = tic;

printEvery = max(1, floor(n_ch / 20));
afterEach(D, @updateProgress);

%% -------------------- Pre-calculate Balanced Distances --------------------
dist_assignments = zeros(n_ch, 1, 'single');
if strcmp(d_mode, 'fixed')
    dist_assignments(:) = d_base;
elseif strcmp(d_mode, 'range')
    dist_assignments(:) = d(1) + (d(2) - d(1)) * rand(n_ch, 1);
elseif strcmp(d_mode, 'discrete')
    num_d = numel(d_in);
    
    if mod(n_ch, num_d) ~= 0
        warning(['n_ch = %d is not divisible by the number of discrete distances = %d. ' ...
                 'The dataset will be nearly balanced, not perfectly balanced.'], ...
                 n_ch, num_d);
    end
    
    for i = 1:n_ch
        dist_assignments(i) = d_in(mod(i-1, num_d) + 1);
    end
    dist_assignments = dist_assignments(randperm(n_ch));
end

%% -------------------- Monte Carlo generation --------------------
parfor n = 1:n_ch
    p_local = p;

    random_d = dist_assignments(n);
    p_local.positionRx = [random_d; 0; 0];

    angle_shift = -max_angle + 2 * max_angle * rand(3,1);
    p_local.eulerRx = p_local.eulerRx + angle_shift;

    p_local = update_channel_param_TIV(p_local);
    H_local = channel_TIV_AE_freq_domain_spherical(p_local, K_abs);

    H(:,:,:,n) = single(H_local);
    d_samples(n) = single(random_d);
    angle_samples(:,n) = single(angle_shift);

    send(D, 1);
end

fprintf('Generation loop finished.\n');

%% -------------------- Metadata --------------------
meta = struct();
meta.channelType = type;
meta.Fc = p.Fc;
meta.BW = p.BW;
meta.Nsub_c = p.Nsub_c;
meta.Nsub_b = p.Nsub_b;
meta.delta = delta;
meta.Delta = Delta;
meta.max_angle = max_angle;
meta.distance_mode = d_mode;
meta.distance_input = d_in;
meta.N_t = N_t;
meta.N_r = N_r;
meta.num_workers = pool.NumWorkers;

%% -------------------- Save --------------------
script_dir = fileparts(mfilename('fullpath'));
save_dir = fullfile(script_dir, 'data');

if exist(save_dir, 'file') == 2
    error('"data" exists as a file, not a folder. Rename or delete that file first.');
end

if ~exist(save_dir, 'dir')
    [ok, msg] = mkdir(save_dir);
    if ~ok
        error('Could not create data folder: %s', msg);
    end
end

filename = fullfile(save_dir, sprintf( ...
    'channel-r%dt%dk%d-n%dd%sdelta%gDelta%gtheta%g-%s-universal.mat', ...
    N_r, N_t, K, n_ch, d_str, delta, Delta, max_angle, type));

fprintf('Saving to:\n%s\n', filename);
save(filename, 'H', 'd_samples', 'angle_samples', 'meta', '-v7.3');

fprintf('Saved successfully.\n');
fprintf('Total time: %.2f seconds\n', toc(tGlobal));
fprintf('============================================================\n');

    %% -------------------- Nested progress function --------------------
    function updateProgress(~)
        progressCount = progressCount + 1;

        if progressCount == 1 || mod(progressCount, printEvery) == 0 || progressCount == n_ch
            elapsed = toc(tLoop);
            pct = 100 * progressCount / n_ch;
            rate = progressCount / max(elapsed, eps);
            eta = (n_ch - progressCount) / max(rate, eps);

            fprintf('Progress: %d/%d (%.1f%%) | Elapsed: %.1fs | ETA: %.1fs\n', ...
                progressCount, n_ch, pct, elapsed, eta);
        end
    end
end