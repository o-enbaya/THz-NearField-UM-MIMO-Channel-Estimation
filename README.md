# Near-Field Channel Estimation in THz UM-MIMO Using Deep Compressed Sensing

[![MATLAB](https://img.shields.io/badge/MATLAB-R2022b%2B-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Bachelor's Thesis](https://img.shields.io/badge/Bachelor's%20Thesis-EE599%20Graduation%20Project-success.svg)](documentation/EE599_Final_Project_Report.pdf)

Official implementation, simulation engine, and deep learning framework for the graduation project Bachelor's thesis:
> **"Near-field Channel Estimation in THz UM-MIMO Using Deep Compressed Sensing"**  
> **Authors**: **Osama Younes Enbaya** & **Zaher Samir AlGndi**  
> **Supervisor**: Eng. Aya H.A. Alsmoee  
> *Department of Electrical and Electronic Engineering, Faculty of Engineering, University of Tripoli (EE599)*  
> *Academic Year: 2025 -- 2026*

> **Corrections (October 2026):** Some statements in Chapter 4 of the report, and several numbers previously in this README, did not match the plotted results or the scripts (BER text, distance explanation, runtime units, DR discussion). The report PDF is kept as submitted. All corrections are listed in [Section 8](#8-corrections-to-the-thesis-report).

---

## Table of Contents
1. [System & Architecture Overview](#1-system--architecture-overview)
2. [Prerequisites & System Requirements](#2-prerequisites--system-requirements)
3. [Repository Directory Structure](#3-repository-directory-structure)
4. [Deep Compressed Sensing (DCS) Pipeline](#4-deep-compressed-sensing-dcs-pipeline)
   - [Phase 0: Dataset Generation & Conversion](#phase-0-dataset-generation--conversion)
   - [Phase 1: Offline Meta-Training (Algorithm 1)](#phase-1-offline-meta-training-algorithm-1)
   - [Phase 2: Online Real-Time Inference (Algorithm 2)](#phase-2-online-real-time-inference-algorithm-2)
   - [Bachelor's Thesis Chapter 4 Python Evaluation Scripts](#bachelors-thesis-chapter-4-python-evaluation-scripts)
5. [MATLAB Conventional Compressive Sensing Pipeline](#5-matlab-conventional-compressive-sensing-pipeline)
   - [Step 1: Environment & Path Setup](#step-1-environment--path-setup)
   - [Step 2: Physics-Based TeraMIMO Channel Engine](#step-2-physics-based-teramimo-channel-engine)
   - [Step 3: Conventional Sparse Recovery Algorithms](#step-3-conventional-sparse-recovery-algorithms)
   - [Step 4: Running MATLAB Simulations](#step-4-running-matlab-simulations)
   - [Step 5: Generating Publication-Quality Figures](#step-5-generating-publication-quality-figures)
6. [Comparative Evaluation & Key Metrics](#6-comparative-evaluation--key-metrics)
   - [6.1 Evaluation Metrics Defined & Explained](#61-evaluation-metrics-defined--explained)
   - [6.2 Master Performance Comparison Matrix](#62-master-performance-comparison-matrix)
   - [6.3 Key Bachelor's Thesis Findings](#63-key-bachelors-thesis-findings)
7. [Citation & Academic Reference](#7-citation--academic-reference)
8. [Corrections to the Thesis Report](#8-corrections-to-the-thesis-report)

---

## 1. System & Architecture Overview

The system models an **Ultra-Massive Multiple-Input Multiple-Output (UM-MIMO)** Terahertz (THz) communication link operating within the radiative **near-field region**. The link distances used ($0.4$ to $1.6\text{ m}$) are below the Rayleigh distance of even a single $64$-element transmit subarray ($2D^2/\lambda \approx 2\text{ m}$ at $300\text{ GHz}$). 

Because the transceiver operates at $300\text{ GHz}$ with massive antenna apertures, the traditional planar wavefront assumption fails. Electromagnetic waves exhibit distinct **spherical wavefront curvature**, introducing a dual-dependence on both spatial angles $(\theta, \phi)$ and radial propagation distance $r$.

```
                    TERAHERTZ NEAR-FIELD CHANNEL ESTIMATION
                    
   +-----------------------+                    +-----------------------+
   |   Transmitter (Tx)    |                    |    Receiver (Rx)      |
   |   N_t = 256 antennas  |   Spherical Wave   |   N_r = 32 antennas   |
   |   (4x64 AoSA subarrays| ~~~~~~~~~~~~~~~~~>|   (4x8 AoSA subarrays |
   |   N_RF = 4 chains     |   fc = 300 GHz     |   N_RF = 4 chains     |
   +-----------------------+   B = 10 GHz       +-----------------------+
              |                                             |
              +===================+=========================+
                                  |
               +------------------+------------------+
               |                                     |
    [Conventional Sparse CS]            [Deep Compressed Sensing (DCS)]
    - TeraMIMO Channel Engine           - Continuous Generative Prior G_theta(z)
    - Spherical Wavefront Steering      - 1-Bit RF Quantization Phi Operator
    - OMP & SOMP Sparse Recovery        - Offline MAML + RIP Auxiliary Loss
    - Dictionary Reduction (DR)         - Online Latent Space Adam Optimization
```

### Table 4.1: Physical & Hardware Parameters
| Parameter | Symbol | Value | Physical Significance / Description |
| :--- | :--- | :--- | :--- |
| **Carrier Frequency** | $f_c$ | $300\text{ GHz}$ | Sub-THz low-absorption atmospheric transmission window |
| **Bandwidth** | $B$ | $10\text{ GHz}$ | Ultra-wideband Terahertz transmission |
| **Subcarriers** | $K$ | $8$ | Wideband Orthogonal Frequency Division Multiplexing (OFDM) |
| **Transmit Antennas** | $N_t$ | $256$ | $Q_T = 4$ subarrays $\times 64$ Antennas ($64\times 1$ ULA per subarray) |
| **Receive Antennas** | $N_r$ | $32$ | $Q_R = 4$ subarrays $\times 8$ Antennas ($8\times 1$ ULA per subarray) |
| **RF Chains / Streams** | $N_{\mathrm{RF}}, N_s$ | $4, (1, 2)$ | Array of Subarrays (AoSA) sub-connected hybrid beamforming |
| **Antenna Spacing** | $d_{\mathrm{AE}}$ | $0.5\text{ mm}$ | Half-wavelength ($\\lambda / 2$) at $300\text{ GHz}$ |
| **Subarray Separation** | $d_{\mathrm{SA}}$ | $10\text{ mm}$ | Physical separation between AoSA modules ($10\lambda_c$) |
| **Operating Distance** | $d$ | $1.0\text{ m}$ | Radiative near field (below the $\approx 2\text{ m}$ Rayleigh distance of one $64$-element subarray) |
| **Multipath Channels** | $L$ | $4$ | 1 dominant Line-of-Sight (LoS) + 3 Non-LoS (NLoS) clusters |

---

## 2. Prerequisites & System Requirements

### A. Hardware Specifications
* **GPU**: NVIDIA GPU with CUDA support (e.g., GTX 1660, RTX 2060/3060/4070/4090, V100, A100, or Google Colab GPU).
  * **Training VRAM**: $\ge 6\text{ GB}$ recommended for training batch sizes of $50\text{--}100$.
  * **Inference/Evaluation**: GPU or standard multi-core CPU. In our runtime test, online inference took $\approx 0.36\text{ s}$ per channel estimate (not ms; see [Section 8](#8-corrections-to-the-thesis-report)).
* **System RAM**: $\ge 16\text{ GB}$ recommended (for processing full $5000$-sample channel tensors in memory).
* **Storage**: $\ge 5\text{ GB}$ free disk space (for dataset tensors, checkpoints, and simulation outputs).
* **Operating System**: Windows 10/11 (64-bit) or Linux (Ubuntu 20.04/22.04 LTS).

---

### B. Python Environment (Deep Learning Pipeline)
* **Python Version**: `3.10` or `3.11` (Python 3.12+ users should ensure PyTorch wheels are supported).
* **PyTorch Version**: `2.0.0` or later with CUDA support (tested with PyTorch 2.0 -- 2.4, CUDA 11.8 / 12.1).

#### Required Python Packages (`requirements.txt`)
```txt
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
scipy>=1.10.0
numpy>=1.24.0
matplotlib>=3.7.0
tensorboard>=2.12.0
h5py>=3.8.0
tqdm>=4.65.0
```

#### Setup Option 1: Conda Environment (Recommended)
```bash
# Using the automated shell script:
chmod +x python_dcs/create-conda-env.sh
./python_dcs/create-conda-env.sh

# Or directly from the YAML definition:
conda env create -f python_dcs/environment.yml
conda activate dcs-thz
```

#### Setup Option 2: Pip & Virtualenv
```bash
# Create and activate a Python virtual environment:
# On Linux/macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.1 acceleration:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the remaining dependencies:
pip install -r python_dcs/requirements.txt
```

---

### C. MATLAB Environment (Conventional CS & TeraMIMO)
* **MATLAB Version**: `R2022b` or later (tested on R2022b, R2023a/b, R2024a).
* **Required MATLAB Toolboxes**:
  * Signal Processing Toolbox
  * Communications Toolbox
  * Phased Array System Toolbox (optional, for spatial steering visualization)
  * Deep Learning Toolbox (optional, only if inspecting neural networks within MATLAB)

---

## 3. Repository Directory Structure

```
THz-NearField-UM-MIMO-Channel-Estimation/
│
├── README.md                            # Comprehensive project guide & documentation
├── setup_paths.m                        # Universal path configuration script for MATLAB
│
├── channel_simulator/                   # TeraMIMO Physics & Molecular Absorption Engine
│   ├── Molecular_Absorption/            # Hitran-based spectroscopic atmospheric absorption
│   │   ├── Data/                        # Molecular spectral line data (H2O, O2, N2, CO2, CH4)
│   │   ├── get_Abs_Coef.m               # Molecular absorption coefficient (HITRAN based)
│   │   └── Gas_Abs_Coef.m               # Atmospheric transmission path loss
│   ├── TeraMIMO_Channel/                # Near-field spherical-wave channel engine (TeraMIMO, Tarboush et al.)
│   │   ├── ChannelGeneration/           # Channel generation, LoS distances/angles, path loss
│   │   ├── InitializeParameters/        # AoSA, geometry, frequency and HITRAN parameters
│   │   └── Ch_Utilities/                # Direction vector and angle utilities
│   └── Utilities/                       # Dictionaries, codebooks, SNR setup, achievable rate
│
├── Algorithms/                          # Conventional CS baselines (code by S. Tarboush, ICASSP 2023)
│   ├── OMP_SMC.m                        # OMP solved separately for each subcarrier (SMC = separate multi-carriers)
│   ├── SOMP.m                           # Simultaneous OMP, one joint problem for all subcarriers
│   ├── OMP_SMC_DR.m                     # Per-subcarrier OMP with dictionary reduction (DR)
│   ├── get_proposed_DR.m                # Builds the reduced (localized) sub-dictionaries
│   └── get_proposed_HSPM_estimation.m   # Estimation loop over subarray pairs (hybrid spherical/planar-wave model, HSPM)
│
├── simulations/                         # Automated Simulation Testbenches
│   ├── NMSEvsSNR.m                      # Evaluates Figure 4.2 (NMSE vs Transmit SNR)
│   ├── EARvsNumberofMeasurements.m      # Evaluates Figure 4.3 (Effective Achievable Rate)
│   ├── NMSEvsRho.m                      # Evaluates Figure 4.4 (NMSE vs Compression Ratio)
│   ├── BERvsSNR_Cal.m                   # Earlier BER version (Figure 4.5 was made by thesis_figure_scripts/matlab/Untitled4.m and NS2.m)
│   ├── BERvsSNR_QPSK.m                  # Monte Carlo QPSK transmission simulation (not used in the thesis figures)
│   ├── NMSE_EAR_Vs_Distance.m           # Evaluates Figure 4.6 (NMSE and AR vs distance, baselines, fixed 10 dB SNR)
│   ├── run_nmse_vs_distance_nearfield.m # Generates per-distance channel datasets (DCS distance study, not in the thesis)
│   ├── Complexity.m                     # Evaluates Table 4.3 (Analytical FLOPs)
│   ├── Runtime.m                        # Plots Figure 4.7(b) from the measured runtimes (values entered by hand)
│   └── genreate_thz_channel_universal.m # Synthetic channel realization generator
│
├── plotting/                            # Earlier plot versions (final thesis plots: thesis_figure_scripts/matlab/plotting)
│   ├── plot_NMSE_Results.m              # Figure 4.2 (earlier style)
│   ├── plot_EAR_Results.m               # Figure 4.3 (earlier style)
│   ├── plot_NMSE_vs_Rho.m               # Figure 4.4
│   └── plot_Distance_Sweep_Results.m    # Figure 4.6
│
├── thesis_figure_scripts/               # Unchanged originals of the scripts that produced every thesis figure, with an index
│
├── results/                             # Centralized Simulation Workspaces & Results
│   ├── data/                            # Raw simulation MAT-files (.mat)
│   ├── csv/                             # Result data (.csv), including some earlier runs (see Section 8)
│   └── figures/                         # Exported publication plots (.png, .pdf, .fig)
│
└── python_dcs/                          # PyTorch Deep Compressed Sensing (DCS) Pipeline
    ├── environment.yml                  # Official conda environment specification
    ├── create-conda-env.sh              # Bash script to create and configure conda env
    ├── requirements.txt                 # Python dependencies
    ├── model.py                         # Generator G_theta and Discriminator neural networks
    ├── utils.py                         # Complex-real conversions, ChannelDataset, measurement matrices
    ├── utils_ear.py                     # Effective Achievable Rate numerical utilities
    │
    ├── training/                        # Offline Meta-Training Pipeline
    │   ├── dcs_train.py                 # Algorithm 1: MAML offline training loop (300 epochs)
    │   ├── dcs_finetune.py              # Latent space fine-tuning with L1 loss
    │   └── mat2pt.py                    # Converts MATLAB .mat channels to PyTorch .pt tensors
    │
    ├── evaluation/                      # Bachelor's Thesis Chapter 4 Figure Reproduction Scripts
    │   ├── eval_fig4_2_nmse_vs_snr.py   # Evaluates Figure 4.2 (NMSE vs Transmit SNR)
    │   ├── eval_fig4_3_ear_vs_meas.py   # Evaluates Figure 4.3 (EAR vs Pilot Overhead)
    │   ├── plot_dcsEAR.py               # Plots Figure 4.3 EAR curves
    │   ├── eval_fig4_4_nmse_vs_rho.py   # Evaluates Figure 4.4 (NMSE vs Compression Ratio)
    │   ├── eval_fig4_5_ber_vs_snr.py    # DCS BER vs SNR (latest exploratory version: QPSK, Ns = 2; see Section 8)
    │   ├── eval_fig4_6_distance_sweep.py# Exploratory DCS distance sweep (DCS is not in thesis Figure 4.6)
    │   ├── eval_fig4_7_runtime.py       # Measures the DCS runtime of Figure 4.7(b)
    │   └── eval_runtime.py              # Prints Table 4.3 FLOPs and the measured runtimes
    │
    ├── checkpoints/                     # Model weights directory (gitignored for large files)
    │   └── README.md                    # Instructions for pre-trained .pth.tar checkpoints
    └── data/                            # Evaluation result CSV files
```

---

## 4. Deep Compressed Sensing (DCS) Pipeline

The Deep Compressed Sensing framework replaces discrete angular/polar dictionaries with a **continuous generative prior** $G_\theta(z)$, completely eliminating off-grid quantization errors.

The pipeline operates in two distinct phases:
1. **Offline Meta-Learning (Algorithm 1)**: Meta-trains the generator $G_\theta$ across diverse channel realizations using Model-Agnostic Meta-Learning (MAML) and an auxiliary Restricted Isometry Property (RIP) loss.
2. **Online Real-Time Inference (Algorithm 2)**: Freezes network weights $\theta^{\ast}$ and reconstructs the channel from compressed pilot observations by optimizing only a low-dimensional latent vector $z$ via gradient descent ($T = 10\text{--}20$ steps; $\approx 0.36\text{ s}$ per estimate in our runtime test).

```
[Phase 0: Generation]             [Phase 1: Offline Training]            [Phase 2: Online Inference]
TeraMIMO Simulator (MATLAB) ---> MAML Meta-Training (dcs_train.py) ---> Freeze Weights theta*
         |                                    |                                    |
         v                                    v                                    v
Save channel .mat file           Train G_theta with RIP Loss L_F        Receive Pilots y = Phi*h + n
         |                                    |                                    |
         v                                    v                                    v
Convert to .pt (mat2pt.py)       Save Checkpoint (.pth.tar)            Optimize latent z (10 steps)
                                                                                   |
                                                                                   v
                                                                       Synthesize H_hat = G(z*)
```

---

### Phase 0: Dataset Generation & Conversion

#### 1. Generate Synthetic Near-Field Channels in MATLAB
Open MATLAB and run:
```matlab
setup_paths;

% Generate 5,000 near-field channel realizations at d = 1.0 m, theta = 30 deg (pi/6):
genreate_thz_channel(5000, 1.0, 0.0005, 0.01, 'Multipath+LoS', pi/6);
```
* **Output**: A MAT-file containing the 4-dimensional channel array `H_all` with shape $(N_r, N_t, K, N_{\mathrm{samples}}) = (32, 256, 8, 5000)$.

#### 2. Convert `.mat` to PyTorch `.pt` Tensor
Run `mat2pt.py` to format the complex channels into real/imaginary tensor channels $(N, 2K, N_r, N_t) = (5000, 16, 32, 256)$ and compute channel normalization statistics $(\mu, \sigma)$:
```bash
python -m python_dcs.training.mat2pt   --input data/channel-r32t256k8-n5000d1.2delta0.0005Delta0.01theta0.523599-Multipath+LoS.mat   --output python_dcs/data/channel-r32t256k8-n5000.pt
```

---

### Phase 1: Offline Meta-Training (Algorithm 1)

In the offline stage, the generative network $G_\theta(z)$ is trained using bi-level optimization via Model-Agnostic Meta-Learning (MAML):

$$
\min_{\theta} \; \frac{1}{B} \sum_{i=1}^B \left( \mathcal{L}_G\left(\mathbf{h}_i, G_\theta(z_i^{\ast})\right) + \lambda_F \cdot \mathcal{L}_{\mathrm{RIP}} \right)
$$

#### What each term represents:
* **$\mathbf{h}_i = \mathrm{vec}(\mathbf{H}_i)$**: The vectorized ground-truth near-field channel for sample task $i$ ($N_r N_t K = 32 \times 256 \times 8 = 65,536$ complex elements).
* **$B = 100$**: The outer-loop **batch size** of distinct channel realization tasks sampled from the training set.
* **$z_i^{\ast} \in \mathbb{R}^{d_z}$ ($d_z = 100$)**: The **optimized latent vector** found by the inner loop for task $i$.
* **$\mathcal{L}_G(\mathbf{h}_i, G_\theta(z_i^{\ast})) = \frac{\|\mathbf{h}_i - G_\theta(z_i^{\ast})\|_2^2}{\|\mathbf{h}_i\|_2^2}$**: The **full-channel reconstruction loss** (Normalized Mean Squared Error) between the true channel and the generator output.
* **$\mathcal{L}_{\mathrm{RIP}}$**: An **auxiliary Restricted Isometry Property (RIP) triplet loss** enforcing metric preservation across generated channel pairs ($(\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3)$), preventing latent manifold distortion and mode collapse.
* **$\lambda_F$**: Regularization weighting factor balancing RIP geometric preservation with reconstruction fidelity.
* **$\theta$**: The learnable convolutional weights and biases of generator $G_\theta$, updated via Adam (learning rate $\alpha_2 = 0.0002$).

#### The Inner Optimization Loop (Task Adaptation):
For each sampled channel task $i$, a latent vector $z_i^{(0)} \sim \mathcal{N}(0, \mathbf{I}_{d_z})$ is initialized and optimized over $T = 10$ steps using Adam (inner learning rate $\alpha_1 = 0.1$):

$$
\mathcal{L}_{\mathrm{inner}}(z_i) = \|\tilde{\mathbf{y}}_i - \mathbf{\Phi} G_\theta(z_i)\|_2^2 + \lambda_z \|z_i\|_2^2
$$

where $\tilde{\mathbf{y}}_i = \mathbf{\Phi} \mathbf{h}_i + \tilde{\mathbf{n}}$ is the simulated pilot measurement vector, and $z_i$ is projected onto the unit sphere after every step ($z_i \leftarrow z_i / \|z_i\|_2$).

#### Execute Offline Training:
```bash
python -m python_dcs.training.dcs_train   --n_epochs 300   --batch_size 100   --lr 0.0002   --lr_est 0.1   --latent_dim 100   --gd_step 10   --n_p 100   --sample_ckpt 5
```

#### Training Parameter Reference:
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--n_epochs` | `300` | Total number of outer training epochs |
| `--batch_size` | `100` | Channel batch size per outer update |
| `--lr` | `0.0002` | Outer loop Adam learning rate for generator weights $\theta$ |
| `--lr_est` | `0.1` | Inner loop Adam learning rate for latent vector $z$ |
| `--latent_dim` | `100` | Latent space dimension ($d_z = 100$) |
| `--img_size` | `16 32 256` | Tensor dimensions: $2K=16$ channels, $N_r=32$, $N_t=256$ |
| `--gd_step` | `10` | Inner loop gradient descent steps ($T = 10$) |
| `--n_p` | `100` | Number of pilot beam measurements ($N_p = 100$) |
| `--sample_ckpt` | `5` | Save checkpoint every $N$ epochs |

#### Monitor Training with TensorBoard:
```bash
tensorboard --logdir python_dcs/logs
# Open your browser at http://localhost:6006 to inspect loss curves and sample reconstructions
```

#### (Optional) Precision Latent Fine-Tuning:
To sharpen the NMSE below $-16.7\text{ dB}$ without gradient oscillation, run the second-stage precision pass using L1 loss:
```bash
python -m python_dcs.training.dcs_finetune
```

#### Checkpoints & Pre-Trained Weights:
* Checkpoint weights are saved to `python_dcs/checkpoints/` (e.g., `e300b50gd10np100dl100.pth.tar`).
* If you do not wish to train the model from scratch, pre-trained weights can be downloaded and placed directly into `python_dcs/checkpoints/`.

---

### Phase 2: Online Real-Time Inference (Algorithm 2)

During online operation at the base station or user terminal, the receiver estimates the unknown near-field channel from real-time compressed pilot observations. Crucially, the generator network weights $\theta^{\ast}$ are **frozen**: they are not updated. Note that computing the gradient with respect to $z$ still requires a backward pass through the generator at every step, so each step costs roughly one forward and one backward pass.

#### Step-by-Step Online Execution Flow

1. **Pilot Transmission**:
   The transmitter transmits $N_p = 100$ pilot beams across the near-field propagation channel.

2. **Compressed Baseband Observation**:
   The receiver captures compressed observations across all $K = 8$ OFDM subcarriers:

$$
\tilde{\mathbf{y}} = \mathbf{\Phi} \mathbf{h} + \tilde{\mathbf{n}}
$$

3. **Frozen Network Weights**:
   The pre-trained generator weights $\theta^{\ast}$ are **frozen** (not updated during deployment). Gradients are still back-propagated through the generator to reach $z$.

4. **Low-Dimensional Latent Optimization**:
   The receiver searches for the optimal latent vector $z^{\ast} \in \mathbb{R}^{d_z}$ ($d_z = 100$) that minimizes the pilot measurement fitting error:

$$
z^{\ast} = \arg\min_{\|z\|_2 = 1} \; \|\tilde{\mathbf{y}} - \mathbf{\Phi} G_{\theta^{\ast}}(z)\|_2^2
$$

   (The implementation has no explicit regularization term on $z$; instead, $z$ is normalized to unit length after every step.)

5. **Gradient Updates ($T = 10\text{--}20$ steps, depending on the experiment)**:
   The latent code is iteratively updated via Adam gradient descent and projected onto the unit hypersphere:

$$
z^{(t+1)} \leftarrow \frac{z^{(t)} - \alpha_{\mathrm{online}} \cdot \nabla_z \mathcal{L}(z^{(t)})}{\|z^{(t)} - \alpha_{\mathrm{online}} \cdot \nabla_z \mathcal{L}(z^{(t)})\|_2}
$$

6. **Single-Pass Channel Reconstruction**:
   The estimated full-dimensional channel tensor $\hat{\mathbf{H}}$ is synthesized in a single forward evaluation:

$$
\hat{\mathbf{H}} = G_{\theta^{\ast}}(z^{\ast})
$$

7. **Execution Time**:
   With $T = 20$ steps, the measured runtime was **$\approx 0.36\text{ s}$ per channel estimate** (Figure 4.7(b)), about $2\times$ faster than SOMP ($0.77\text{ s}$) in our simulation code. These are unoptimized research scripts, so the absolute times are far from real-time and should only be compared with each other. See [Section 8](#8-corrections-to-the-thesis-report) for the timing procedure.

---

#### Comprehensive Mathematical & Symbol Breakdown

* **$\tilde{\mathbf{y}} \in \mathbb{C}^{M_r N_p \times 1}$**: The **compressed pilot measurement vector** received at the baseband output across all $N_p = 100$ pilot transmissions. With $M_r = 4$ RF chains at the receiver, this vector contains only $M_r \times N_p = 4 \times 100 = 400$ complex measurements.
* **$\mathbf{h} = \mathrm{vec}(\mathbf{H}) \in \mathbb{C}^{N_r N_t K \times 1}$**: The **unknown true near-field channel vector** that must be estimated. It is the vectorized form of the channel tensor across $N_r = 32$ receive antennas, $N_t = 256$ transmit antennas, and $K = 8$ OFDM subcarriers ($32 \times 256 \times 8 = 65,536$ complex channel coefficients).
* **$\mathbf{\Phi} \in \mathbb{C}^{M_r N_p \times N_r N_t K}$**: The **equivalent measurement sensing operator**. It mathematically models the combined physical effects of:
  1. Transmit analog pilot beamforming precoders ($\mathbf{F}_{\mathrm{RF}}$).
  2. Receive analog combining matrices ($\mathbf{W}_{\mathrm{RF}}$).
  3. Low-resolution 1-bit RF phase shifters ($\pm 1$ phase quantization) in the hybrid Array-of-Subarrays (AoSA) architecture.
* **$\tilde{\mathbf{n}} \sim \mathcal{CN}(\mathbf{0}, \sigma_n^2 \mathbf{I}_{M_r N_p})$**: The **Additive White Gaussian Noise (AWGN)** vector introduced by thermal noise and analog front-end components at the receiver RF chains, with noise variance $\sigma_n^2$ determined by the operating SNR.
* **$z \in \mathbb{R}^{d_z}$ ($d_z = 100$)**: The **low-dimensional latent code vector**. It serves as the compact semantic coordinates identifying the near-field channel on the generator's learned manifold.
* **$z^{(0)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_z})$**: The **initial latent vector**, initialized randomly at iteration $t = 0$ from a standard multivariate Gaussian distribution.
* **$\theta^{\ast}$**: The pre-trained, meta-learned generator network parameters, which remain **frozen** (not updated) during online inference. Gradients are still back-propagated through the network layers to compute $\nabla_z$.
* **$G_{\theta^{\ast}}(z)$**: The **synthesized candidate channel** generated by passing the latent vector $z$ through the frozen neural network $G_{\theta^{\ast}}$.
* **$\mathbf{\Phi} G_{\theta^{\ast}}(z)$**: The **simulated pilot observation** that the receiver *would* observe if the true physical channel were exactly $G_{\theta^{\ast}}(z)$.
* **$\|\tilde{\mathbf{y}} - \mathbf{\Phi} G_{\theta^{\ast}}(z)\|_2^2$**: The **measurement residual (fitting error)** using the squared Euclidean $\ell_2$-norm (where $\|\mathbf{v}\|_2^2 = \sum_j |v_j|^2$). It measures the discrepancy between the synthesized pilot response and the actual physical pilots $\tilde{\mathbf{y}}$ observed over the air.
* **$\|z\|_2 = 1$**: The latent vector is kept on the unit sphere by normalizing it after each update (there is no separate regularization term in the code).
* **$\arg\min_z$**: The minimization operator seeking the latent code $z^{\ast}$ that minimizes the measurement residual.
* **$\alpha_{\mathrm{online}} = 0.1$**: The **online step size (learning rate)** used to update the latent vector.
* **$\nabla_z \mathcal{L}(z^{(t)})$**: The **gradient vector** of the loss taken *exclusively* with respect to the $100$ variables in $z^{(t)}$. Computing $\nabla_z$ requires a backward pass through the generator (the weights are not updated).
* **Denominator $\|\cdot\|_2$**: Normalizes the updated vector to lie on the **unit hypersphere** $\mathbb{S}^{d_z - 1}$. This projection step prevents gradient magnitude explosion and restricts the search trajectory to the high-density sphere of the prior.
* **$\hat{\mathbf{H}} \in \mathbb{C}^{N_r \times N_t \times K}$**: The **final estimated channel tensor** across all $N_r = 32$ receive antennas, $N_t = 256$ transmit antennas, and $K = 8$ OFDM subcarriers ($65,536$ complex channel gains).

---

### Bachelor's Thesis Chapter 4 Python Evaluation Scripts

All Chapter 4 experimental figures can be directly evaluated using the dedicated scripts in `python_dcs/evaluation/`:

#### 1. Figure 4.2: NMSE vs. Transmit SNR ($-20\text{ dB}$ to $20\text{ dB}$)
Tests channel estimation accuracy under varying noise levels with fixed $N_p = 100$ pilots:
```bash
python -m python_dcs.evaluation.eval_fig4_2_nmse_vs_snr
```

#### 2. Figure 4.3: Effective Achievable Rate (EAR) vs. Pilot Measurements
Evaluates spectral efficiency accounting for pilot training overhead under infinite coherence vs. fast fading ($T_{\mathrm{coh}} = 512$ symbols):
```bash
python -m python_dcs.evaluation.eval_fig4_3_ear_vs_meas
python -m python_dcs.evaluation.plot_dcsEAR
```

#### 3. Figure 4.4: NMSE vs. Spatial Compression Ratio $\rho = N_p / N_t$
Sweeps pilot compression ratio $\rho$ from $0.1$ to $1.0$ at fixed $\text{SNR} = 10\text{ dB}$:
```bash
python -m python_dcs.evaluation.eval_fig4_4_nmse_vs_rho
```

#### 4. Figure 4.5: Bit Error Rate (BER) vs. SNR & Spatial Multiplexing Limits
DCS BER evaluation. **Note:** this script is a later exploratory variant (QPSK, $N_s = 2$, stage-3 fine-tuned checkpoint, 10 restarts, 100 to 600 gradient steps). The Deep CS curves in thesis Figure 4.5 were produced by `thesis_figure_scripts/python/NMSEandBER.py` (16-QAM, e300 checkpoint, 20 steps, 1 restart). See [Section 8](#8-corrections-to-the-thesis-report).
```bash
python -m python_dcs.evaluation.eval_fig4_5_ber_vs_snr
```

#### 5. Exploratory DCS Distance Sweep ($0.4\text{ m}$ to $1.6\text{ m}$)
Evaluates DCS across distances. This is **not** part of thesis Figure 4.6, which shows only the conventional baselines (`simulations/NMSE_EAR_Vs_Distance.m`):
```bash
python -m python_dcs.evaluation.eval_fig4_6_distance_sweep
```

#### 6. Figure 4.7: Online Runtime
Measures the DCS runtime per channel estimate (batch of 100 channels, 20 gradient steps, wall-clock time divided by 100):
```bash
python -m python_dcs.evaluation.eval_fig4_7_runtime
```

> **Note**: Result CSV files are stored in `results/csv/` and `python_dcs/data/`. Some of them come from earlier runs; Section 8 says which results the thesis figures use.

---

## 5. MATLAB Conventional Compressive Sensing Pipeline

The conventional Compressive Sensing framework uses the **TeraMIMO near-field channel simulator** with spherical wavefront steering vectors, molecular absorption physics, and sparse recovery algorithms.

---

### Step 1: Environment & Path Setup

Every time you open MATLAB, navigate to the repository root directory and initialize all module paths:
```matlab
setup_paths;
```
*(All simulation and plotting scripts also call `setup_paths.m` automatically on launch).*

---

### Step 2: Physics-Based TeraMIMO Channel Engine

The near-field wideband channel matrix at subcarrier $k$ is synthesized using exact spherical wave geometry:

$$
\mathbf{H}_k = \sum_{\ell=1}^L \alpha_{\ell,k} \mathbf{a}_r(\theta_r^\ell, \phi_r^\ell, r_r^\ell, f_k) \mathbf{a}_t^H(\theta_t^\ell, \phi_t^\ell, r_t^\ell, f_k)
$$

* **Spherical Wavefront Array Vectors**: Implemented in `channel_simulator/TeraMIMO_Channel/Ch_Utilities/get_ArrayResponse.m` using exact phase delays:
  

$$
\phi_{n} = \frac{2\pi f_k}{c} \left( \sqrt{r^2 + \delta_n^2 - 2 r \delta_n \sin\theta} - r \right)
$$

* **Atmospheric Molecular Absorption**: Implemented in `channel_simulator/Molecular_Absorption/` using actual spectroscopic HITRAN data for water vapor ($	ext{H}_2	ext{O}$), carbon dioxide ($	ext{CO}_2$), and oxygen ($	ext{O}_2$) at $300\text{ GHz}$.

---

### Step 3: Conventional Sparse Recovery Algorithms

Located in `Algorithms/`:
These baseline implementations are by Simon Tarboush (Tarboush, Ali and Al-Naffouri, ICASSP 2023); we adapted them to our shared simulation setup.
* **`OMP_SMC.m`**: Orthogonal Matching Pursuit solved separately for each subcarrier (SMC = separate multi-carriers).
* **`SOMP.m`**: Simultaneous OMP exploiting common spatial sparsity across all $K=8$ wideband subcarriers.
* **`OMP_SMC_DR.m`**: Per-subcarrier OMP with Dictionary Reduction (DR).
* **`get_proposed_DR.m`**: Builds localized sub-dictionaries from the support estimated on a neighbouring subarray.
* **`get_proposed_HSPM_estimation.m`**: Runs the estimation over every Tx/Rx subarray pair under the hybrid spherical- and planar-wave model (HSPM): planar wavefront within a subarray, spherical across subarrays.

---

### Step 4: Running MATLAB Simulations

Run any of the comprehensive simulation scripts located in `simulations/`:

```matlab
% 1. Figure 4.2: NMSE vs Transmit SNR (-20 dB to 20 dB, Step 5 dB)
run('simulations/NMSEvsSNR.m');

% 2. Figure 4.3: Effective Achievable Rate vs Number of Pilot Beams
run('simulations/EARvsNumberofMeasurements.m');

% 3. Figure 4.4: NMSE vs Spatial Compression Ratio rho (Np / Nt)
run('simulations/NMSEvsRho.m');

% 4. Earlier BER simulation (thesis Figure 4.5 was made by
%    thesis_figure_scripts/matlab/Untitled4.m and NS2.m)
run('simulations/BERvsSNR_Cal.m');

% 5. QPSK variant of the BER simulation (not used in the thesis figures)
run('simulations/BERvsSNR_QPSK.m');

% 6. Figure 4.6: NMSE and achievable rate vs distance (d = 0.4 m to 1.6 m, fixed 10 dB SNR)
run('simulations/NMSE_EAR_Vs_Distance.m');

% 7. Table 4.3: Analytical Computational Complexity (FLOPs)
run('simulations/Complexity.m');

% 8. Figure 4.7: Online Execution Time & Latency Profiling
run('simulations/Runtime.m');
```

---

### Step 5: Generating Publication-Quality Figures

The plotting scripts in `plotting/` merge conventional MATLAB simulation results with DCS deep learning benchmarks to generate camera-ready Bachelor's thesis figures:

```matlab
% Generate Figure 4.2 (NMSE vs. SNR Comparison):
run('plotting/plot_NMSE_Results.m');

% Generate Figure 4.3 (Spectral Efficiency / EAR vs. Pilot Beams):
run('plotting/plot_EAR_Results.m');

% Generate Figure 4.4 (NMSE vs. Compression Ratio rho at SNR = 10 dB):
run('plotting/plot_NMSE_vs_Rho.m');

% Generate Figure 4.6 (NMSE vs. Near-Field Distance):
run('plotting/plot_Distance_Sweep_Results.m');
```

Figures are automatically displayed and saved in publication formats (`.png`, `.pdf`, `.fig`) inside `results/figures/`.

---

## 6. Comparative Evaluation & Key Metrics

### 6.1 Evaluation Metrics Defined & Explained

#### 1. Normalized Mean Squared Error (NMSE)
Measures the overall channel estimation error normalized by the true channel power, expressed in decibels (dB):

$$
\mathrm{NMSE} = 10 \log_{10} \left( \frac{\sum_{k=1}^K \|\mathbf{H}_k - \hat{\mathbf{H}}_k\|_F^2}{\sum_{k=1}^K \|\mathbf{H}_k\|_F^2} \right) \quad [\text{dB}]
$$

* **What each symbol means**:
  * $\mathbf{H}_k \in \mathbb{C}^{N_r \times N_t}$: The **true channel matrix** at subcarrier $k$ ($32$ receive antennas $\times 256$ transmit antennas).
  * $\hat{\mathbf{H}}_k \in \mathbb{C}^{N_r \times N_t}$: The **estimated channel matrix** reconstructed by the algorithm.
  * $\|\cdot\|_F$: The **Frobenius norm** ($\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} |a_{i,j}|^2}$), representing the total signal energy across all antenna pairs.
  * $K = 8$: Total number of wideband **OFDM subcarriers**.
  * **Numerator** $\sum_{k=1}^K \|\mathbf{H}_k - \hat{\mathbf{H}}_k\|_F^2$: The **total squared error energy** across all subcarriers.
  * **Denominator** $\sum_{k=1}^K \|\mathbf{H}_k\|_F^2$: The **total energy of the true channel**, normalizing the metric so it is independent of path loss or scale.
  * $10 \log_{10}(\cdot)$: Converts the normalized ratio to **decibels (dB)**.
* **How to interpret the values**:
  * **$0\text{ dB}$ ($1.0$)**: $100\%$ error (estimation failed completely).
  * **$-10\text{ dB}$ ($0.10$)**: $10\%$ residual error power.
  * **$-15\text{ dB}$ ($0.032$)**: $\approx 3.2\%$ residual error -- the target precision required for reliable near-field beam focusing.
  * **$-20\text{ dB}$ ($0.01$)**: $1\%$ residual error (near-perfect CSI).
  * *Lower (more negative) is better.*

---

#### 2. Effective Achievable Rate (EAR)
Measures the true data throughput per unit bandwidth, taking into account the transmission time lost to pilot training:

$$
\mathrm{EAR} = \left( 1 - \frac{N_p}{T_{\mathrm{coh}}} \right) R \quad [\text{bps/Hz}]
$$

* **What each symbol means**:
  * $R$: The instantaneous **spectral efficiency (capacity)** computed using the estimated channel beamformers.
  * $N_p$: Number of **pilot symbols** transmitted during channel training.
  * $T_{\mathrm{coh}} = 512$: Channel **coherence block length** in symbols (time window before the channel fades).
  * $\left( 1 - \frac{N_p}{T_{\mathrm{coh}}} \right)$: The **pre-log pilot penalty factor** -- represents the fraction of the frame left for actual payload user data.
* **Why it matters**: In fast-fading THz channels ($T_{\mathrm{coh}} = 512$), conventional CS needs $\approx 80$ pilots, losing $15\%$ of the frame. Deep CS achieves peak rate with only **$16$ pilots** ($3.1\%$ overhead), leaving **$96.9\%$** for data transmission.

---

#### 3. Spatial Compression Ratio ($\rho$)
Measures the degree of sub-Nyquist pilot compression relative to the transmit antenna dimension:

$$
\rho = \frac{N_p}{N_t} = \frac{N_p}{256}
$$

* Traditional CS breaks down when $\rho < 0.35$. Deep CS maintains accurate estimation down to $\rho = 0.15$.

---

#### 4. Bit Error Rate (BER) & Spatial Multiplexing Limits
Measures the fraction of misdetected bits during end-to-end data communication evaluated under single-stream ($N_s = 1$) and multi-stream ($N_s = 2$) configurations:

$$
\mathrm{BER} = \frac{\text{Number of Bit Errors}}{\text{Total Transmitted Bits}}
$$

* **Physical Insight**: Tests the interaction between channel estimation accuracy and transceiver hardware constraints. In sub-connected Array-of-Subarrays (AoSA) architectures, each RF chain connects to a localized subset of antennas, imposing a block-diagonal constraint on analog beamforming. While single-stream transmission ($N_s = 1$) achieves full array beamforming gain, multi-stream transmission ($N_s = 2$) faces inter-stream interference due to the low rank of the THz channel (closely spaced subarrays see highly correlated channels) and the restricted analog beamforming. All methods degrade in this case, including perfect CSI.

---

#### 5. Near-Field Distance Dependence ($d$)
Evaluates NMSE and achievable rate of the conventional baselines across $d \in [0.4\text{ m}, 1.6\text{ m}]$. The received SNR is held fixed at $10\text{ dB}$: the transmit power is raised at each distance to cancel the path loss. Changes with distance therefore come from the channel geometry, not from SNR.

---

#### 6. Computational Complexity & Runtime
* **FLOPs**: Analytical operation counts per channel estimate (Table 4.3). The DCS count covers only the measurement-operator products and leaves out the generator's forward and backward passes, so it underestimates the DCS cost.
* **Runtime**: Measured wall-clock time per estimate in our simulation code (MATLAB for the baselines, PyTorch for DCS). Useful for relative comparison only.

---

### 6.2 Summary of Results

All values are read from the figures and tables of the report (approximate where read from a plot). This table replaces an earlier version that contained numbers not supported by the results.

| Metric (source) | OMP | SOMP | OMP-DR | SOMP-DR | Deep CS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| NMSE at 0 dB SNR (Fig. 4.2) | $\approx -3.3$ dB | $\approx -4.8$ dB | $\approx -4.3$ dB | $\approx -5.9$ dB | $\approx -14.3$ dB |
| NMSE at 20 dB SNR (Fig. 4.2) | $\approx -6.2$ dB | $\approx -6.2$ dB | $\approx -7.8$ dB | $\approx -7.8$ dB | $\approx -14.5$ dB |
| Peak EAR, $T_{\mathrm{coh}} = 512$ (Fig. 4.3b) | $\approx 12.2$ b/s/Hz at 80 meas. | $\approx 12.6$ at 80 | $\approx 14.4$ at 80 | $\approx 14.9$ at 80 | $\approx 18.4$ at the lowest tested ($\approx 16$) |
| NMSE at $\rho = 1$ (Fig. 4.4) | $\approx -5.7$ dB | $\approx -6.1$ dB | $\approx -7.4$ dB | $\approx -8.0$ dB | $\approx -15$ dB (flat from $\rho \approx 0.1$) |
| BER, $N_s = 1$, 20 dB (Fig. 4.5a) | $1.4\times10^{-2}$ | $1.2\times10^{-2}$ | $1.5\times10^{-2}$ | $1.3\times10^{-2}$ | no errors observed from 10 dB |
| BER, $N_s = 2$, 30 dB (Fig. 4.5b) | $0.11$ | $0.10$ | $0.08$ | $0.07$ | $1.9\times10^{-3}$ (perfect CSI: $1.0\times10^{-3}$) |
| Analytical FLOPs (Table 4.3) | $3.36\times10^8$ | $3.36\times10^8$ | $3.26\times10^8$ | $3.14\times10^8$ | $6.55\times10^7$ (excludes generator passes) |
| Measured runtime per estimate (Fig. 4.7b) | $6.45$ s | $0.77$ s | $7.53$ s | $0.75$ s | $0.36$ s |

---

### 6.3 Key Bachelor's Thesis Findings
1. **Grid mismatch vs. learned prior**: The conventional methods reach an error floor around $-6$ dB (OMP, SOMP) and $-8$ dB (DR variants) because the true near-field paths fall between dictionary grid points. Deep CS reaches about $-14.5$ dB. It also saturates (flat from about $-5$ dB SNR), but at a much lower level, set by the learned prior rather than by a grid.
2. **Lower pilot overhead**: With a coherence block of $512$ symbols, Deep CS reaches its peak effective rate (about $18.4$ b/s/Hz) with about $16$ measurements, while the best baseline (SOMP-DR) peaks at about $14.9$ b/s/Hz with $80$ measurements.
3. **Spatial multiplexing limit**: With one stream, Deep CS performs close to perfect CSI. With two streams, every method degrades, including perfect CSI: Deep CS reaches about $1.9\times10^{-3}$ only at $30$ dB, while the baselines stay near $10^{-1}$. In further tests with up to four streams, the BER approached $0.5$ for all methods. The cause is the low rank of the channel (closely spaced subarrays see highly correlated channels) combined with the block-diagonal AoSA constraint, not the estimation algorithms.
4. **Runtime**: In our simulation code, Deep CS took about $0.36$ s per estimate, about $2\times$ faster than SOMP and SOMP-DR and about $18\times$ faster than OMP. OMP is slow mainly because it solves each of the $K = 8$ subcarriers separately.

---

## 7. Citation & Academic Reference

If you use this simulation codebase, deep learning models, or parts of this framework in your research, please cite:

```bibtex
@bachelorsthesis{EnbayaAlGndi2026NearField,
  author       = {Osama Younes Enbaya and Zaher Samir AlGndi},
  title        = {Near-field Channel Estimation in THz UM-MIMO Using Deep Compressed Sensing},
  school       = {University of Tripoli, Faculty of Engineering, Department of Electrical and Electronic Engineering},
  year         = {2026},
  type         = {Graduation Project Bachelor's Thesis (EE599)},
  address      = {Tripoli, Libya},
  note         = {Supervised by Eng. Aya H.A. Alsmoee}
}
```

### Acknowledgements
We express our deepest gratitude to our supervisor, **Eng. Aya H.A. Alsmoee**, for her guidance, continuous encouragement, and technical insights throughout this project. We also thank the **Department of Electrical and Electronic Engineering** at the **University of Tripoli**.

---

## 8. Corrections to the Thesis Report

Some of the points below were raised by a reader of the report; the rest were found while re-checking the report against the scripts and result files (October 2026). The report PDF in `documentation/` is kept as submitted. Page numbers refer to the printed page numbers of the report.

**1. BER text (p. 59) vs. Figure 4.5(b).**
The report says that with two streams Deep CS reaches a BER of about $10^{-3}$ near $15$ dB and that the baselines struggle to go below $10^{-2}$. The plotted data shows Deep CS at about $1.4\times10^{-1}$ at $15$ dB and $1.9\times10^{-3}$ at $30$ dB, while the baselines stay between $0.07$ and $0.11$ at $30$ dB. The text was written from an earlier run (`results/csv/DCS_BER_vs_SNRNS2.csv`, which gives $4.7\times10^{-3}$ at $15$ dB) and was not updated when the figure was regenerated. In Figure 4.5(a), points drawn at $10^{-6}$ are runs with zero bit errors, plotted at a floor value.

**2. Distance explanation (p. 61) vs. Figure 4.6(a).**
The report says NMSE worsens with distance because the received SNR drops. In `NMSE_EAR_Vs_Distance.m` the received SNR is held fixed at $10$ dB (the transmit power compensates the path loss), and Figure 4.6(a) shows NMSE *improving* with distance. The likely reason is that the baselines use the hybrid spherical/planar-wave model, which treats the wavefront as planar within each subarray. That approximation is poor at $0.4$ m and becomes more accurate toward the $\approx 2$ m Rayleigh distance of a $64$-element subarray. The achievable rate in Figure 4.6(b) still falls with distance, even with perfect CSI; since SNR is fixed, this is not path loss. A likely cause is that the channel loses spatial rank as the receiver moves away; this has not yet been verified.

**3. DR discussion (p. 63) vs. Figure 4.6.**
The report says OMP/SOMP overlap with their DR versions and that DR gives exactly the same accuracy. In Figure 4.6(b) the overlapping pairs are OMP with SOMP, and OMP-DR with SOMP-DR. In Figure 4.6(a) the DR versions are $1$ to $2.5$ dB better. DR therefore does change the accuracy in this setup.

**4. Runtime units and procedure (Figure 4.7(b)).**
This README previously stated $0.36$ ms on GPU. The measured value is **$0.364$ s per estimate**: the mean of the per-sample runtimes at $9$ SNR points ($0.354$ to $0.412$ s; the first point includes warm-up). `eval_fig4_7_runtime.py` processes a batch of $100$ test channels with $20$ Adam steps and $1$ restart, measures wall-clock time with `time.perf_counter()` (including the measurement-batch construction), and divides by $100$. The baseline runtimes were measured in MATLAB by `thesis_figure_scripts/matlab/Untitled3.m`: each method was timed over one full channel realization (all subarray pairs, all $8$ subcarriers, $512$ measurements per subarray pair), for one realization at each of $11$ SNR points from $-20$ to $30$ dB, and the results were averaged and entered into `Runtime.m`. The timer lines were inside an older version of `get_proposed_HSPM_estimation.m` that was later overwritten, so that exact timing code is no longer available. Hardware: the DCS model was trained on an Intel Core i9 with an NVIDIA RTX 5070 (12 GB); all MATLAB simulations, DCS inference and both runtime measurements ran on a second computer with an NVIDIA RTX 3050 (6 GB) and **[TO FILL: CPU model]**. The MATLAB baselines run on the CPU; the DCS scripts use the GPU when available. The other runtime figures previously in this README ($2.10$, $7.50$, $1.15$, $4.20$ ms and "$20.8\times$ faster") were not measured and have been removed. `python_dcs/evaluation/eval_runtime.py` previously contained placeholder numbers; it now prints the actual values.

**5. Complexity (Table 4.3, Figure 4.7(a)).**
Figure 4.7(a) shows $5.01\times$ and $5.07\times$ for OMP-DR and SOMP-DR, while Table 4.3 gives $4.98\times$ and $4.80\times$; they use average sub-dictionary sizes from different runs. The measured reduction is small (between about $1\%$ and $6\%$ of the $8192$ atoms, depending on the run), which is why DR barely changes the runtime. The DCS count ($T N_s N_p N_t N_r$) includes only the measurement-operator products and leaves out the generator's forward and backward passes, so it underestimates the DCS cost. The baseline counts also use $N_p = 512$ measurements per subarray pair, while the DCS count uses $N_p = 100$. Finally, OMP is about $8.4\times$ slower than SOMP despite nearly equal FLOP counts because it solves each of the $K = 8$ subcarriers separately.

**6. Deep CS error floor (p. 55).**
The report says Deep CS does not suffer the early saturation of the baselines. In Figure 4.2 it levels off at about $-14$ to $-14.5$ dB from $-5$ dB SNR onward, so it also has an error floor, set by the learned prior rather than by a grid.

**7. BER script in this repository.**
`python_dcs/evaluation/eval_fig4_5_ber_vs_snr.py` is a later exploratory version (QPSK, $N_s = 2$, stage-3 fine-tuned checkpoint, $10$ restarts, $100$ to $600$ gradient steps). It does not match the 16-QAM setting in Section 4.4.4 or the $20$-step setting used for the runtime and complexity results. The Deep CS curves in Figure 4.5 were produced by `thesis_figure_scripts/python/NMSEandBER.py` (16-QAM, e300 MAML checkpoint, $20$ Adam steps, $1$ restart, $100$ test channels); the saved version is the two-stream one, and the one-stream version was not kept. The baseline curves were produced by `thesis_figure_scripts/matlab/Untitled4.m` (one stream, $100$ realizations) and `NS2.m` (two streams; saved with $10$ realizations). Several other scripts that produced thesis figures were also missing from this repository; all of them are now in `thesis_figure_scripts/`, unchanged, with an index.

**8. Naming.**
"Cal" / "Calibrated" in file and variable names means the BER is computed with an MMSE equalizer built from a pilot-based estimate of the effective (beamformed) channel. It does not mean the data was adjusted.

**9. BER equalizers are not identical across methods.**
In `Untitled4.m` and `NS2.m` the baselines' MMSE equalizer uses twice the noise variance (`lambda_fac = 2`), while perfect CSI uses the normal value, and `NMSEandBER.py` uses the normal value for Deep CS. In `Untitled4.m` the baselines also estimate their effective channel from 8 noisy pilots, while `NMSEandBER.py` uses the exact effective channel. This favors Deep CS in Figure 4.5; both BER figures should be re-run with the same equalizer and channel knowledge for every method.

**10. Channel data.**
The MATLAB baseline scripts generate their own channel realizations with `TeraMIMO_Channel/ChannelGeneration/channel_TIV.m` (the saved `generate_channel_param_TIV.m` is set to `channelType = 'LoS'`). Deep CS was trained and tested on a pre-generated TeraMIMO dataset with multipath and line-of-sight components; `data/mat2pt.py` points to a file generated at $1.2$ m, while the BER scripts use $1$ m. The two methods therefore share the simulator family and the main system parameters, but not the same channel realizations or channel model.
