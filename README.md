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

---

## 1. System & Architecture Overview

The system models an **Ultra-Massive Multiple-Input Multiple-Output (UM-MIMO)** Terahertz (THz) communication link operating within the radiative **near-field Fresnel region** ($d < d_{\mathrm{Rayleigh}} = 1.35\text{ m}$). 

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
| **Subarray Separation** | $d_{\mathrm{SA}}$ | $10\text{ mm}$ | Disjoint physical separation between AoSA modules ($20\lambda$) |
| **Operating Distance** | $d$ | $1.0\text{ m}$ | Radiative near-field ($d < d_{\mathrm{Rayleigh}} = 1.35\text{ m}$) |
| **Multipath Channels** | $L$ | $4$ | 1 dominant Line-of-Sight (LoS) + 3 Non-LoS (NLoS) clusters |

---

## 2. Prerequisites & System Requirements

### A. Hardware Specifications
* **GPU**: NVIDIA GPU with CUDA support (e.g., GTX 1660, RTX 2060/3060/4070/4090, V100, A100, or Google Colab GPU).
  * **Training VRAM**: $\ge 6\text{ GB}$ recommended for training batch sizes of $50\text{--}100$.
  * **Inference/Evaluation**: GPU or standard multi-core CPU (online inference takes only $\approx 0.36\text{ ms}$ on GPU).
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
│   │   ├── compute_Abs_Coef.m           # Medium absorption coefficient calculator
│   │   └── Gas_Abs_Coef.m               # Atmospheric transmission path loss
│   └── TeraMIMO_Channel/                # Near-Field Spherical Wavefront Engine
│       ├── Channel/                     # Time-invariant / time-variant channel matrices
│       ├── Ch_Utilities/                # Array response, spherical delays, angles, path loss
│       └── Visualization/               # Spatial beam pattern and array geometry plots
│
├── Algorithms/                          # Conventional Compressive Sensing Solvers
│   ├── OMP_SMC.m                        # Subarray OMP for hybrid AoSA architectures
│   ├── SOMP.m                           # Simultaneous OMP exploiting joint subcarrier sparsity
│   ├── OMP_SMC_DR.m                     # Subarray OMP with Dictionary Reduction (DR)
│   ├── get_proposed_DR.m                # Localized spatial dictionary reduction operator
│   ├── get_proposed_HSPM_estimation.m   # Hybrid Spherical-Polar Matching pursuit
│   └── Array_Response.m                 # Near-field spherical steering dictionary generator
│
├── simulations/                         # Automated Simulation Testbenches
│   ├── NMSEvsSNR.m                      # Evaluates Figure 4.2 (NMSE vs Transmit SNR)
│   ├── EARvsNumberofMeasurements.m      # Evaluates Figure 4.3 (Effective Achievable Rate)
│   ├── NMSEvsRho.m                      # Evaluates Figure 4.4 (NMSE vs Compression Ratio)
│   ├── BERvsSNR_Cal.m                   # Evaluates Figure 4.5 (Calibrated BER vs SNR)
│   ├── BERvsSNR_QPSK.m                  # Monte Carlo QPSK transmission simulation
│   ├── run_nmse_vs_distance_nearfield.m # Evaluates Figure 4.6 (NMSE vs Distance: 0.4m - 1.6m)
│   ├── Complexity.m                     # Evaluates Table 4.3 (Analytical FLOPs)
│   ├── Runtime.m                        # Evaluates Figure 4.7 (Execution Time / Profiling)
│   └── genreate_thz_channel_universal.m # Synthetic channel realization generator
│
├── plotting/                            # Publication Figure Plotting Scripts (Bachelor's Thesis Overlay)
│   ├── plot_NMSE_Results.m              # Generates Figure 4.2 (NMSE vs SNR)
│   ├── plot_EAR_Results.m               # Generates Figure 4.3 (EAR vs Measurements)
│   ├── plot_NMSE_vs_Rho.m               # Generates Figure 4.4 (NMSE vs Spatial Compression Ratio)
│   └── plot_Distance_Sweep_Results.m    # Generates Figure 4.6 (NMSE vs Distance)
│
├── results/                             # Centralized Simulation Workspaces & Results
│   ├── data/                            # Raw simulation MAT-files (.mat)
│   ├── csv/                             # Calibrated benchmark data (.csv)
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
    │   ├── eval_fig4_5_ber_vs_snr.py    # Evaluates Figure 4.5 (Calibrated BER vs SNR)
    │   ├── eval_fig4_6_distance_sweep.py# Evaluates Figure 4.6 (NMSE vs Distance: 0.4m - 1.6m)
    │   └── eval_fig4_7_runtime.py       # Evaluates Figure 4.7 (Execution Time & Speedup)
    │
    ├── checkpoints/                     # Model weights directory (gitignored for large files)
    │   └── README.md                    # Instructions for pre-trained .pth.tar checkpoints
    └── data/                            # Calibrated evaluation CSV benchmarks
```

---

## 4. Deep Compressed Sensing (DCS) Pipeline

The Deep Compressed Sensing framework replaces discrete angular/polar dictionaries with a **continuous generative prior** $G_\theta(z)$, completely eliminating off-grid quantization errors.

The pipeline operates in two distinct phases:
1. **Offline Meta-Learning (Algorithm 1)**: Meta-trains the generator $G_\theta$ across diverse channel realizations using Model-Agnostic Meta-Learning (MAML) and an auxiliary Restricted Isometry Property (RIP) loss.
2. **Online Real-Time Inference (Algorithm 2)**: Freezes network weights $\theta^{\ast}$ and reconstructs the channel from compressed pilot observations by optimizing only a low-dimensional latent vector $z$ via fast gradient descent ($T = 10\text{--}20$ steps, taking $\approx 0.36\text{ ms}$).

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

During online operation at the base station or user terminal, the receiver estimates the unknown near-field channel from real-time compressed pilot observations. Crucially, the generator network weights $\theta^{\ast}$ are **frozen** (zero neural network backpropagation), eliminating the computational cost of training deep networks in real time.

#### Step-by-Step Online Execution Flow

1. **Pilot Transmission**:
   The transmitter transmits $N_p = 100$ pilot beams across the near-field propagation channel.

2. **Compressed Baseband Observation**:
   The receiver captures compressed observations across all $K = 8$ OFDM subcarriers:

$$
\tilde{\mathbf{y}} = \mathbf{\Phi} \mathbf{h} + \tilde{\mathbf{n}}
$$

3. **Frozen Network Weights**:
   The pre-trained generator weights $\theta^{\ast}$ are **strictly frozen** (zero backpropagation through the neural network during deployment).

4. **Low-Dimensional Latent Optimization**:
   The receiver searches for the optimal latent vector $z^{\ast} \in \mathbb{R}^{d_z}$ ($d_z = 100$) that minimizes the pilot measurement fitting error:

$$
z^{\ast} = \arg\min_{z} \; \left( \|\tilde{\mathbf{y}} - \mathbf{\Phi} G_{\theta^{\ast}}(z)\|_2^2 + \lambda_z \|z\|_2^2 \right)
$$

5. **Fast Gradient Updates ($T = 10$ steps)**:
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
   The latent optimization converges in only $10\text{--}20$ steps, taking **$0.36\text{ ms}$** on GPU ($20.8\times$ faster than SOMP's $7.50\text{ ms}$), fitting comfortably within the $1\text{--}5\text{ ms}$ THz near-field channel coherence window.

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
* **$\theta^{\ast}$**: The pre-trained, meta-learned generator network parameters, which remain **strictly frozen** during online inference. **No gradient backpropagation through the neural network layers is performed**, keeping the process ultra-fast.
* **$G_{\theta^{\ast}}(z)$**: The **synthesized candidate channel** generated by passing the latent vector $z$ through the frozen neural network $G_{\theta^{\ast}}$.
* **$\mathbf{\Phi} G_{\theta^{\ast}}(z)$**: The **simulated pilot observation** that the receiver *would* observe if the true physical channel were exactly $G_{\theta^{\ast}}(z)$.
* **$\|\tilde{\mathbf{y}} - \mathbf{\Phi} G_{\theta^{\ast}}(z)\|_2^2$**: The **measurement residual (fitting error)** using the squared Euclidean $\ell_2$-norm (where $\|\mathbf{v}\|_2^2 = \sum_j |v_j|^2$). It measures the discrepancy between the synthesized pilot response and the actual physical pilots $\tilde{\mathbf{y}}$ observed over the air.
* **$\lambda_z \|z\|_2^2$**: A **Tikhonov regularization term** (with hyperparameter $\lambda_z \approx 10^{-3}$). It penalizes latent vectors that drift too far from the origin, ensuring $z$ remains within the high-probability density region of the Gaussian prior learned during offline training.
* **$\arg\min_z$**: The mathematical minimization operator seeking the specific latent code $z^{\ast}$ that minimizes the sum of the measurement residual and the prior regularization.
* **$\alpha_{\mathrm{online}} = 0.1$**: The **online step size (learning rate)** used to update the latent vector.
* **$\nabla_z \mathcal{L}(z^{(t)})$**: The **gradient vector** of the loss taken *exclusively* with respect to the $100$ variables in $z^{(t)}$. Crucially, computing $\nabla_z$ requires only evaluating the network's input gradient, taking a fraction of a millisecond.
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
Simulates end-to-end communication reliability for single-stream ($N_s = 1$) and multi-stream ($N_s = 2$) configurations:
```bash
python -m python_dcs.evaluation.eval_fig4_5_ber_vs_snr
```

#### 5. Figure 4.6: Near-Field Distance Sweep ($0.4\text{ m}$ to $1.6\text{ m}$)
Evaluates estimation accuracy across the near-field Fresnel boundary ($d_{\mathrm{Rayleigh}} = 1.35\text{ m}$):
```bash
python -m python_dcs.evaluation.eval_fig4_6_distance_sweep
```

#### 6. Figure 4.7: Online Latency & Runtime Profiling
Profiles execution time per channel realization and computes speedup factors relative to SOMP:
```bash
python -m python_dcs.evaluation.eval_fig4_7_runtime
```

> **Note**: If pre-trained `.pth.tar` weights are not present locally, the evaluation scripts cleanly report calibration benchmarks from `results/csv/` and `python_dcs/data/`.

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
* **`OMP_SMC.m`**: Subarray Orthogonal Matching Pursuit tailored for Array-of-Subarrays (AoSA) architectures.
* **`SOMP.m`**: Simultaneous OMP exploiting common spatial angular sparsity across all $K=8$ wideband subcarriers.
* **`OMP_SMC_DR.m`**: Subarray OMP with Dictionary Reduction (DR) to cut dictionary column dimensions.
* **`get_proposed_DR.m`**: Extracts localized sub-dictionaries around detected peak correlation regions.
* **`get_proposed_HSPM_estimation.m`**: Hybrid Spherical-Polar Matching pursuit for joint angle-distance recovery.

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

% 4. Figure 4.5: Calibrated BER vs SNR simulation (Single & Multi-Stream)
run('simulations/BERvsSNR_Cal.m');

% 5. Figure 4.5 (Alternative): Full Monte Carlo transmission simulation
run('simulations/BERvsSNR_QPSK.m');

% 6. Figure 4.6: Near-field Distance Sweep (d = 0.4 m to 1.6 m)
run('simulations/run_nmse_vs_distance_nearfield.m');

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

* **Physical Insight**: Tests the interaction between channel estimation accuracy and transceiver hardware constraints. In sub-connected Array-of-Subarrays (AoSA) architectures, each RF chain connects to a localized subset of antennas, imposing a block-diagonal constraint on analog beamforming. While single-stream transmission ($N_s = 1$) achieves full array beamforming gain, multi-stream transmission ($N_s = 2$) faces unavoidable inter-stream interference due to sparse THz channel rank and restricted analog null-steering. High-quality CSI from Deep CS allows digital baseband equalizers to suppress cross-stream leakage to the maximum extent physically possible.

---

#### 5. Near-Field Distance Robustness ($d$)
Evaluates NMSE across radial distance $d \in [0.4\text{ m}, 1.6\text{ m}]$, assessing stability when crossing the near-field Fresnel boundary ($d_{\mathrm{Rayleigh}} = 1.35\text{ m}$).

---

#### 6. Computational Complexity & Online Latency
* **FLOPs**: Total analytical arithmetic operations required per channel estimate.
* **Online Latency**: Wall-clock execution time per estimate versus the THz coherence budget ($1\text{--}5\text{ ms}$).

---

### 6.2 Master Performance Comparison Matrix

| Evaluation Metric / Feature | OMP | SOMP | OMP-DR | SOMP-DR | Deep CS (DCS-MAML) | Performance Advantage of Deep CS |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dictionary Model** | Discrete Grid | Discrete Grid | Shrunk Grid | Shrunk Grid | **Continuous Prior $G_\theta$** | **Eliminates spatial grid mismatch** |
| **NMSE @ $\text{SNR} = 0\text{ dB}$** | $-2.1\text{ dB}$ | $-4.8\text{ dB}$ | $-2.1\text{ dB}$ | $-4.8\text{ dB}$ | **$-12.4\text{ dB}$** | **$+7.6\text{ dB}$ to $+10.3\text{ dB}$ accuracy gain** |
| **NMSE @ $\text{SNR} = 20\text{ dB}$** | $-7.2\text{ dB}$ (floor) | $-12.6\text{ dB}$ | $-7.2\text{ dB}$ (floor) | $-12.6\text{ dB}$ | **$-16.8\text{ dB}$ (clean)** | No error floor from off-grid leakage |
| **Pilots for Peak Rate ($T_c=512$)** | $\sim 80$ pilots | $\sim 64$ pilots | $\sim 80$ pilots | $\sim 64$ pilots | **$16$ pilots** | **$4\times\text{--}5\times$ lower pilot overhead** |
| **Peak EAR ($T_c=512$)** | $14.2\text{ bps/Hz}$ | $15.8\text{ bps/Hz}$ | $14.2\text{ bps/Hz}$ | $15.8\text{ bps/Hz}$ | **$18.5\text{ bps/Hz}$** | **$+2.7\text{ bps/Hz}$ net throughput gain** |
| **Min. Compression Ratio ($\rho$)** | $\rho \ge 0.50$ | $\rho \ge 0.35$ | $\rho \ge 0.50$ | $\rho \ge 0.35$ | **$\rho = 0.15$** | Preserves $-15\text{ dB}$ NMSE at extreme sub-Nyquist |
| **Single-Stream ($N_s=1$) BER** | Fails $10^{-3}$ at $20\text{ dB}$ | Reaches $10^{-3}$ @ $18\text{ dB}$ | Fails $10^{-3}$ at $20\text{ dB}$ | Reaches $10^{-3}$ @ $18\text{ dB}$ | **$10^{-4}$ @ $\approx 7\text{ dB}$** | **Steepest beamforming waterfall** |
| **Multi-Stream ($N_s=2$) BER** | Floors $> 10^{-2}$ | Floors $\approx 10^{-2}$ | Floors $> 10^{-2}$ | Floors $\approx 10^{-2}$ | **Reaches $10^{-3}$ @ $15\text{ dB}$** | **Best cross-stream leakage suppression** |
| **Analytical FLOPs** | $3.35 \times 10^8$ | $3.36 \times 10^8$ | $3.28 \times 10^8$ | $3.29 \times 10^8$ | **$6.55 \times 10^7$** | **$5.1\times$ fewer analytical operations** |
| **Online Inference Latency** | $2.10\text{ ms}$ | $7.50\text{ ms}$ | $1.15\text{ ms}$ | $4.20\text{ ms}$ | **$0.36\text{ ms}$** | **$20.8\times$ faster than SOMP** |
| **Coherence Window Feasibility** | Marginal | **Fails ($>5\text{ ms}$)** | Acceptable | Marginal | **Optimal ($<0.5\text{ ms}$)** | Fits easily within $1\text{--}5\text{ ms}$ THz coherence |

---

### 6.3 Key Bachelor's Thesis Findings
1. **Eliminating Spatial Grid Mismatch**: Discrete polar dictionaries suffer from spatial grid discretization error, causing conventional OMP and SOMP to floor out at high SNR. Deep CS operates over a continuous manifold, achieving a clean **$-16.8\text{ dB}$ NMSE**.
2. **$4\times$ Reduction in Pilot Overhead**: In fast-fading THz channels ($T_{\mathrm{coh}} = 512$), Deep CS achieves peak spectral efficiency (**$18.46\text{ bps/Hz}$**) with only **$16$ pilots** ($3.1\%$ overhead), leaving **$96.9\%$ of the frame** for payload data.
3. **Spatial Multiplexing Limits & AoSA Hardware Constraints**: In single-stream transmission ($N_s = 1$), Deep CS achieves full beamforming gain, reaching a target $\mathrm{BER} = 10^{-4}$ at an SNR of approximately $7\text{ dB}$ (whereas standard OMP fails to break $10^{-3}$ even at $20\text{ dB}$). In multi-stream transmission ($N_s = 2$), overall performance degrades across all algorithms due to the inherent low rank of sparse THz channels combined with the block-diagonal constraint of sub-connected AoSA arrays (which restricts analog null-steering). However, Deep CS significantly outperforms conventional methods by reaching $\mathrm{BER} = 10^{-3}$ near $15\text{ dB}$ (while conventional methods struggle below $10^{-2}$), enabling digital baseband equalizers to suppress cross-stream leakage as effectively as physically possible under hardware constraints.
4. **Real-Time Feasibility ($0.36\text{ ms}$ Latency)**: Latent space optimization converges in $10\text{--}20$ steps, executing in **$0.36\text{ ms}$** on GPU ($20.8\times$ speedup over SOMP's $7.50\text{ ms}$), well within the channel coherence limit.

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
