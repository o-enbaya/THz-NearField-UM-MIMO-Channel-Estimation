"""
Prints the complexity values of Table 4.3 and the measured runtimes of Figure 4.7(b).

The previous version of this file contained placeholder numbers that were never measured.
It was replaced in October 2026; see README, Section 8.

Usage:
    python -m python_dcs.evaluation.eval_runtime
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DCS_RUNTIME_CSV = os.path.join(HERE, "..", "data", "dcs_runtime_per_sample_per_snr.csv")

# ---------------------------------------------------------------------------
# Table 4.3: analytical operation counts per channel estimate
# ---------------------------------------------------------------------------
# Baselines (simulations/Complexity.m):
#   K = 8 subcarriers, L = 10 (sparsity level), Np = 64 * 8 = 512 measurements
#   per subarray pair, G = 8192 dictionary atoms.
#   OMP  = K * (L*Np*G + L^3)
#   SOMP = L*K*Np*G + K*Np*L^2 + L^3
#   The DR values use the average reduced dictionary size of the thesis run,
#   so they are copied from Table 4.3 rather than recomputed.
# DCS:
#   T * Ns * Np * Nt * Nr with T = 20, Ns = 4, Np = 100, Nt = 256, Nr = 32.
#   This counts only the measurement-operator products. It does NOT include
#   the generator's forward and backward passes, so it underestimates the cost.
K, L, NP_CLASSICAL, G = 8, 10, 512, 8192
T, NS, NP_DCS, NT, NR = 20, 4, 100, 256, 32

flops = {
    "OMP": K * (L * NP_CLASSICAL * G + L**3),
    "SOMP": L * K * NP_CLASSICAL * G + K * NP_CLASSICAL * L**2 + L**3,
    "OMP-DR": 326_466_573,   # Table 4.3
    "SOMP-DR": 314_387_432,  # Table 4.3
    "DCS": T * NS * NP_DCS * NT * NR,
}

# ---------------------------------------------------------------------------
# Figure 4.7(b): measured runtime per channel estimate, in seconds
# ---------------------------------------------------------------------------
# Baselines: measured in MATLAB and entered by hand in simulations/Runtime.m.
# DCS: measured by eval_fig4_7_runtime.py (batch of 100 channels, 20 Adam
# steps, wall-clock time / 100), averaged over 9 SNR points.
runtime_s = {
    "OMP": 6.450301,
    "SOMP": 0.766663,
    "OMP-DR": 7.527417,
    "SOMP-DR": 0.745208,
}
dcs_per_snr = np.loadtxt(DCS_RUNTIME_CSV, delimiter=",")
runtime_s["DCS"] = float(np.mean(dcs_per_snr))

if __name__ == "__main__":
    print("Table 4.3 and Figure 4.7(b)")
    print(f"{'Method':<8} {'FLOPs':>14} {'vs DCS':>8} {'Runtime [s]':>12} {'vs DCS':>8}")
    for name in ["OMP", "SOMP", "OMP-DR", "SOMP-DR", "DCS"]:
        print(
            f"{name:<8} {flops[name]:>14,d} {flops[name] / flops['DCS']:>7.3f}x "
            f"{runtime_s[name]:>12.3f} {runtime_s[name] / runtime_s['DCS']:>7.2f}x"
        )
    print()
    print("DCS runtime per SNR point [s]:", np.round(dcs_per_snr, 4))
    print("Note: the first SNR point includes warm-up.")
