# Original scripts behind the thesis figures

These are **unchanged copies** of the scripts that produced the figures and tables in Chapter 4 of the report, taken from the working folders used for the thesis (added October 2026). Some of them were not in this repository before, and some of the scripts elsewhere in the repository are later or earlier versions. When the two differ, these are the ones that match the report.

The MATLAB scripts expect the original folder layout (`Algorithms/`, `TeraMIMO_Channel/`, `Molecular_Absorption/` and `Utilities/` next to the script, run from that folder). The Python scripts expect `model.py`, `utils.py`, `./ckpt/` and `./data/` as in the original DCS repository. Datasets, checkpoints and `.mat` result files are not included.

| Report item | Baselines (MATLAB) | Deep CS (Python) | Plot script |
| :--- | :--- | :--- | :--- |
| Fig. 4.2 NMSE vs SNR | `matlab/NMSEvsSNR.m` | `python/Runtime.py` (20 Adam steps, 100 test channels) | `matlab/plotting/nmseplot_new.m` |
| Fig. 4.3 EAR vs measurements | `matlab/EARvsNumberofMeasurements.m` | `python/plot_dcsEAR.py` (SNR 24.8 dB, 50 steps) | `matlab/plotting/EARplot_new.m` |
| Fig. 4.4 NMSE vs compression ratio | `matlab/NMSEvsRho.m` | `python/nmseVSrho gemini.py` (SNR 10 dB, 50 steps, 3 restarts) | `matlab/plotting/plot_NMSE_vs_Rho.m` |
| Fig. 4.5(a) BER, 1 stream | `matlab/Untitled4.m` (16-QAM, 100 realizations) | `python/NMSEandBER.py` with `n_streams = 1` (16-QAM, e300, 20 steps) | `matlab/plotting/plotBER.m` |
| Fig. 4.5(b) BER, 2 streams | `matlab/NS2.m` (16-QAM; saved with 10 realizations) | `python/NMSEandBER.py` with `n_streams = 2` (16-QAM, e300 checkpoint, 20 steps, 1 restart) | `matlab/plotting/plotBERns2.m` |
| Fig. 4.6 NMSE and rate vs distance | `matlab/NMSE_EAR_Vs_Distance.m` (fixed 10 dB SNR) | not in this figure | `matlab/plotting/plotNMSE_EAR_dist.m` |
| Table 4.3, Fig. 4.7(a), Fig. 4.8 | `matlab/Untitled3.m` (reduced-dictionary sizes), then `matlab/Complexity.m` | formula only | `matlab/plotting/plot_BarCharts.m` |
| Fig. 4.7(b) runtime | `matlab/Untitled3.m` | `python/Runtime.py` | `matlab/plotting/Runtime.m` |

See [Section 8 of the main README](../README.md#8-corrections-to-the-thesis-report) for the known issues with these scripts (equalizer settings in the BER scripts, separate channel data for the two methods, and the overwritten timer code).
