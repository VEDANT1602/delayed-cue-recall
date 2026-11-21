# Reproducing the delayed-recall experiments

The primary contrast is the difference in restricted mean optimizer updates to sustained 90% terminal recall in 16 paired GRU runs. The public [prospective plan](https://github.com/VEDANT1602/delayed-cue-recall/blob/68aa06cc899fd30562a3dd6359e2a350f06bcb2d/analysis-plans/time-to-recall/confirmatory-time.md) is reproduced verbatim as `confirmatory-time.md` (SHA-256 `2a286e6ec261788cb52bfd601129f1789e51f7a349bee69dfa7e98f9ce29d69f`). Reserved training ran at 12:32:32–12:32:57 UTC on 21 November 2026 after the plan appeared. The executed core has SHA-256 `a06e13657ed09ccae83c3a0ea8d8cbd9aba2e2626929cfc63ada0d5cf0f6f8ff` and is preserved in the [public code record](https://github.com/VEDANT1602/delayed-cue-recall/blob/3aad46646d158ca44af5ae02e9350ebbe5cd2777/analysis-plans/time-to-recall/timecourse.py). The archive of the plan is a GitHub commit. An OSF registration has not been claimed.

## Install and run

Use a writable copy of this directory. Retraining regenerates the CSVs in place. The programs run on a CPU. The recorded environment used CPython 3.12.14, PyTorch 2.9.1+cpu, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, and Matplotlib 3.10.8. A participant-level cache is only needed for retraining the measured experiments.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python har_data.py har.zip har.npz
for cell in RNN GRU LSTM; do
  python timecourse.py corridor "$cell"
  python timecourse.py measured "$cell" --cache har.npz
done
python confirm_gru.py
for seed in $(seq 7600 7603); do python activity.py "$seed" --cache har.npz; done
for seed in $(seq 8800 8811); do python activity_memory.py "$seed" --cache har.npz; done
```

The included trajectory files also permit verification without retraining or downloading the participant data.

```bash
python aggregate.py
python measurement_summary.py
python plot_survival.py
python verify.py
```

The first program reconstructs 63 sensitivity and primary summaries, event flags, and survival curves from 13,440 observations. `measurement_summary.py` combines 24 measured-label activity fits, eight intact activity GRUs and one logistic baseline. `verify.py` independently integrates the survival proportion across the intervisit intervals. Freshly training both reserved arms for seed 7900 reproduced all 120 recorded evaluations. Use the CPU dependencies above when comparing exact counts.

## Read the outputs

| File pattern | Unit of observation |
| --- | --- |
| `curve_*.csv` | Counts correct for a seed, objective, test delay, and scheduled visit. Each denominator contains 256 paths paired across the two cues. |
| `events.csv`, `survival.csv`, `time_summary.csv` | Per-seed event or censoring, event-free curve, and paired restricted mean contrast, respectively. |
| `activity_memory_<seed>.csv` | 30 validation visits per objective. Only the 1,500-update row includes final test counts and the prefix-erasure check. |
| `measured_label_*.csv` | Event times, survival curves and seed-paired summaries from all 12 measured-label pairs. |
| `activity_curve_<seed>.csv`, `activity_test_<seed>.csv` | 35 validation epochs for each recurrent objective and final scores for the selected epoch. Subject −1 denotes the pooled nine-person test set. |
| `activity_summary.csv` | Means of the test metrics across seeds. A single logistic baseline is fitted. |

Training seeds are replications of the learning algorithm, conditional on the specified task and split. The same physical participant’s overlapping windows are correlated. Holding out entire participants prevents these windows from appearing in both fitting and evaluation. The summary intervals do not measure uncertainty from recruiting new participants.

## Parameter choices and sensitivity

- The symbolic cycle has four symbols and transition probability 0.8. It provides a forecastable distractor with no information about the independently assigned cue. The assumption tests interference in a controlled setting. Generalizing to other transition laws would require new tests.
- Training delays 2, 4 and 8 give short, variable retention periods. Testing at 8, 16 and 32 checks in-range and longer delays. Panels of 256 cue-balanced pairs make the erased-cue control exactly one half.
- Hidden width 12, learning rate 0.003, batch size 24 paired paths, Adam, norm clipping at one and auxiliary cross-entropy weight 0.5 hold the optimization setting fixed across paired arms. The older [18-cell width, rate and optimizer sweep](https://github.com/VEDANT1602/delayed-cue-recall/tree/c8a7031a587260b06416deb4af622c6fafcbd580/revision/supplement) includes every per-seed score and encoder-gradient cosine. At width 12 and Adam 0.003, the older fixed-update gate recovered 12.01 percentage points of recall relative to fixed weight. Negative gradient cosines also occurred near ceiling. That analysis used other seeds, an earlier implementation and a checkpoint outcome. It does not identify how a gate affects the new time-to-criterion endpoint.
- Two consecutive visits guard against scoring a solitary threshold excursion. Ninety percent is a declared task-quality target. The complete 80% and 95% sensitivities are in `time_summary.csv`. Ten-update observation spacing resolves the transition coarsely. The 200-update limit caps exposure. A positive interval for the 16 reserved GRU pairs is the sole confirmed directional test here. Time counts optimizer updates, with no claim of matched computation per update across objectives.
- The measured-label task uses six classes, a 1.28-second prefix and a 1.28-second suffix. Equal durations make the memory requirement explicit. Reusing each suffix for all six classes removes suffix-label shortcuts. The phase flag is supplied. It is an experimental construction from observed activity labels and signals.
- For the measured-label task 75% corresponds to substantially above six-class chance, two visits reduce transient crossings, 50 updates is the inspection period, and 1,500 is the common censoring horizon. The width of 48 gives a larger encoder for nine continuous axes. Weight 0.5 scales mean squared residuals on the standardized inputs. There is no scale equivalence between this loss and the symbolic cross-entropy. The 12 pairs vary the initialization and training draws while fixing the balanced validation and test panels.
- In the intact-window task a complete 35-epoch budget and validation-selected checkpoint prevent an undertrained transient from defining the comparison. The logistic features are mean, standard deviation, extrema, mean absolute first differences, and average magnitude in DFT bins 1–4 and 5–11 for each of nine standardized axes. Scikit-learn additionally standardizes these 63 features on the fitting subjects. All models see the official test set only for final scoring.
- Bootstrap intervals use 20,000 paired resamples (PCG64 seed 19373); all seed pairs, including jointly tied and individually censored runs, remain in the analysis. The local exploration record `design.md` preserves choices made before each local fitting block. Its opening chronology predates the later, separately posted GRU plan.

## Data and terms

The input is [Human Activity Recognition Using Smartphones](https://doi.org/10.24432/C54S4K), J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto and X. Parra (2013), released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The official UCI outer archive SHA-256 is `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`. `har_data.py` verifies the download, preserves activity labels and person IDs, and averages each nonoverlapping group of four raw readings. The supplied repository contains software and aggregate training records. Obtain the raw windows directly from UCI. The repository MIT software license applies to the author's software. Third-party data and dependencies retain their own terms.

## Checkpoint resolution

Run `python fine_resolution.py` to retrain the 16 reserved GRU seed pairs and measure the fixed delay-16 panel after each of 200 updates. Run `python fine_endpoints.py` to recover scored endpoints from the included counts without training. The script asserts agreement with all 640 archived visits. It writes rule-specific per-seed and summary files. Every-update records are retrospective analyses of known trajectories. All four event rules are specified in the paper; a passing score requires at least 461 of the 512 panel decisions to be correct. Time is the update confirming the run of qualifying assessments.

The auxiliary fit for seed 8807 in the measured-label experiment failed to cross by 1,500 and had exactly 128 correct of 768 erased and intact terminal test decisions. Its paired cue-only fit scored 703 of 768 at the same horizon. The source records are `activity_memory_8807.csv` and `measured_label_events.csv`. Its exclusion is shown only as a diagnostic.

A prospective external-registry draft for a fresh, unrun seed block is in [future-registration.md](future-registration.md). It does not retrospectively register the completed cadence comparison.
