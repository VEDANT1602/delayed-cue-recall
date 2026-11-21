# Early cue recall | computational supplement, 21 November 2026

## Chronology and scope

The 12-seed discovery and the previous 24-seed follow-up were already known. previous_revision/ preserves that follow-up, its own plan and environment files. The 18-cell corridor and measured activity studies were locally specified before computing their results at Git commit `f8c7467a014f4b876e974f4ff9688de7f4140ea6` on 2026-11-21 11:40:10 UTC. Its subsequent public record was posted during the extension and is not a prior public registration for those runs. [Public sweep record](https://github.com/VEDANT1602/delayed-cue-recall/blob/8fb36e6ef37ef6c5b5b0beb30ced5d04d5671d2f/revision/analysis-plan.md)

A distinct public commit specified the confirmatory 24-seed block **before that block was run**. confirm_execution.txt records execution from 11:43:29 through 11:43:39 UTC on 2026-11-21. The plan was not retroactive registration for the prior studies. [Frozen confirmatory plan](https://github.com/VEDANT1602/delayed-cue-recall/blob/ae1c428475388cee0045d20cdd02f8e175da67cb/revision/confirmatory-plan.md)

## Reproduce

The original-model file is a verbatim copy of run.py from [commit 686f4f7](https://github.com/VEDANT1602/delayed-cue-recall/tree/686f4f7e50229f061b4efdc5cb5d3f4812a2fc48) by Vedant Thaker. The archived Python 3.11 baseline and Python 3.12 revision environment locks are in previous_revision/. New analyses were run CPU-only with Python 3.12.14, NumPy 2.3.5 and PyTorch 2.9.1+cpu on Linux x86_64.

1. In a compatible Python 3.12 environment, install `python -m pip install -r previous_revision/requirements.txt`. Run `python experiment.py corridor` for 648 fits at 60 updates each.
2. Run `python experiment.py har`. The first run downloads the official HAR archive (~61 MB) to har.zip; the program verifies its SHA-256. To use a local copy, add `--har-archive PATH`. The raw archive is not included. This task trains 24 models.
3. Run `python confirm.py` for the 48 additional fixed-endpoint fits. Run `python summarize.py` to regenerate all three paired-seed summary tables. `python plot_grid.py` regenerates grid.png; matplotlib is its only extra dependency.
4. All summary intervals use the paired differences within seed, taking 20,000 percentile resamples at seed 42117. Repeated windows are not independent participant resamples. *_runs.csv contains each model's test counts. *_gradients.csv contains all 60 encoder-gradient cosine values per run. The summary file pairs each gate with both fixed and control modes. CPU or library changes can produce small floating-point differences; compare original choice/erasure counts as described in the previous revision.

## Input attribution

Human Activity Recognition Using Smartphones, J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto and X. Parra (2013), UCI Machine Learning Repository. The source defines six labeled activities and a subject-separated split. The extension selects total-acceleration channels, subsamples and standardizes them; it makes no changes to the downloaded archive. SHA-256: `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`. [Dataset](https://doi.org/10.24432/C54S4K) · [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)

No new license is asserted over third-party software. Archive software rights are separate from the UCI data license.
