# External registration attachment for the delayed-recall replication

The object of registration is the new symbolic, fixed-width GRU experiment on 32 seed pairs. The 16 previously analyzed pairs and the smartphone studies are prior evidence. This attachment contains a plan and runnable code. The external registry assigns the timestamp and accession.

## What to register

Use the title **Exact-update replication of a distractor-forecasting delay in a width-12 GRU**. Identify Vedant Thaker as investigator and attach this packet to the OSF registration with the full contents of `future-registration.md`. The hypothesis, primary contrast, units, censoring, sign rule, interval, matching, exclusions and known results are set out there. Use the registry's archived URL and timestamp to connect the plan to the later execution record. A draft or editable OSF project alone leaves this step open.

## Code and environment

`timecourse.py` defines the model and paired sampler. `fine_resolution.py` defines the unchanged single-seed training and exact-update observation function. `aggregate.py` defines the paired percentile bootstrap. `registered_replication.py` runs these operations on seeds 11900–11931 and validates all 12,800 observations before reporting the arm summaries and paired estimate. The only Python packages required for this experiment are the versions in `registry-requirements.txt`.

Install CPython 3.12.14 and create an isolated Linux CPU environment. With the environment active, install the pinned packages:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1+cpu
python -m pip install numpy==2.3.5
sha256sum -c registry-files.sha256
```

After the registration is archived, replace REGISTRY_URL below with its actual URL. The script records that URL, start and finish UTC times and package versions. It writes one CSV per pair. After an interruption, repeat the command in the same folder to regenerate an unfinished pair. A complete folder also supports summary-only reanalysis.

```sh
python registered_replication.py run --registration REGISTRY_URL external-results
python registered_replication.py summarize external-results
```

The expected files are `seed-11900.csv` through `seed-11931.csv`, `pairs.csv`, `summary.json` and `execution.json`. Each of the 64 training trajectories contributes one correct count per update. If reproducing the registered science under different package versions, report the deviation.
