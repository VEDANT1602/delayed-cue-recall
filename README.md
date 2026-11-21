# Auxiliary texture prediction can delay early cue recall in a supervised recurrent network

Vedant Thaker · November 21, 2026

The entrance cue in a symbolic corridor determines a later choice. Its intervening textures are predictable and independent of the cue.

- [Revised manuscript (PDF)](manuscript.pdf) · [Editable manuscript](manuscript.md)
- [Revision code, results and reproduction instructions](revision/supplement/) · [Preparation record](revision/PREPARATION.md)
- [Baseline software and results at the study commit](https://github.com/VEDANT1602/delayed-cue-recall/tree/686f4f7e50229f061b4efdc5cb5d3f4812a2fc48) · [Software license](LICENSE) · [Third-party terms](LICENSING.md)

## Earlier study and reproduction instructions

The following instructions and results describe the original study. The revised paper and its supplemental tests are linked above.

### Why 79% delayed recall was a misleading average

At 60 training updates, a recurrent controller learning to predict the next corridor texture recalled the entrance cue on 79.17% of trials with a 16-step delay. Looking at individual seeds changes that picture. Seven networks got all 1,024 choices right. Four got exactly half right. The twelfth got 513 right. [Perfect seed](results.csv?plain=1#L25) · [Chance-level seed](results.csv?plain=1#L205) · [513-correct seed](results.csv?plain=1#L685)

Every texture sequence appears twice, once with each cue label. Erasing the cue gives identical inputs for the pair and yields exactly 50% accuracy. This controls for the possibility that the network inferred the turn from later textures. Twelve seeds pair initial weights and training batches across choice-only training and three auxiliary objectives. Training delays are 2, 4 and 8 steps. [Pair construction](run.py#L29)

![Choice recall over checkpoints at a 16-step delay](figure.png)

The choice-only comparison at the same 60-update checkpoint scored 95.15%. The archived paired difference is 15.97 percentage points in its favor, calculated before rounding the group means. By 220 updates, all 48 models had perfect recall at every tested delay. The earlier checkpoint was examined after that ceiling appeared. A new set of seeds with a checkpoint chosen in advance would be needed to test the early difference independently. This experiment concerns supervised recall of one bit on symbolic paths. [Seed-level records](results.csv) · [Experiment and limits](https://github.com/VEDANT1602/delayed-cue-recall/blob/686f4f7e50229f061b4efdc5cb5d3f4812a2fc48/manuscript.md) · [Paper PDF](https://github.com/VEDANT1602/delayed-cue-recall/blob/686f4f7e50229f061b4efdc5cb5d3f4812a2fc48/manuscript.pdf)

## Reproduce

Run the numerical verifier first. It needs only the Python standard library.

```console
python verify.py
```

The CPU experiment was tested with Python 3.11, NumPy 1.26 and a compatible CPU build of PyTorch 2.9. From the repository root, install the research dependencies, regenerate the 720-row results file, and rerun the checks.

```console
python -m pip install -r requirements.txt
python run.py
python verify.py
```

Each row represents a seed, an objective, a test delay and a checkpoint. The verifier checks choice counts, cue-erasure counts, next-texture accuracy and the paired difference. To regenerate the figure after the experiment, run the plotting script with its additional dependencies.

```console
python -m pip install -r requirements-plot.txt
python make_figure.py
```

## Research context

The design isolates cue retention from environmental prediction in a much smaller task than full navigation. [Ororbia and Kelly on maze learning and predictive processing](https://arxiv.org/abs/2204.00619) · [Jaderberg et al. on auxiliary learning objectives](https://arxiv.org/abs/1611.05397)

Maintained by Vedant Thaker.
