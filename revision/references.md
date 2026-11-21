# RIT reference-to-claim ledger

Access checked 21 November 2026. The entries distinguish what each publication supports from evidence generated in this study.

| Ref. | Claim verified | Locator and access |
|---|---|---|
| 1 | Auxiliary tasks were investigated in deep reinforcement learning; this motivates, but does not validate, a supervised corridor. | Jaderberg et al., open author paper, Fig. 1b for A3C plus auxiliary pixel-control heads and Fig. 5 for comparison of self-supervised variants in a random maze. [Full manuscript](https://arxiv.org/pdf/1611.05397) |
| 2 | Depth and loop-closure auxiliaries were used in 3D navigation. | Mirowski et al., open author paper, Fig. 4 and accompanying loop-closure experiment, and Fig. 8 for the depth-prediction and loop-closure head architecture. [Full manuscript](https://arxiv.org/pdf/1611.03673) |
| 3 | The cited architecture connects predictive processing and associative memory to maze tasks, including memory. | Ororbia and Kelly, Springer abstract, AGI 2022, pp. 321–331, first online 2023; full author preprint abstract and §3. [Publisher](https://doi.org/10.1007/978-3-031-19907-3_31) · [Open version](https://arxiv.org/abs/2204.00619) |
| 4 | Auxiliary weighting can depend on primary–auxiliary gradient alignment. The manuscript implements a related sign gate at weight 0.5 with its own optimizer and head-update policy. | Du et al., arXiv:1812.02224v2, PDF p. 5, Algorithm 1; weighted variant in Appendix C, Algorithm 2. [Open PDF](https://arxiv.org/pdf/1812.02224) |
| 5 | Thirty volunteers, six activities, waist-worn sensors, 50-Hz sampling, 128-reading windows and participant-disjoint train/test partitions. | UCI dataset page, Dataset Information / Additional Information and Citation; independent archive dimensions and disjoint subject IDs checked in experiment.py. [Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) |

The old neural population-code citation was removed because it added no evidence about the present encoder. Claims about early learning, cosine distributions and gate performance come from the separate generated CSVs. The public confirmatory plan and baseline source are linked in Methods/Availability and the supplement.
