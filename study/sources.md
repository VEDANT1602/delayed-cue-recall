# Claim-to-source review

Each source below was checked against the sentence for which it is used. The experimental numbers in the paper are computed from the accompanying records.

| Manuscript reference | Evidence inspected and limit of attribution |
| --- | --- |
| 1. Jaderberg et al. | [Full text](https://arxiv.org/pdf/1611.05397), §3 and Figure 1. The architecture and auxiliary-task examples establish the multitask precedent. The paper does not test the delayed recall target defined here. |
| 2. Du et al. | [Full text](https://arxiv.org/pdf/1812.02224), §2 and Algorithm 1. Shared gradients inform how to weight or gate an auxiliary update. The present paired timing contrast estimates an outcome, not that gradient opposition caused it. |
| 3. Arjovsky et al. | [PMLR paper](https://proceedings.mlr.press/v48/arjovsky16.html), §5.1, copying-memory problem. Its ten-character delayed copying benchmark establishes precedent for measuring retention after a delay. Our input schedule differs. |
| 4. Mattson et al. | [MLSys full paper](https://proceedings.mlsys.org/paper_files/paper/2020/file/411e39b117e885341f25efb8912945f7-Paper.pdf), discussion of time to train to target quality. Their systems metric counts elapsed time. This study counts optimizer updates. |
| 5. Royston & Parmar | [Full article](https://link.springer.com/article/10.1186/1471-2288-13-152), Methods, restricted mean survival time. The area under the survival curve to a chosen horizon supplies the definition used here. Seed resampling and the common administrative horizon are choices in this study. |
| 6. Cho et al. | [ACL full paper](https://aclanthology.org/D14-1179.pdf), §2.3, Eqs. 5–8. The paper defines a gated hidden unit with reset and update gates. PyTorch supplies the numerical cell used here. |
| 7. Greff et al. | [Full paper](https://arxiv.org/pdf/1503.04069), §II and Figure 1. Discussion of LSTM gates and variants establishes the architecture family. The implementation is PyTorch’s one-layer LSTM. |
| 8. Reyes-Ortiz et al. | [UCI release](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and the supplied README, label, subject and inertial signal files. Six labels, 30 participants, 50 Hz recording, overlapping 128-sample windows and official subject split are documented. The source archive digest and exact transforms appear in README.md. |
| 9. Dehghani et al. | [Full article](https://doi.org/10.3390/s19225026), §5 and §6. Experiments compare subject-dependent and subject-independent validation in segmented activity recordings. The citation supports grouping by person. Their measured performance changes are not transferred to our data. |

The prospective GRU time-to-criterion plan, the historical vanilla fixed-update plan and the unregistered local experiments are separate records. Timing, limitations and historical links appear in README.md. No conference or journal acceptance, editorial review, or independent domain review is claimed.
