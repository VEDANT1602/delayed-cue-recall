# Distractor forecasting delayed recall acquisition on two controlled tasks

Learning experiment by Vedant Thaker, Independent Researcher, Jeffersonville, Indiana, USA. Revised November 24, 2026.

A recurrent classifier must preserve a cue through a sequence of distractors. A second output can predict the next distractor well even though the distractor sequence carries no information about the cue. In a width-12 gated recurrent unit (GRU), adding a next-symbol loss with weight 0.5 increased the mean time to sustained 90% recall from 70.00 to 78.75 optimizer updates. The paired difference was 8.75 updates (95% seed-bootstrap interval 5.63–11.88) across the 16 seeds and ten-update assessment schedule specified in a GitHub commit. A later analysis of the same runs at every update gave 8.81 additional updates under an eleven-consecutive-pass rule. On a second task constructed from measured smartphone movements, forecasting delayed acquisition in all twelve pairs. One failed run accounted for 88% of the summed terminal accuracy gap on that task. For intact activity windows the two training objectives differed by 0.66 percentage points in mean macro-F1 across four seeds. The evidence for a timing cost concerns the two delayed decisions with deliberately uninformative distractors.

[Paper (PDF)](manuscript.pdf) · [Editable text](manuscript.md) · [Methods and study files](study/README.md) · [Manuscript history](study/VERSION.md)

[Author-owned software license](LICENSE) · [Third-party terms](LICENSING.md) · [Previous version](https://github.com/VEDANT1602/delayed-cue-recall/tree/cd872c60f76cf462f21d640185c099bb2c1359fa)
