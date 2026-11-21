# Predicting the distractor slows acquisition of delayed recall in recurrent networks

Research working paper by Vedant Thaker, Independent Researcher, Jeffersonville, Indiana, USA.

Predicting incoming observations gives a recurrent network an extra learning signal. Does that signal help it remember a label presented before a distractor? In 16 prospectively specified pairs of GRU runs, adding a distractor forecast delayed sustained 90% recall by 8.75 optimizer updates (70.00 versus 78.75, paired 95% interval 5.63–11.88). The original evaluation took place every ten updates. Measuring the same trajectories after every update and requiring eleven successive passing assessments gave a delay of 8.81 updates (95% interval 6.44–11.19). The effect extended to a constructed task that joined two separately recorded smartphone movements. There, one auxiliary run stayed at chance and accounted for most of the final test-accuracy difference. On intact activity windows, the mean classification scores were close. The evidence supports a learning-time cost on the two delayed tasks studied. Its practical importance for ordinary activity recognition remains uncertain.

[Paper (PDF)](manuscript.pdf) · [Editable text](manuscript.md) · [Methods and study files](study/README.md) · [Manuscript history](study/VERSION.md)

[Author-owned software license](LICENSE) · [Third-party terms](LICENSING.md) · [Previous version](https://github.com/VEDANT1602/delayed-cue-recall/tree/7483bd1958edf461f7550038e8e09be09d6941e8)
