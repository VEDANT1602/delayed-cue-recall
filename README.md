# Predicting the distractor slows acquisition of delayed recall in recurrent networks

Working paper by Vedant Thaker, Independent Researcher, Jeffersonville, Indiana. Correspondence: vedant.thaker1602@gmail.com.

A recurrent network can learn to predict the next observation while it learns which earlier observation to remember. Does that extra objective change how long reliable recall takes? Across 16 prospectively specified, paired GRU training runs, a network trained only to recover a binary cue reached sustained 90% accuracy after 70.0 optimizer updates on average. Predicting the intervening symbols as well took 78.75 updates. The paired increase was 8.75 updates (95% seed-bootstrap interval 5.63–11.88). A separate experiment asked a network to retain the recorded activity of a smartphone user across an unrelated segment of movement. There, adding a forecast of the distractor increased restricted mean time to sustained 75% validation accuracy from 350 to 733 updates. Ordinary activity classification on intact windows gave similar average scores with either objective. The learning delay appeared when the information required at the end preceded a predictable distractor.

- [Read the paper](manuscript.pdf)
- [Read the plain-text manuscript](manuscript.md)
- [Reproduce the analysis and inspect the source and data records](study/README.md)
- [Software license](LICENSE) and [third-party terms](LICENSING.md)
- [Preparation record](study/PREPARATION.md)
- [Previous manuscript and supplements](https://github.com/VEDANT1602/delayed-cue-recall/tree/c8a7031a587260b06416deb4af622c6fafcbd580)
