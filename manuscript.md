# Predicting the distractor slows acquisition of delayed recall in recurrent networks

Vedant Thaker  
Independent Researcher, Jeffersonville, Indiana, USA  
vedant.thaker1602@gmail.com

21 November 2026

### Abstract

A recurrent network can learn to predict the next observation while it learns which earlier observation to remember. Does that extra objective change how long reliable recall takes? Across 16 prospectively specified, paired GRU training runs, a network trained only to recover a binary cue reached sustained 90% accuracy after 70.0 optimizer updates on average. Predicting the intervening symbols as well took 78.75 updates. The paired increase was 8.75 updates (95% seed-bootstrap interval 5.63–11.88). A separate experiment asked a network to retain the recorded activity of a smartphone user across an unrelated segment of movement. There, adding a forecast of the distractor increased restricted mean time to sustained 75% validation accuracy from 350 to 733 updates. Ordinary activity classification on intact windows gave similar average scores with either objective. The learning delay appeared when the information required at the end preceded a predictable distractor.

## Why measure how long recall takes?

A network trained to remember a cue can also receive supervision at every intervening step. Predicting the next observation supplies many additional training targets. If those observations carry no information about the cue, a better forecast need not improve the final decision. Auxiliary tasks can help representation learning [1], and their gradients can support or oppose a primary objective [2]. Whether a recurrent model learns a delayed decision sooner under such supervision is an empirical question.

Delayed-copy benchmarks test whether a recurrent state carries information over intervening observations [3]. Here a terminal readout must retain a single input while a second head learns a predictable distractor. The comparison pairs the same initialization and training data across objectives. The endpoint is the update at which recall first meets a sustained accuracy criterion. Time to a specified quality target is an established perspective on training [4]. It is particularly relevant here. At update 60, six of 20 exploratory tanh networks trained with the auxiliary loss scored at most 55% while the other fourteen scored at least 95%. Their mean describes neither group well.

## A paired experiment with a declared stopping criterion

Each example starts with one of two cue symbols. Subsequent symbols occupy four positions on a cycle. From a uniform initial position the cycle advances one position with probability 0.8 and retreats one with probability 0.2. Both cue values accompany each sampled path. Predicting the cue from the path alone therefore gives exactly 50% accuracy. The decision head reads the final hidden state. In the joint objective a second head predicts the next texture at each distractor step, adding 0.5 times mean next-texture cross-entropy to terminal cue cross-entropy. The decision-only arm instantiates the same heads and sets auxiliary weight to zero.

At each of 200 training updates, draw a delay uniformly from 2, 4 and 8 and sample 24 paths, giving 48 examples. Both arms within a pair see identical paths and start from identical weights. Adam uses learning rate 0.003 and gradient-norm clipping at one. Every ten updates, measure accuracy on a fixed panel of 256 independent paired paths for each test delay, giving 512 decisions. Delay 16, twice the longest training delay, is primary. Delay 8 stays inside the training span and delay 32 tests a longer extrapolation. Panel generation uses a separate random stream.

The event occurs on the second of two consecutive visits at or above 90% accuracy. A run without an event by update 200 is right-censored there. The restricted mean time is the area under the event-free curve up to 200 [5]. Because everyone has the same administrative horizon and no earlier censoring, it also equals the mean of the event times capped at 200. Positive paired differences mean that forecasting required more updates. The percentile interval resamples complete training-seed pairs 20,000 times. It describes training variability under the specified sampling scheme. A ten-update inspection interval limits temporal resolution.

The public plan reserved 16 GRU seed pairs, numbered 7900–7915, and fixed the endpoint and directional decision rule before these models ran. The rule required a positive lower endpoint of a two-sided 95% seed-bootstrap interval. Three exploratory panels used 20 distinct seed pairs per recurrent cell. These compared a tanh RNN, a gated recurrent unit [6] and an LSTM [7], each with hidden width 12. The symbolic encoders have 240, 720 and 960 trainable parameters respectively. The matched comparison is between objectives within a cell.

## Forecasting delayed acquisition in the reserved GRU runs

All 16 runs in each confirmed condition reached the criterion. Twelve pairs took longer with joint supervision and four tied. The average increase was 8.75 updates, or 12.5% of the 70-update control mean. Using 80% or 95% accuracy also gave an 8.75-update difference in this block. All models guessed exactly half the terminal decisions when the cue was erased at the horizon. The auxiliary arm eventually passed even in pairs with an early deficit.

The three exploratory comparisons showed the same direction, with smaller differences for the gated cells than for the tanh cell. Their point estimates answer how each implementation responded to the extra target at the chosen width and optimizer. They do not isolate a causal effect of gate type.


| Seed panel | Pairs | Decision only | With forecast | Paired difference [95% interval] |
| --- | --- | --- | --- | --- |
| GRU, confirmation | 16 | 70.0 | 78.75 | +8.75 [5.63, 11.88] |
| Tanh, exploratory | 20 | 56.5 | 71.5 | +15.0 [9.0, 22.5] |
| GRU, exploratory | 20 | 65.0 | 71.5 | +6.5 [3.5, 9.5] |
| LSTM, exploratory | 20 | 57.5 | 63.5 | +6.0 [3.0, 9.0] |
Table 1. Restricted mean acquisition time in optimizer updates at a 16-observation delay. Every symbolic run crossed by 200 updates. The three exploratory intervals are descriptive.

## Remembering a measured movement after another movement

In a six-activity experiment, the cue itself comes from recordings. The public UCI smartphone activity dataset supplies nine waist-mounted inertial channels and the activity label for each window [8]. After averaging consecutive groups of four 50 Hz samples, choose a 16-reading prefix from a labeled window and independently draw a 16-reading suffix. In each group of six examples, give the very same suffix to one prefix from each activity. The target is the observed prefix label. A tenth channel indicates which segment is currently visible. This construction makes the suffix alone exactly chance for the balanced six-class decision. It also makes the junction artificial. No claim about naturally occurring transitions follows.

Original windows overlap within subjects, so a random split of windows could leak person-specific information [9]. Seventeen people (5,996 source windows) contributed to fitting, four distinct people (1,356 windows) to validation, and the nine official test participants (2,947 windows) to final evaluation. Standardization uses fitting data alone. A width-48 GRU reads only its final state for the activity decision. The auxiliary term is 0.5 times the mean squared error of the next standardized reading within the distractor segment. Both arms see the same training examples for 1,500 updates, with 36 examples per update and Adam at 0.002. Twelve seed pairs use fixed 768-example validation and test panels. Validation visits occur every 50 updates. The event requires two consecutive visits at or above 75%, well above the one-in-six baseline.

Decision-only training reached criterion in 12 of 12 runs, compared with 11 of 12 with forecasting. The restricted means were 350.0 and 733.3 updates. All twelve paired differences were positive (mean +383.3, 95% seed-bootstrap interval 237.5–566.7). At the 1,500-update horizon, the held-out test means were 90.12% and 83.06%. One auxiliary run remained at chance and largely drove that terminal difference. For the other eleven pairs, mean test accuracy was 89.99% versus 89.10%. Erasing the prefix reduced every model to exactly 128 correct out of 768. Inference about new individuals would require additional participant samples.

![Figure 1. Fraction of runs yet to reach the criterion at each visit. Left, 16 prospectively specified symbolic GRU pairs. Right, 12 exploratory measured activity-label pairs. The final step at 8.3% on the right contains the one administratively censored auxiliary run. The panels have different horizons and target accuracies.](study/acquisition.png)

## What carries over to intact activity windows?

An intact activity window addresses a different decision. Every reading can carry information about the activity currently underway. Using the same 17/4/9 participant split, fit a width-48 GRU to classify complete windows and compare the same optional next-reading objective. The readout averages the recurrent states. Each of four seed pairs trains for 35 complete epochs. The earliest epoch with the best validation macro-F1 determines the checkpoint for testing. A multinomial logistic regression on 63 window summaries provides a nonrecurrent baseline.

Mean held-out macro-F1 across the four paired GRU fits was 90.48% for activity only and 89.82% with forecasting. The logistic baseline achieved 90.56%. These close scores in four seed pairs cannot establish equivalence, but they discourage a general claim that forecasting sensor data worsens activity recognition. The measured prefix task makes the forecast target independent of the remembered activity by design. In an intact window, forecasting may reward features that the classifier needs.

As a further control, binary cue recall also ran on three-axis acceleration distractors from held-out subjects. The delays associated with forecasting at 16 readings were 23.3 updates for tanh, 26.7 for GRU, and 30.0 for LSTM across twelve seed pairs per architecture. This control changes where the observations come from while retaining the artificial cue. The supplement gives the full trajectories, survivor counts and threshold sensitivities.

## Interpretation

Within the specified recurrent models, forecasting a predictable but task-irrelevant segment delayed reliable access to an earlier label. Timing the transition avoids treating an average over runs at chance and runs near ceiling as a common level of partial recall. It also keeps a run that has not crossed by the horizon in the analysis.

The causal intervention here is the addition of a fixed auxiliary loss in individually matched runs. The experiment does not identify where in the network the delay arises. Numerical weight 0.5 multiplies cross-entropy in the symbol task and squared error in the sensor task, so sizes of effects across the two scales are not directly comparable. Fixed panels also leave uncertainty about other participants, sensors and tasks. Rates under longer training, larger parameter budgets and adaptive weighting remain distinct questions.

## Data and code availability

The companion repository contains the training scripts, per-visit counts, event times, environment versions, source audit and parameter rationale. The UCI data and license are available from reference [8]. The prospective GRU plan is preserved in public commit 68aa06cc899fd30562a3dd6359e2a350f06bcb2d. README.md identifies the exact implementation and its digest.

[Study files, software and source records](https://github.com/VEDANT1602/delayed-cue-recall/tree/main/study)

## Author and disclosures

Vedant Thaker is the sole author and identified the question and the need to evaluate learning time, gated recurrent models and measured tasks. OpenAI Codex assisted in specifying the prospective test, implementing and running the training and analysis, checking sources and drafting the manuscript.

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors. The author declares no competing interests.

## References

[1] M. Jaderberg et al. Reinforcement learning with unsupervised auxiliary tasks. International Conference on Learning Representations (2017). <https://arxiv.org/abs/1611.05397>

[2] Y. Du, W. M. Czarnecki, S. M. Jayakumar, M. Farajtabar, R. Pascanu and B. Lakshminarayanan. Adapting auxiliary losses using gradient similarity. arXiv:1812.02224 (2018, revised 2020). <https://arxiv.org/abs/1812.02224>

[3] M. Arjovsky, A. Shah and Y. Bengio. Unitary evolution recurrent neural networks. Proceedings of Machine Learning Research 48, 1120–1128 (2016). <https://proceedings.mlr.press/v48/arjovsky16.html>

[4] P. Mattson et al. MLPerf training benchmark. Proceedings of Machine Learning and Systems 2, 336–349 (2020). <https://proceedings.mlsys.org/paper_files/paper/2020/hash/411e39b117e885341f25efb8912945f7-Abstract.html>

[5] P. Royston and M. K. B. Parmar. Restricted mean survival time: an alternative to the hazard ratio for the design and analysis of randomized trials with a time-to-event outcome. BMC Medical Research Methodology 13, 152 (2013). <https://doi.org/10.1186/1471-2288-13-152>

[6] K. Cho et al. Learning phrase representations using RNN encoder–decoder for statistical machine translation. Proceedings of EMNLP, 1724–1734 (2014). <https://doi.org/10.3115/v1/D14-1179>

[7] K. Greff, R. K. Srivastava, J. Koutník, B. R. Steunebrink and J. Schmidhuber. LSTM: A search space odyssey. IEEE Transactions on Neural Networks and Learning Systems 28, 2222–2232 (2017). <https://doi.org/10.1109/TNNLS.2016.2582924>

[8] J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto and X. Parra. Human Activity Recognition Using Smartphones [dataset]. UCI Machine Learning Repository (2013). <https://doi.org/10.24432/C54S4K>

[9] A. Dehghani, O. Sarbishei, T. Glatard and E. Shihab. A quantitative comparison of overlapping and non-overlapping sliding windows for human activity recognition using inertial sensors. Sensors 19, 5026 (2019). <https://doi.org/10.3390/s19225026>
