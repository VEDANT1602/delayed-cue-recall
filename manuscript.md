# Predicting corridor texture during delayed cue recall

**Vedant Thaker**  
Working paper · 18 November 2026

## Abstract

At a junction, the controller needs a cue from the corridor entrance even when it can already predict the textures along the way. We test that distinction in a two-choice corridor with four textures. Each texture sequence appears with both entrance cues. Recurrent controllers start from paired weights and receive identical batches and update budgets under four objectives. After 220 updates, all 48 controllers chose correctly at every tested delay, including an interval four times the longest training delay. At a held-out delay of 16 steps, the next-texture head predicted 79.89% of upcoming symbols correctly, close to the generator's 80% ceiling. Removing the cue returned choice accuracy to exactly 50%. An exploratory look at earlier checkpoints revealed what final accuracy concealed. At 60 updates and a delay of 16, mean recall was 95.15% with choice supervision alone and 79.17% with next-texture prediction. The paired difference across 12 seeds was -15.97 percentage points. In this task, learning predictable sensory structure did not accelerate retention of the cue.

## 1. What must a controller remember?

A corridor can have an obvious visual rhythm and still reveal nothing about the turn at its end. Consider a sign at the entrance that indicates left or right. A controller must keep one bit through the corridor. Predicting the next wall texture is a different problem. Whether these two objectives help one another is an empirical question about the representation and the training budget.

Jaderberg et al. showed that auxiliary objectives can improve learning in reinforcement-learning environments [1]. Mirowski et al. used auxiliary predictions in agents learning to navigate complex three-dimensional environments [2]. Ororbia and Kelly evaluated a predictive-processing cognitive architecture with associative memory on four maze tasks, including one that tests memory [3]. These systems combine multiple mechanisms and evaluate agents that act in an environment. The present task isolates the memory component using a supervised recurrent choice model and a known transition distribution. It asks how a useful sensory predictor affects the speed and eventual accuracy of a different decision.

The distinction also matters for interpreting a model's outputs. Work on population codes shows how a particular interpretation of the variables carried by neural activity depends on an explicit generative model [4]. Here the generating variables are known by design. A texture can be predicted well even when the controller has forgotten the cue. The experiment therefore tests the choice directly and includes an intervention that removes its only source of information.

## 2. A corridor in which texture cannot reveal the turn

Each episode begins with a left or right entrance token. At each of d following steps the controller observes one of four textures arranged on a cycle. The first texture is uniform. Each transition advances one position clockwise with probability 0.8 and moves one position counterclockwise with probability 0.2. After the final observed texture, the correct response is the entrance cue. The model receives six binary input channels, two for cue identity and four for texture. Cue channels are zero throughout the corridor. Textures have no dependence on the cue. The task has forced forward movement until the terminal choice. There is no path planning or reward-driven exploration.

Every sampled texture path appears twice in the same batch, once with each cue. This pairing removes an accidental correlation that a finite training set could otherwise contain. Balancing the held-out examples in the same way also gives an exact intervention check. With the initial cue erased, the two episodes in a pair are identical and have opposite correct answers. Any deterministic classifier must get exactly one right. Test paths use a separate random stream from training. At short delays the small alphabet allows repeated paths, so independent draws need not be unique paths. Delays of 16 and 32 are entirely absent from training.

All models use one tanh recurrent layer of width 12, a two-logit choice head and a four-logit auxiliary head. They contain 318 parameters, of which 52 belong to the auxiliary head and receive no gradient in the choice-only condition. The first objective minimizes terminal cross-entropy. The second adds next-texture cross-entropy averaged across corridor steps at weight 0.5. Its target after the last observation is one additional sample from the same transition rule. The third uses the same auxiliary weight but shuffles the target trajectories across batch members. This preserves their marginal distribution while weakening their relationship to a particular input. The fourth asks the auxiliary head to repeat the original cue at every corridor step. It uses the same four logits with labels restricted to two classes and receives extra cue supervision.

The experiment uses 12 seeds and four objectives. Within a seed, all four start with identical network weights and see the same ordered training batches. Adam takes 220 updates at a learning rate of 0.003. Each update sees 24 counterbalanced pairs at one uniformly sampled delay from 2, 4 and 8. The joint gradient norm is clipped at one. Equal updates match data exposure. A model with the additional head loss performs more computation per update. There is no early stopping or tuning on the longer delays. At each test delay, each seed contributes 512 independently drawn texture paths, or 1,024 cue-labeled episodes.

## 3. A perfect final score conceals a learning difference

All conditions reach 100% choice accuracy on all five delays after 220 updates. That includes 32 distractor steps after training only as far as eight. The full comparison gives 240 seed-by-objective-by-delay cells at the final checkpoint. Every cell has 1,024 correct choices. The cue-erasure test gives 512 correct in all cells. The paired construction makes the latter count exact. The former establishes successful recall on these samples, without establishing how long an arbitrarily extended recurrence could maintain the bit.

Earlier checkpoints were added after the terminal ceiling became apparent. Their comparisons are exploratory. Table 1 reports mean choice accuracy across the 12 seeds at update 60. The first three rows use delays drawn from the training set, and the last two use unseen lengths. Every entry uses new draws of the corridor textures.

| Delay in steps | Choice only % | Next texture % | Shuffled target % | Repeated cue % |
| --- | --- | --- | --- | --- |
| 2 | 100.00 | 99.10 | 100.00 | 92.47 |
| 4 | 100.00 | 87.80 | 98.86 | 87.72 |
| 8 | 94.57 | 80.40 | 89.31 | 74.00 |
| 16 | 95.15 | 79.17 | 88.24 | 66.69 |
| 32 | 91.79 | 79.17 | 85.92 | 66.64 |

![Figure 1](figure.png)

Figure 1. Mean recall across 12 seeds at the 16-step held-out delay. Points show the three measured checkpoints. The lines connect measurements. Performance at 220 updates hides the early differences between objectives.

At delay 16, seven of the next-texture runs already make all 1,024 choices correctly. The other five get 512 or 513 right. A mean near 80% therefore describes a mixture of nearly perfect and chance-level controllers. It does not describe how a typical controller behaves on four out of five individual trials. The choice-only and texture conditions differ by -15.97 percentage points when computed as texture minus choice. A descriptive percentile bootstrap across the 12 paired seeds gives an interval from -28.62 to -3.73 points. The included script uses 20,000 resamples. Episodes sharing the same trained network are kept together. This interval applies to the chosen checkpoint and generator and carries no correction for the other inspected checkpoints or delays.

At 20 updates, all four conditions score between 50.00% and 50.22% at delay 16. Shorter delays already separate them. At delay two, choice supervision alone gives 81.66% and texture prediction gives 71.00%. By the final checkpoint that gap has disappeared. Repeating the task-relevant cue at every step also produces slower recall at update 60. Extra supervision, even for the right bit, does not guarantee faster optimization under a single shared learning rate and loss weight.

## 4. Did the prediction head learn its own task?

At delay 16 and update 220, the next-texture head gets 79.89% of transitions right. A predictor that always chooses the clockwise successor has an expected accuracy of 80% under the generator. The untrained auxiliary head in the choice-only model gets 24.92% right. Thus the final decision ceiling coexists with a large difference in predictive skill. The texture head's accuracy is 58.89% at update 60. It continues to improve after several seeds have already learned the terminal choice. These measurements separate two outcomes that an aggregate task score would combine.

The matched architecture and training batches rule out different exposure to particular corridors as an explanation for the objective ordering. Several other explanations remain available. The auxiliary gradient might redirect a small recurrent state toward texture. It might simply alter gradient scale and interact with clipping or Adam. The repeated-cue variant starts with an additional head trained to suppress two unused output classes. Those optimization details can affect early results independently of a target's relevance. Hidden-state probes, gradient alignment measurements, and a sweep that changes loss weight and hidden width together could distinguish these mechanisms.

## 5. Scope and next experiment

The present model has perfect one-hot sensory observations, one cue bit, a short path and no consequential action until the end. A richer task might make next observations depend on the cue, reveal a noisy version of it, or let the agent choose which sensory evidence to gather. Each change would alter what a good predictor needs to remember. A direct follow-up would vary the transition's dependence on the cue while holding the marginal texture frequencies fixed. Cues could then be erased selectively at evaluation to separate evidence reacquisition from memory retention. A biologically motivated predictive-coding model would provide a distinct architecture for that test. The current results are computational observations about supervised learning on generated sequences. They contain no measurements of human or animal cognition.

## Reproducibility

Run `python run.py` with Python 3.11, NumPy 1.26 and CPU PyTorch 2.9. The script generates all observations, fixes model and sampling seeds, requests deterministic operations and writes 720 rows to `results.csv`. Each row records the seed, objective, delay, update, choice count, cue-erasure count and next-texture accuracy. Repeated evaluation at the same seed and delay uses the same held-out examples across objectives and checkpoints.

## Research and writing assistance

Vedant Thaker supplied the research brief and source materials. OpenAI Codex assisted with study design, implementation, execution, literature review and drafting. Clever Humanizer suggested edits to the abstract. The final wording was checked against the analysis.

## References

[1] M. Jaderberg, V. Mnih, W. M. Czarnecki, T. Schaul, J. Z. Leibo, D. Silver and K. Kavukcuoglu. Reinforcement Learning with Unsupervised Auxiliary Tasks. ICLR, 2017. https://arxiv.org/abs/1611.05397

[2] P. Mirowski et al. Learning to Navigate in Complex Environments. ICLR, 2017. https://arxiv.org/abs/1611.03673

[3] A. Ororbia and M. A. Kelly. Maze Learning using a Hyperdimensional Predictive Processing Cognitive Architecture. AGI, 2022. https://arxiv.org/abs/2204.00619

[4] S. Shivkumar, R. Lange, A. Chattoraj and R. Haefner. A probabilistic population code based on neural samples. NeurIPS, 2018. https://proceedings.neurips.cc/paper/2018/hash/5401acfe633e6817b508b84d23686743-Abstract.html
