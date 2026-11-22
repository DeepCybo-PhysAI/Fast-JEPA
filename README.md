<p align="center"><img src="assets/logo.png" width="640" alt="Fast-JEPA: fast goal-directed planning"></p>
<p align="center"><strong>Advancing the Success–Latency Frontier for Latent World Models</strong></p>
<p align="center">Goal-directed proposals and direct terminal predictions for low-latency visual planning.</p>
<p align="center">
Yao Zhang · Bin Yu · Shijie Lian · Haishan Liu · Changti Wu · Yu Wu<sup>†</sup> · Kai Chen<sup>†</sup><br>
<sub>† Corresponding author</sub>
</p>

<div align="center">
<a href="https://github.com/DeepCybo-PhysAI/Fast-JEPA">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-DeepCybo%2FFastJEPA-blue?logo=github">
</a>
<a href="https://www.alphaxiv.org/abs/2611.25927">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2601.15197-b31b1b.svg">
</a>
<a href="https://github.com/DeepCybo-PhysAI/Fast-JEPA/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
</a>
</div>
---

## Overview

Comparing action sequences through recursive world-model rollout can make online planning expensive. **Fast-JEPA** reorganizes predictive computation: it constructs a goal-directed reference plan, then compares complete candidate sequences through direct terminal predictions under a shared visual-goal objective.

This retains intermediate predictions where action construction needs them while avoiding recursive state rollout during candidate evaluation.

## Method

<p align="center"><a href="assets/figures/method-overview.pdf"><img src="assets/figures/method-overview.svg" width="100%" alt="Fast-JEPA pipeline: observation and goal encoding, sequential action proposal, light CEM, direct terminal predictions, goal scoring, and execution"></a></p>

1. **Encode the current observation and visual goal.** Represent both in a shared latent space.
2. **Construct a goal-directed proposal.** Sequential action construction uses intermediate predictions anchored to the current observed representation.
3. **Refine with lightweight CEM.** Sample alternatives around the proposal and score complete candidates with batched terminal queries. The proposal and best-so-far plan remain eligible.
4. **Select, execute, and replan.** Select the lowest goal-cost plan, execute a short action prefix, and replan from the next observation.

## Results

Main four-task comparison, from Table 1. Values are mean ± standard deviation as reported in the paper; task columns and average success rate (SR) are percentages. These rows are **locally evaluated reference configurations**.

| Method | PushT | Reacher | Cube | TwoRoom | Avg. SR (%) | Planning latency (ms) |
| :-- | --: | --: | --: | --: | --: | --: |
| INTACT | 87.33 ± 2.31 | 96.00 ± 0.00 | 100.00 ± 0.00 | 99.00 ± 0.00 | 95.58 ± 0.58 | 26.19 ± 0.22 |
| INTACT Guarded A | 91.33 ± 1.53 | 97.00 ± 0.00 | 99.67 ± 0.58 | 97.33 ± 0.58 | 96.33 ± 0.38 | 127.09 ± 2.81 |
| **Fast-JEPA** | **90.00 ± 3.00** | **96.33 ± 1.15** | **100.00 ± 0.00** | **99.67 ± 0.58** | **96.50 ± 0.66** | **10.88 ± 0.04** |

Fast-JEPA reaches **96.50%** average success at **10.88 ms** mean complete planning latency, a **2.41×** speedup over the locally evaluated INTACT configuration (26.19 / 10.88).

The main evaluation uses 100 episodes per task and three independently trained models. Latencies are warmed, batch-one, complete planning measurements on an NVIDIA RTX PRO 6000 Blackwell Server Edition. Hardware, timing boundaries, model configurations, and protocols matter when comparing results; published reference results are distinguished from local evaluations in the full paper.

<p align="center"><a href="assets/figures/success-latency.pdf"><img src="assets/figures/success-latency.svg" width="580" alt="Paper Figure 1: success-latency frontier with published and locally evaluated configurations distinguished"></a></p>

## DynaGoal

**DynaGoal** evaluates original-task models under changing control conditions and timing constraints, without benchmark-specific training or fine-tuning.

| Protocol | What changes | Fast-JEPA avg. SR (%) | Planning latency (ms) |
| :-- | :-- | --: | --: |
| Shift | Goals or environment geometry change during control | 73.83 ± 4.16 | 13.42 ± 0.11 |
| Deadline | Goal completion must fit planning + execution budgets | 87.67 ± 0.38 | 13.75 ± 0.03 |
| Delay | Controlled response delays of 0, 300, 500, and 800 ms | See paper curves | See paper |

Shift and Deadline values are from Tables 2 and 3. Each protocol covers PushT, Reacher, Cube, and TwoRoom. OOD here refers to changed control conditions and execution constraints, rather than visual-domain or episode disjointness. See Appendix B.2 for case definitions, success criteria, and timing boundaries.

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{zhang2026fastjepa,
  title         = {Fast-JEPA: Advancing the Success--Latency Frontier for Latent World Models},
  author        = {Zhang, Yao and Yu, Bin and Lian, Shijie and Liu, Haishan and Wu, Changti and Wu, Yu and Chen, Kai},
  year          = {2026},
  eprint        = {2611.25927},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2611.25927}
}
```

## Contributing and support

For now, contributions focus on documentation corrections and questions about the paper. See [contribution guidance](.github/CONTRIBUTING.md) and [support information](.github/SUPPORT.md). Implementation-related instructions will be added when the code is released.

## License

A public redistribution license has not yet been selected for this repository. No open-source license grant is asserted by this project package. The maintainers should choose a license and verify the rights for included materials before public release; paper and code licensing may differ.
