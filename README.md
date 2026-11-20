<p align="center">
  <img src="assets/badges/paper.svg" alt="Paper: arXiv">
  <img src="assets/badges/code.svg" alt="Code: coming soon">
</p>

<p align="center">
  <img src="assets/logo.png" width="640" alt="Fast-JEPA">
</p>

<p align="center">
  <strong>Advancing the Success–Latency Frontier for Latent World Models</strong>
</p>

<p align="center">
  Goal-directed proposals and direct terminal predictions for low-latency visual planning.
</p>

<p align="center">
  Yao Zhang · Bin Yu · Shijie Lian · Haishan Liu · Changti Wu<br>
  Yu Wu<sup>†</sup> · Kai Chen<sup>†</sup><br>
  <sub>† Corresponding author</sub>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.25927">Paper</a> ·
  <a href="#method">Method</a> ·
  <a href="#results">Results</a> ·
  <a href="#dynagoal">DynaGoal</a> ·
  <a href="#citation">Citation</a>
</p>

---

## Overview

Latent world models predict the outcomes of action sequences, but comparing candidate plans through recursive rollout remains expensive at test time.

**Fast-JEPA** is a latent world model planning framework that advances the **success–latency frontier** of predictive search. It reorganizes predictive computation into two complementary stages:

- **Goal-directed proposal:** Construct an action sequence using intermediate predictions to guide action selection.
- **Direct terminal evaluation:** Predict the outcomes of complete candidate sequences directly, avoiding recursive state rollout during refinement.

By using predictions differently across planning stages, Fast-JEPA enables efficient action-sequence search while maintaining competitive task success.

## News

- **November 2026:** Fast-JEPA is available on [arXiv](https://arxiv.org/abs/2611.25927).
- **Coming soon:** Source code, pretrained checkpoints, and evaluation instructions.

## Method

<p align="center">
  <a href="assets/figures/method-overview.pdf">
    <img src="assets/figures/method-overview.svg" width="100%" alt="Fast-JEPA method overview">
  </a>
</p>

Fast-JEPA separates action-sequence planning into proposal construction and candidate refinement.

1. **Visual representation.** Encode the current observation and visual goal into a shared latent space.
2. **Goal-directed proposal.** Sequentially construct an action sequence using intermediate world-model predictions and goal-conditioned action generation.
3. **Efficient refinement.** Apply lightweight Cross-Entropy Method (CEM) search around the proposal, evaluating complete candidate sequences through direct terminal predictions.
4. **Execution and replanning.** Select the lowest-cost candidate, execute the selected action sequence according to the planning horizon, and replan from subsequent observations.

Both intermediate and terminal predictions are produced by the same forward world model. The key distinction lies in how predictions are used: intermediate predictions support action construction, while terminal predictions enable efficient candidate comparison.

For technical details, see our [paper](https://arxiv.org/abs/2611.25927).

## Results

### Main Results

Fast-JEPA is evaluated on four goal-conditioned visual control tasks: **PushT, Reacher, Cube, and TwoRoom**.

| Method | PushT | Reacher | Cube | TwoRoom | Avg. SR (%) | Latency (ms) |
| :-- | --: | --: | --: | --: | --: | --: |
| INTACT | 87.33 ± 2.31 | 96.00 ± 0.00 | 100.00 ± 0.00 | 99.00 ± 0.00 | 95.58 ± 0.58 | 26.19 ± 0.22 |
| INTACT Guarded A | 91.33 ± 1.53 | 97.00 ± 0.00 | 99.67 ± 0.58 | 97.33 ± 0.58 | 96.33 ± 0.38 | 127.09 ± 2.81 |
| **Fast-JEPA** | **90.00 ± 3.00** | **96.33 ± 1.15** | **100.00 ± 0.00** | **99.67 ± 0.58** | **96.50 ± 0.66** | **10.88 ± 0.04** |

Fast-JEPA achieves **96.50% average success rate** with **10.88 ms planning latency**, providing a **2.41× speedup** over the locally evaluated INTACT baseline.

Results are reported as mean ± standard deviation across three independently trained models, with 100 evaluation episodes per task and model. Latency represents complete planning time measured on an NVIDIA RTX PRO 6000 Blackwell Server Edition.

### Success–Latency Frontier

<p align="center">
  <a href="assets/figures/success-latency.pdf">
    <img src="assets/figures/success-latency.svg" width="680" alt="Success-latency frontier of Fast-JEPA">
  </a>
</p>

Fast-JEPA maintains competitive success rates at substantially lower planning latency across varying search budgets, advancing the success–latency frontier of latent world-model planning.

## DynaGoal

We introduce **DynaGoal**, a benchmark designed to evaluate world-model planning under dynamic out-of-distribution control conditions, execution deadlines, and controlled response delays.

DynaGoal consists of three evaluation protocols:

- **Shift:** Evaluates adaptation to changing goals or environment configurations during execution.
- **Deadline:** Evaluates goal completion under time budgets that include both planning and action execution.
- **Delay:** Evaluates robustness to controlled planning response delays.

All protocols evaluate models trained on the original tasks without benchmark-specific fine-tuning.

| Protocol | Evaluation Setting | Fast-JEPA Avg. SR (%) | Latency (ms) |
| :-- | :-- | --: | --: |
| Shift | Dynamic goal and environment changes | 73.83 ± 4.16 | 13.42 ± 0.11 |
| Deadline | Planning and execution time constraints | 87.67 ± 0.38 | 13.75 ± 0.03 |
| Delay | Response delays: 0 / 300 / 500 / 800 ms | See paper | See paper |

Fast-JEPA achieves **73.83%** average success on DynaGoal-Shift and **87.67%** on DynaGoal-Deadline.

For complete benchmark definitions and evaluation protocols, refer to the [paper](https://arxiv.org/abs/2611.25927).

## Code Release

The official implementation of Fast-JEPA will be released soon.

The release is planned to include:

- Training and evaluation code
- Model configurations and pretrained checkpoints
- Benchmark and evaluation instructions
- Reproduction scripts for the main experiments

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

## License

License information will be provided with the code release.
