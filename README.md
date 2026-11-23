
<p align="center"><img src="assets/logo.png" width="640" alt="Fast-JEPA: fast goal-directed planning"></p>
<p align="center"><strong>Advancing the Success–Latency Frontier for Latent World Models</strong></p>
<p align="center">Goal-directed proposals and direct terminal predictions for low-latency visual planning.</p>
<p align="center">
  Yao Zhang<sup>1,2,8</sup> · Bin Yu<sup>3,2,8</sup> · Shijie Lian<sup>4,2,8</sup> · Haishan Liu<sup>5,2</sup> · Changti Wu<sup>6,2</sup> · Yu Wu<sup>1,†</sup> · Kai Chen<sup>2,7,8,†</sup><br>
  <sup>1</sup>Wuhan University &nbsp;&nbsp; <sup>2</sup>Zhongguancun Academy<br>
  <sup>3</sup>Harbin Institute of Technology &nbsp;&nbsp; <sup>4</sup>Huazhong University of Science and Technology<br>
  <sup>5</sup>Nanjing University &nbsp;&nbsp; <sup>6</sup>East China Normal University<br>
  <sup>7</sup>Zhongguancun Institute of Artificial Intelligence &nbsp;&nbsp; <sup>8</sup>DeepCybo<br>
  <sub>† Corresponding author</sub>
</p>


<div align="center">
<a href="https://deepcybo-physai.github.io/Fast-JEPA/">
    <img alt="GitHub" src="https://img.shields.io/badge/Page-DeepCybo%2FFastJEPA-blue?logo=github">
</a>
<a href="https://www.alphaxiv.org/abs/2611.25927">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2611.25927-b31b1b.svg">
</a>
<a href="https://github.com/DeepCybo-PhysAI/Fast-JEPA/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg">
</a>
</div>

<br>

<p align="center">
  <a href="#overview">Why Fast-JEPA?</a> &nbsp;&middot;&nbsp;
  <a href="#installation">Installation</a> &nbsp;&middot;&nbsp;
  <a href="#training">Training</a> &nbsp;&middot;&nbsp;
  <a href="#evaluation">Evaluation</a> &nbsp;&middot;&nbsp;
  <a href="#method">Method</a> &nbsp;&middot;&nbsp;
  <a href="#results">Results</a> &nbsp;&middot;&nbsp;
  <a href="#dynagoal">DynaGoal</a>
</p>

---

## 🎉 News & Roadmap

- [x] 📄 **Paper & results released!** Check out our [paper](https://arxiv.org/abs/2611.25927) and [results](#results).
- [x] 🚀 **Code is now open source!** Get started with [installation](#installation), [training](#training), and [evaluation](#evaluation).
- [ ] 👀 **…**

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

Main four-task comparison, from Table 1. Values are mean ± standard deviation as reported in the paper; task columns and average success rate (SR) are percentages. Plan (ms) denotes complete planning latency. These rows are **locally evaluated reference configurations**.

<table width="100%">
  <thead>
    <tr>
      <th align="center" width="160">Method</th>
      <th align="center" width="125">PushT</th>
      <th align="center" width="125">Reacher</th>
      <th align="center" width="125">Cube</th>
      <th align="center" width="125">TwoRoom</th>
      <th align="center" width="125">Avg.&nbsp;SR&nbsp;(%)</th>
      <th align="center" width="150">Plan&nbsp;(ms)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">INTACT</td>
      <td align="center">87.33±2.31</td>
      <td align="center">96.00±0.00</td>
      <td align="center">100.00±0.00</td>
      <td align="center">99.00±0.00</td>
      <td align="center">95.58±0.58</td>
      <td align="center">26.19±0.22</td>
    </tr>
    <tr>
      <td align="center">INTACT&nbsp;Guarded&nbsp;A</td>
      <td align="center">91.33±1.53</td>
      <td align="center">97.00±0.00</td>
      <td align="center">99.67±0.58</td>
      <td align="center">97.33±0.58</td>
      <td align="center">96.33±0.38</td>
      <td align="center">127.09±2.81</td>
    </tr>
    <tr>
      <td align="center"><strong>Fast&#8209;JEPA</strong></td>
      <td align="center"><strong>90.00±3.00</strong></td>
      <td align="center"><strong>96.33±1.15</strong></td>
      <td align="center"><strong>100.00±0.00</strong></td>
      <td align="center"><strong>99.67±0.58</strong></td>
      <td align="center"><strong>96.50±0.66</strong></td>
      <td align="center"><strong>10.88±0.04</strong></td>
    </tr>
  </tbody>
</table>

Fast-JEPA reaches **96.50%** average success at **10.88 ms** mean complete planning latency, a **2.41×** speedup over the locally evaluated INTACT configuration (26.19 / 10.88).

The main evaluation uses 100 episodes per task and three independently trained models. Latencies are warmed, batch-one, complete planning measurements on an NVIDIA RTX PRO 6000 Blackwell Server Edition. Hardware, timing boundaries, model configurations, and protocols matter when comparing results; published reference results are distinguished from local evaluations in the full paper.

<p align="center"><a href="assets/figures/success-latency.pdf"><img src="assets/figures/success-latency.svg" width="580" alt="Paper Figure 1: success-latency frontier with published and locally evaluated configurations distinguished"></a></p>

### Search-budget comparison

**Figure 5.** Five methods are evaluated with 32 search configurations each. The left panel shows per-task success–latency operating points; the right panel shows the highest observed success within each mean-latency budget.

<p align="center"><a href="assets/figures/search-budget-points.pdf"><img src="assets/figures/search-budget-points.svg" width="49%" alt="Figure 5a: per-task success–latency operating points across search configurations"></a> <a href="assets/figures/search-budget-frontier.pdf"><img src="assets/figures/search-budget-frontier.svg" width="49%" alt="Figure 5b: best observed success within each mean-latency budget"></a></p>

### Predictive refinement in action

**Figure 6.** Matched PushT and Cube cases share the same initial states and goals. Predictive refinement rescues two rollouts, with callouts highlighting alignment and grasp differences. Success is evaluated within 50 steps; asterisks mark retained successful terminal observations, and the rightmost panels show the goals.

<p align="center"><a href="assets/figures/predictive-refinement-cases.pdf"><img src="assets/figures/predictive-refinement-cases.svg" width="100%" alt="Figure 6: matched PushT and Cube rollouts showing how predictive refinement improves alignment and grasping"></a></p>

## DynaGoal

**DynaGoal** evaluates original-task models under changing control conditions and timing constraints, without benchmark-specific training or fine-tuning.

<table width="100%">
  <thead>
    <tr>
      <th align="center" width="125">Protocol</th>
      <th align="center" width="430">Evaluation&nbsp;condition</th>
      <th align="center" width="235">Fast-JEPA&nbsp;avg.&nbsp;SR&nbsp;(%)</th>
      <th align="center" width="145">Planning&nbsp;latency&nbsp;(ms)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Shift</td>
      <td align="center">Changing goals or environment geometry</td>
      <td align="center">73.83±4.16</td>
      <td align="center">13.42±0.11</td>
    </tr>
    <tr>
      <td align="center">Deadline</td>
      <td align="center">Joint planning and execution time budget</td>
      <td align="center">87.67±0.38</td>
      <td align="center">13.75±0.03</td>
    </tr>
  </tbody>
</table>

Fast-JEPA results from Tables 2 and 3. Avg. SR denotes average success rate. Each protocol covers PushT, Reacher, Cube, and TwoRoom. OOD here refers to changed control conditions and execution constraints, rather than visual-domain or episode disjointness. See Appendix B.2 for case definitions, success criteria, and timing boundaries.

### DynaGoal-Delay

Success rates across PushT, Reacher, Cube, and TwoRoom under controlled total response delays of **0, 300, 500, and 800 ms**, with 100 episodes per task at each delay level.

<p align="center"><a href="assets/figures/dynamic-response-delay.pdf"><img src="assets/figures/dynamic-response-delay.svg" width="1000" alt="DynaGoal-Delay results: success-rate curves for INTACT, INTACT Guarded A, Fast-JEPA, LeWM, and Fast-LeWM across four tasks at response delays of 0, 300, 500, and 800 milliseconds"></a></p>

---

## Installation

Clone the repository and run the commands below from its root:

```bash
git clone https://github.com/DeepCybo-PhysAI/Fast-JEPA.git
cd Fast-JEPA
```

Datasets and checkpoints are not bundled. Prepare the datasets and train checkpoints using the steps below.

Use Linux x86_64, Python 3.10 and a CUDA-capable NVIDIA GPU. The locked environment
uses PyTorch 2.7.1+cu128 and TorchVision 0.22.1. Reference planning timings were
measured on an NVIDIA RTX PRO 6000 Blackwell GPU.

On Ubuntu 22.04, install the system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip \
  build-essential swig curl ca-certificates git ffmpeg zstd \
  libegl1 libgl1 libgles2 libglfw3 libglib2.0-0 \
  libx11-6 libxext6 libxrender1 libsm6 libasound2
```

Then install and check the Python environment:

```bash
bash scripts/install.sh
bash scripts/check_environment.sh
bash scripts/check_environment.sh --gpu
```

The installer creates `.venv` and installs the exact versions in
`requirements-blackwell.lock`. It also selects **pygame 2.6.1 / SDL 2.28.4** and
**opencv-python** as the active rendering backends. Keep this installation order:
pygame-ce and pygame share an import namespace, as do the two OpenCV variants.
The evaluation wrapper rejects the wrong pygame backend.

A compatible NVIDIA driver and EGL rendering support are required. GPU containers
must enable `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` when created;
setting it only after startup cannot inject missing driver libraries. Scripts set
`MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`.

`FAST_JEPA_PYTHON=/path/to/python` selects an existing compatible interpreter.
`FAST_JEPA_WHEELHOUSE` selects a cache for the three prefetched NVIDIA wheels.
Alternatively, use the supplied Dockerfile:

```bash
docker build -t fast-jepa .
docker run --rm --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v /path/to/cache:/data \
  -v /path/to/results:/opt/fast-jepa/outputs fast-jepa
```

## Data preparation

Set the cache root and prepare the four datasets:

```bash
export STABLEWM_HOME="${STABLEWM_HOME:-$PWD/cache}"
bash scripts/prepare_data.sh --dry-run
bash scripts/prepare_data.sh
```

Pinned revisions and archive SHA256 values are in `data/manifests/`. The preparer
supports resuming downloads, verifies the archives, extracts HDF5 and converts
PushT to Lance. Reserve space for archives and the prepared data together.

<table width="100%">
  <thead><tr>
    <th align="center" width="160">Task</th>
    <th align="center" width="400">Dataset</th>
    <th align="center" width="280">Prepared format</th>
  </tr></thead>
  <tbody>
    <tr><td align="center">PushT</td><td align="center"><a href="https://huggingface.co/datasets/quentinll/lewm-pusht">lewm-pusht</a></td><td align="center">HDF5 + Lance</td></tr>
    <tr><td align="center">Cube</td><td align="center"><a href="https://huggingface.co/datasets/quentinll/lewm-cube">lewm-cube</a></td><td align="center">HDF5</td></tr>
    <tr><td align="center">Reacher</td><td align="center"><a href="https://huggingface.co/datasets/quentinll/lewm-reacher">lewm-reacher</a></td><td align="center">HDF5</td></tr>
    <tr><td align="center">TwoRoom</td><td align="center"><a href="https://huggingface.co/datasets/quentinll/lewm-tworooms">lewm-tworooms</a></td><td align="center">HDF5</td></tr>
  </tbody>
</table>

Both `STABLEWM_HOME` and optional `LOCAL_DATASET_DIR` refer to the **parent** of
`datasets/`, not the dataset directory itself:

```text
cache/
  datasets/
    pusht_expert_train.h5
    pusht_expert_train.lance/
    ogbench/cube_single_expert.h5
    reacher.h5
    tworoom.h5
  checkpoints/
```

To reuse existing prepared data, set
`LOCAL_DATASET_DIR=/path/to/benchmark-cache`. Training outputs still use
`STABLEWM_HOME/checkpoints/`. PushT trains from Lance and evaluates from HDF5;
the other tasks use HDF5 for both.

Optional preparation commands:

```bash
bash scripts/prepare_data.sh --tasks pusht cube
bash scripts/prepare_data.sh --download-only
bash scripts/prepare_data.sh --skip-convert
bash scripts/prepare_data.sh --convert-only
bash scripts/prepare_data.sh --downloader aria2 --connections 8
```

## Training

Train one complete epoch for one task and training seed:

```bash
bash scripts/train.sh pusht 0
```

The tasks are `pusht`, `cube`, `reacher`, and `tworoom`; the training seeds are
0, 42, and 3072. For the complete grid:

```bash
for seed in 0 42 3072; do
  for task in pusht cube reacher tworoom; do
    bash scripts/train.sh "$task" "$seed"
  done
done
```

Training uses a seeded 90/10 split at the **clip** level, not the episode level.
Each run writes to `$STABLEWM_HOME/checkpoints/fast_TASK_sSEED_e1/`. Existing run
directories are not overwritten. Keep these checkpoint files together:

- `weights_epoch_1.pt`: model state dictionary;
- `config.json`: model architecture for strict loading;
- `run_metadata.json`: task, training seed and numerical policy.

## Evaluation

Evaluate a trained checkpoint using evaluation seed 42:

```bash
bash scripts/eval.sh pusht "$STABLEWM_HOME/checkpoints/fast_pusht_s0_e1/weights_epoch_1.pt" 42
```

The default is 100 episodes with **CEM64×2/top8**, CUDA Graph `all`, encoding
cache, unused-terminal proposal pruning and the original Actor action history.
H5/R5/block5 executes 25 primitive environment steps before each real observation.
Results and videos are written under `outputs/`.

Measure warmed, single-environment complete-planning latency:

```bash
PYTHONPATH=inference .venv/bin/python inference/benchmark_paired.py \
  --checkpoint "$STABLEWM_HOME/checkpoints/fast_pusht_s0_e1/weights_epoch_1.pt" \
  --mode fast_guarded --output outputs/pusht_b1_latency.json
```

### Four-task evaluation

Evaluate all four tasks across training seeds 0, 42, and 3072:

```bash
bash scripts/evaluate_all.sh --checkpoint-root "$STABLEWM_HOME/checkpoints" --dry-run
bash scripts/evaluate_all.sh --checkpoint-root "$STABLEWM_HOME/checkpoints" \
  --output "$PWD/outputs/main"
```

This evaluates 100 episodes per task/checkpoint at evaluation seed 42 and measures
batch-one latency independently for each checkpoint. Use a fresh output directory.
The outputs are `RESULTS.csv`, `SUMMARY.json`, and `SUMMARY.md`.

Macro averages the four tasks **within each training seed first**, then reports
mean, sample SD and sample variance across the three training seeds (`ddof=1`).
Missing cells are an error. Full settings are in
[reproducibility/SETTING.json](reproducibility/SETTING.json).

## DynaGoal evaluation

Evaluate Fast-JEPA on changing goals and geometry (**Shift**), execution-time
budgets (**Deadline**), and controlled response delays (**Delay**). Use the same
twelve checkpoints as the static tasks; no additional training is needed.

Run the static, Shift, and Deadline evaluations together:

```bash
bash scripts/reproduce_tables.sh "$STABLEWM_HOME/checkpoints" "$PWD/outputs/tables"
```

Each suite evaluates 100 episodes per task and checkpoint, totaling 1,200
episodes. Results include per-task and macro statistics across training seeds.
Choose a new output directory for each run.

Run Shift or Deadline individually:

```bash
bash scripts/evaluate_dynagoal.sh --subset shift \
  --checkpoint-root "$STABLEWM_HOME/checkpoints" --output "$PWD/outputs/table2"
bash scripts/evaluate_dynagoal.sh --subset deadline \
  --checkpoint-root "$STABLEWM_HOME/checkpoints" --output "$PWD/outputs/table3"
```

Evaluate Delay at **0, 300, 500, and 800 ms** using the Shift dynamics and
three fixed evaluation windows (0, 42, and 3072), with 100 cases per window:

```bash
bash scripts/evaluate_dynagoal.sh --subset delay \
  --checkpoint-root "$STABLEWM_HOME/checkpoints" --output "$PWD/outputs/delay"
```

The full Delay sweep covers four tasks, three training seeds, three evaluation
windows, and four delay levels: 14,400 episodes in total.

## Acknowledgments

Fast-JEPA builds upon the following open-source projects. We thank their authors for their valuable contributions and for making their code publicly available.

- [INTACT-JEPA](https://github.com/zju3dv/INTACT-JEPA)
- [Fast-LeWorldModel](https://github.com/Yuntian-Gao/Fast-LeWorldModel)
- [LeWM](https://github.com/lucas-maes/le-wm)

## License

Fast-JEPA is released under the [MIT License](LICENSE). Copyright and license notices for the upstream code are retained in the same file.

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
