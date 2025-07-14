# CERNet – Class‑Embedding Predictive‑Coding RNN

Note: This repository is intended solely for reproducing the experimental results presented in our paper.
Future development and maintenance of the project are hosted separately at example.github.com (link to be updated upon acceptance to preserve double-anonymity of RA-L).

*Unified robot motion generation, intention recognition & confidence estimation*

CERNet is a hierarchical predictive‑coding recurrent neural network that **simultaneously**

* Generates robust robot motions in real time
* Infers the class (intent) of observed trajectories
* Gauges its own confidence — all inside a single compact model.

We validated CERNet on the humanoid robot **Reachy**. It reproduced 26 kinesthetically‑taught alphabet trajectories with **76 % lower reproduction error** than a parameter‑matched single‑layer baseline, and recognised the demonstrated class online with **68 % Top‑1 / 81 % Top‑2 accuracy**. Internal prediction error naturally encodes confidence, so no extra classifier is required.

---

## Quick start

## 🚀 Running Instructions

### 1. Setup Environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # For macOS/Linux
# OR
.venv\Scripts\activate           # For Windows

pip install -r requirements.txt  # (Assumes pip-based env. If using conda, adapt accordingly.)
cd Code
```

---

### 2. Train Models (Paper: All 60 Models)

```bash
python exp0_TrainAllSixtyModels.py
```

---

### 3. Experiment 1 – Alphabet Drawing (Simulation)

```bash
python exp1_SequenceRenerageAllModels.py --workers 20
```

> ⚠️ To run on **real robot**, you must append `--robot`, but **calibration may differ** and could **damage your hardware**. We do **not recommend** running this on a physical robot unless you collect your own dataset using `Code/experiments/a1_recordTrajectory_Robot.py`.

#### Visualize Results

```bash
python -m experiments.e2_evaluate_dtw
```

---

### 4. Experiment 2 – Perturbation Resistance (Simulation)

```bash
python exp2_PerturbationExperiments.py
```

#### Visualize Results

```bash
python -m experiments.e3_evaluatePerturbation --result_dir ../Data/exp2_Perturbation
```

---

### 5. Experiment 3 – Class Inference & Confidence Estimation (Simulation)

```bash
python exp3_InferLetterExperiments.py \
    --base-config ./toml_configs/multi_large.toml \
    --seq 0-25 \
    --seed 1-10 \
    --num-workers 10
```

#### Visualize Results

```bash
python -m experiments.e4_RecognitionAccuracy --root ../Data/exp3_InferLetter
```

## Pre‑print & citation

If you use this code, please cite:

```bibtex
To be updated after acceptance
```

---

## Videos

| Demo                                   | Link                                                          |
| -------------------------------------- | ------------------------------------------------------------- |
| Alphabet reproduction                  | [https://youtu.be/VZp5y7\_iwJ4](https://youtu.be/VZp5y7_iwJ4) |
| Real‑time class inference & confidence | [https://youtu.be/wFMh7ijGNxk](https://youtu.be/wFMh7ijGNxk)  |

---

## License

Released under the **eCILL license** (see `LICENSE`).

---
