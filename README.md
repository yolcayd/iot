# Attack-Type-Aware Analysis of Model Quantization for In-Vehicle CAN Intrusion Detection

Course project for YZM5015 *Artificial Intelligence and the Internet of Things* (MSc).

Deep-learning intrusion detection systems (IDS) for the in-vehicle Controller Area Network (CAN) are
typically quantized (float32 → float16 → INT8) before being deployed on resource-constrained edge
devices. Prior work reports the cost of quantization only as a change in *overall* accuracy. CAN
attacks, however, are not equally hard to detect: *coarse* attacks (DoS, fuzzing) visibly disturb the
ID distribution and message timing, while *stealthy* attacks (spoofing, malfunction, masquerade) reuse
valid IDs and only perturb payload bytes. This project measures the detection loss caused by
quantization **per attack type**, together with inference latency, model size and memory usage.

## Research questions

- **RQ1 (main):** Does quantization (float32 → float16 → INT8) cause different detection losses for
  different attack types? Is the F1 loss on stealthy attacks significantly larger than on coarse attacks?
- **RQ2:** Under a given inference-latency budget, which model/quantization combination offers the best
  trade-off between detection performance, latency and model size?
- **RQ3:** Do supervised (1D-CNN) and unsupervised (LSTM autoencoder) detectors differ in their
  robustness to quantization?

Datasets: [Car-Hacking (HCRL)](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) and
ROAD (Oak Ridge National Laboratory; Verma et al., 2024). The datasets are licensed and are **not**
included in this repository; place them under `data/raw/`.

## Repository layout

```text
configs/    dataset split, model and grid configuration
src/        data pipeline, models, training, quantization, evaluation, benchmarking
analysis/   aggregation, statistics, figures
tests/      unit and leakage tests
results/    raw run outputs (models are git-ignored), tables, figures
paper/      LaTeX sources
docs/       project plan, proposal, environment notes, phase logs
```

## Setup

Requirements: Python 3.11+ (developed with 3.13 on Windows 11), optionally an NVIDIA GPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# CUDA build of PyTorch (adjust the index URL to your CUDA driver, or omit it for a CPU build)
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pytest
```

On Linux/macOS activate the environment with `source .venv/bin/activate` instead.

## Reproducing results

<!-- To be filled in once the experiment pipeline is implemented. -->
