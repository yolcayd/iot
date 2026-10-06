# Attack-Type-Aware Analysis of Model Quantization for In-Vehicle CAN Intrusion Detection

Course project for YZM5015 *Artificial Intelligence and the Internet of Things* (MSc).

Deep-learning intrusion detection systems (IDS) for the in-vehicle Controller Area Network (CAN) are
typically quantized (float32 → float16 → INT8) before being deployed on resource-constrained edge
devices. Prior work reports the cost of quantization only as a change in *overall* accuracy. CAN
attacks, however, are not equally hard to detect: *coarse* attacks visibly disturb the ID distribution
or message timing, while *stealthy* attacks (e.g. masquerade) keep both intact and only perturb payload
bytes. This project measures the detection loss caused by
quantization **per attack type**, together with inference latency, model size and memory usage.

## Research questions

- **RQ1 (main):** Does quantization (float32 → float16 → INT8) cause different detection losses for
  different attack types? Is the F1 loss on stealthy attacks significantly larger than on coarse attacks?
- **RQ2:** Under a given inference-latency budget, which model/quantization combination offers the best
  trade-off between detection performance, latency and model size?
- **RQ3:** Do supervised (1D-CNN) and unsupervised (LSTM autoencoder) detectors differ in their
  robustness to quantization?

## Data

Datasets: [Car-Hacking (HCRL)](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) and
[ROAD](https://zenodo.org/records/10462796) (Oak Ridge National Laboratory; Verma et al., 2024).
The datasets are **not** included in this repository. Sources, terms of use, citations, file formats
and SHA-256 checksums are documented in [`docs/DATA.md`](docs/DATA.md).

```powershell
python -m src.data.download all   # download, verify and extract into data/raw/ (resumable)
python -m src.data.loaders        # convert every capture to the common parquet schema in data/processed/
python -m analysis.eda            # exploratory analysis -> results/tables/, results/figures/
python -m src.data.features       # fit feature normalisation on training segments only
python -m src.data.splits         # window index + results/tables/split_summary.csv
```

All three commands are idempotent: finished downloads and converted captures are skipped on a re-run.

Every capture is stored as one parquet file with the common schema `timestamp, can_id, dlc,
data (uint8[8]), label, attack_type, source_file, source_capture`. ROAD `*_masquerade` logs are
derived from the matching fabrication logs by removing the conflicting legitimate frames, so they carry
the fabrication capture as their `source_capture`. Captures sharing a `source_capture` must always end
up in the same train/validation/test split.

The split is fixed in [`configs/data.yaml`](configs/data.yaml); nothing is drawn at random. ROAD is
split by `source_capture`. Car-Hacking has one file per attack type, so each file is split into
time-ordered train/validation/test blocks cut between attack bursts, with whole bursts discarded at
every cut. The four Car-Hacking attack files end with the same attack-free recording; it is used once,
as `car_hacking/shared_tail` (`src/data/sources.py`). Each message becomes a 10-dimensional vector
in [0, 1]: the ID as its rank in the vocabulary of IDs seen in benign training frames (unseen IDs share a
reserved code), the 8 payload bytes / 255, and the per-ID inter-arrival time on a log scale between a floor
and a cap fitted on benign training frames. The encoding is chosen so that per-tensor INT8 quantisation of
the input keeps every ID and byte value apart (`results/tables/feature_int8_levels.csv`). Windows are 100
messages long, with stride 50 in training and 100 (no overlap) in validation and test.
`tests/test_no_leakage.py` checks the split.

Attacks are grouped by mechanism in [`configs/attack_groups.yaml`](configs/attack_groups.yaml):
*coarse* attacks change the timing or the ID distribution of the bus (DoS, fuzzing, and also the
spoofing/fabrication attacks, which raise the target ID's rate); *stealthy* attacks keep both intact
and only alter the payload (the ROAD masquerade attacks). The grouping rule uses traffic statistics only
and was fixed before any metric was computed (see `configs/eda.yaml` and `docs/DECISIONS.md`).

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

Requirements: Python 3.13 (developed with 3.13.15 on Windows 11), optionally an NVIDIA GPU.

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

Baseline models (phase P3, work in progress). Hyperparameters are in `configs/models.yaml`, the run list in
`configs/grid.yaml`. Deep models and XGBoost train on the GPU; the CPU thread budget is
`runtime.threads` (default 4) or `--threads`.

```powershell
python -m src.train --model cnn1d --dataset road --seed 0   # one run -> results/raw/<run_id>/
python -m src.evaluate --run <run_id> --split test          # per-attack-type metrics of one run
python -m src.run_grid --stage baseline ae_feasibility      # all P3 runs (resumable) + tables
python -m src.run_grid --stage baseline ae_feasibility --status
python -m src.run_grid --ae-check                           # autoencoder feasibility protocol (validation only)
python -m src.sanity                                        # leakage suspicion checks for near-perfect F1
```

Every run writes `run_config.json` (hyperparameters, library versions, git commit, environment) and the
training curve. After training, the decision threshold of every model is calibrated to a false-positive
rate of 1% on clean validation normals (label 0, no masquerade capture, at least 5 s from any injected
frame; attack labels are not used). The primary per-attack-type
measure is the recall at that threshold; F1, ROC-AUC and average precision are secondary
(`docs/DECISIONS.md` K-035, K-037). The 1D-CNN stops early on validation average precision (K-034).

### Unsupervised model decision (P3)

The LSTM autoencoder (M2) failed the pre-registered feasibility check on ROAD (validation ROC-AUC 0.395 < 0.6;
Car-Hacking 0.807). The single pre-registered retry (hidden 128, lr 3e-4, up to 60 epochs) also failed on ROAD
(0.372). The 1D convolutional autoencoder (M2-alt) passed on both datasets (Car-Hacking 0.999, ROAD 0.722) and
is the unsupervised model on **both** datasets from F3 on. The decision used validation data only; test metrics
were computed after the selection (`docs/DECISIONS.md` K-030, K-033, K-039; `results/tables/ae_feasibility.csv`).
<!-- Unsupervised model decision (K-030) to be added after the baseline runs. -->

### Quantization and benchmark (P4)

Every trained deep model (1D-CNN, convolutional autoencoder) gets three versions; settings in
`configs/quantization.yaml`, decisions `docs/DECISIONS.md` K-042 ... K-047. Random forest and XGBoost are not
quantized; they enter the size and latency benchmark only.

| Level | Version | Accuracy measured with |
|---|---|---|
| Q0 | float32 ONNX (opset 17), output = the thresholded score (logit / reconstruction MSE) | ONNX Runtime, CPU |
| Q1 | float16 ONNX (`keep_io_types`) | real float16 arithmetic in PyTorch: the x86 CPU provider has no float16 kernel for any heavy layer of these models (0/5, 0/6), so Q1 is marked `cpu_unsupported` and gets no CPU latency (K-043) |
| Q2 | INT8 static PTQ, QDQ, asymmetric uint8 activations, symmetric per-channel int8 weights, MinMax calibration on 1000 windows of the model's own **training** selection | ONNX Runtime, CPU |

`quant_meta.json` records, per run:
- the layers that stay float (selective quantization);
- for the INT8 graph, the heavy layers that run an INT8 kernel versus those simulated as DequantizeLinear → float op
  → QuantizeLinear (the autoencoder's ConvTranspose);
- the float16 kernel coverage;
- the input quantizer (scale 1/255, zero point 0, 256 levels; every payload byte and ID code on its own level).

Each version's decision threshold is recalibrated with the validation-normal FPR 1% rule (primary). The float32
threshold is also applied unchanged to measure calibration drift (K-036, K-045).

```powershell
python -m src.quantize --pilot                       # Q0/Q1/Q2 + evaluation of the seed-0 pilot runs (~3 min)
python -m src.quantize --pilot-table                 # rebuild results/tables/quant_pilot.csv
python -m src.benchmark --profile host --session 1   # latency/size/memory, batch 1, 100 warm-up + 1000 timed
python -m src.benchmark --profile host --session 2
python -m src.benchmark --profile host --compare     # repeatability: median within 5 %, p95 within 10 %
.\docker\run_benchmark.ps1 -Session 1                # same in a container limited to 2 CPUs / 2 GB RAM
```

**Latency measurements:**
- Run them on an otherwise idle machine, on AC power.
- The `host` profile fixes the thread count at 4 but sets no CPU or memory quota, so it only approximates a
  constrained device.
- The `container` profile (2 threads) needs Docker Desktop to be running.
- The random forest predicts single-threaded (`n_jobs=1`): scikit-learn's threaded prediction races on the
  process-wide warning filters and then prints a warning per tree inside the timed loop (K-048). A measurement is
  rejected if any warning is shown during the timed inferences.
