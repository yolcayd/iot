# Development environment

Detected on 2026-10-05 (phase P0).

| Component | Value |
|---|---|
| OS | Microsoft Windows 11 Pro, 10.0.26200, 64-bit |
| Python (system / venv) | 3.13.15 (MSC v.1944, AMD64) — the plan specifies 3.11; 3.13 is the only interpreter installed (see P0 log) |
| CPU | 12th Gen Intel Core i7-12650H — 10 cores (6P + 4E), 16 threads, base 2.3 GHz |
| RAM | 15.7 GB |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM (6141 MiB), compute capability 8.9 |
| NVIDIA driver | 610.74 (driver supports CUDA up to 13.3) |
| CUDA toolkit (`nvcc`) | Not installed — not required, PyTorch wheels ship their own CUDA runtime |
| PyTorch CUDA build | torch 2.14.1+cu130 (CUDA 13.0 runtime), `torch.cuda.is_available() == True` |
| ONNX Runtime | 1.30.0, CPU build (`CPUExecutionProvider`) — intended, latency is measured on CPU |
| Integrated GPU | Intel UHD Graphics |
| Docker | Docker Desktop 4.46.0 / Docker CLI 28.4.0 installed; engine **not running** at detection time (`//./pipe/docker_engine` not found) |
| Free disk | C: 761 GB free (169 GB used); G: 723 GB free |

## Notes for later phases

- **Edge-device emulation (F4):** no Raspberry Pi is available at P0. The plan's fallback is a Docker
  container limited to 2 CPU cores and 2 GB RAM (`docker run --cpus=2 --memory=2g ...`). Docker Desktop
  must be started (and its WSL 2 backend running) before that phase.
- **Latency benchmarks** must be run on CPU (the target is an edge ECU); the GPU is used for training only.
  Laptop power/thermal state influences latency: benchmark on AC power with the Windows power mode set to
  "Best performance" and record it in the run metadata.
- TensorFlow has no native-Windows GPU support since 2.10; it is used only for TFLite conversion (CPU).
