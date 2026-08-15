# Environment (frozen 2026-08-15, post-SAM3-install)

## Host
- Machine: Lab PC #27
- CPU: Intel Core i7-14700K
- GPU: NVIDIA GeForce RTX 4080 SUPER (16376 MiB, compute capability 8.9, Ada Lovelace)
- RAM: 64 GB DDR5 (WSL2 cap: 56 GB, verified 54 Gi visible)
- Storage: 1 TB SSD

## OS
- Host: Windows 11
- WSL2 kernel: 6.18.33.2-microsoft-standard-WSL2
- Distro: Ubuntu 22.04.5 LTS

## NVIDIA
- Windows driver: 591.86
- CUDA advertised by nvidia-smi: 13.1
- CUDA used by PyTorch: 12.8

## Python stack
- Miniconda 26.5.3 at /home/user2/miniconda3
- Env name: identitygate, Python 3.12.13
- PyTorch: 2.10.0+cu128
- torchvision: 0.25.0+cu128
- Channel: conda-forge (strict priority)

## Second-tier packages NOT YET installed
Deferred until needed (opencv, pandas, hydra, scipy, scikit-learn, pyarrow, transformers).
Add them in a single controlled pip install when the corresponding Week task requires them.

## Reproducibility artifacts
- `requirements.lock.txt` — full pip freeze
- `environment.yml` — conda env spec (from-history)
- `docs/SAM3_INSTALL.md` — SAM 3 clone commit + required upstream-dep pins
