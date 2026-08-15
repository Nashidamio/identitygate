# Environment (frozen 2026-08-15)

## Host
- Machine: Lab PC #27
- CPU: Intel Core i7-14700K
- GPU: NVIDIA GeForce RTX 4080 SUPER (16376 MiB, compute capability 8.9, Ada Lovelace)
- RAM: 64 GB DDR5 (WSL2 cap: 56 GB, verified 54 Gi visible)
- Storage: 1 TB SSD, C: has 597 GB free

## OS
- Host: Windows 11 (build 26100)
- WSL2 kernel: 6.18.33.2-microsoft-standard-WSL2
- Distro: Ubuntu 22.04.5 LTS (jammy)

## NVIDIA
- Windows driver: 591.86
- CUDA runtime advertised by driver (nvidia-smi): 13.1
- CUDA that PyTorch actually uses: 12.4 (backward-compatible; do NOT chase 13.1)

## Python
- Manager: Miniconda 26.5.3 at /home/user2/miniconda3
- Env name: identitygate
- Python: 3.11.15
- Primary channel: conda-forge (strict priority)

## Key packages (see requirements.lock.txt for full pin)
- torch: 2.5.1+cu124
- torchvision: 0.20.1+cu124
- numpy: 1.26.4  (pinned <2.0 for OpenCV/older-SAM compatibility)
- opencv-python-headless: 4.10.0.84
- transformers: 4.46.2
- huggingface_hub: 0.26.2
- hydra-core: 1.3.2
- pandas: 2.2.3
- pyarrow: 17.0.0
- scikit-learn: 1.5.2

## Verification passed
- nvidia-smi visible from WSL2.
- torch.cuda.is_available() == True.
- 4096x4096 matmul on cuda:0 returns finite mean; ~200 MB VRAM used.

## Reproduction
```bash
conda create -n identitygate python=3.11 -c conda-forge -y
conda activate identitygate
pip install -r requirements.lock.txt
```
