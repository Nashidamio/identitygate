# IdentityGate Environment

Status: FINAL PUBLIC-RELEASE RECORD

## Reference machine

- Host OS: Windows 11
- Execution environment: WSL2 Ubuntu 22.04.5
- CPU: Intel Core i7-14700K
- GPU: NVIDIA GeForce RTX 4080 SUPER, 16 GB
- System RAM: 64 GB
- Conda environment: `identitygate`

The reference hardware documents the environment used for the thesis. Equivalent hardware may be used where memory requirements are satisfied.

## Core software stack

- Python: 3.12.13
- PyTorch: 2.10.0+cu128
- torchvision: 0.25.0+cu128
- CUDA used by PyTorch: 12.8
- SAM source commit: `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`
- Core SAM checkpoint: `facebook/sam3/sam3.pt`

Required SAM-related compatibility pins and installation notes are recorded in `docs/SAM3_INSTALL.md`.

## Reproducibility artifacts

- `environment.yml` records the Conda environment specification.
- `requirements.lock.txt` records the realized Python package environment.
- `docs/SAM3_INSTALL.md` records the SAM source pin and installation-specific dependency fixes.
- `PREREGISTRATION.md` records the frozen confirmatory protocol.
- `EXPERIMENT_REGISTRY.md` records experiment commands and result provenance.

`requirements.lock.txt` is an audit snapshot of the realized environment. Environment-specific build paths can appear in package freezes, so it should be interpreted together with `environment.yml` and `docs/SAM3_INSTALL.md` rather than treated as the only installation source.

## Numerical identity-similarity rule

Identity cosine calculations use:

- FP32
- autocast disabled
- CUDA TF32 disabled
- `torch.set_float32_matmul_precision("highest")`

These settings are part of the scientific protocol and are not optional performance tuning.

## Data and weights

MOSEv2 and SAM model weights are external dependencies and are not bundled into this repository.

The experiment configuration and provenance records identify the dataset partition, checkpoint, source commit, and relevant hashes used by the thesis.
