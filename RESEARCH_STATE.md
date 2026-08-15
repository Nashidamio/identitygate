# IdentityGate — Research State

Thesis: IdentityGate: Supervised, Identity-Verified Memory Write Admission for SAM 3.1 Video Tracking
Plan of record: v4 FINAL + review fixes F1–F6, N1, N3 adopted; N2 conditional on Week-1 audit.
Machine: Lab PC #27 · i7-14700K · RTX 4080 SUPER 16 GB · 64 GB RAM · 1 TB SSD
OS: Windows 11 host + WSL2 Ubuntu 22.04.5 (kernel 6.18.33.2-microsoft-standard-WSL2)
Driver / CUDA runtime seen from WSL: NVIDIA 591.86 / CUDA 13.1

## Current phase
Week 0 — machine bring-up.

## Current task
Task 1 (in progress): repo scaffolding + GitHub remote.

## Completed
- Task 0.1  WSL2 installed; Ubuntu 22.04.5 provisioned.
- Task 0.2  GPU visible inside WSL2 (nvidia-smi OK, RTX 4080 SUPER, 16376 MiB).
- Task 0.3  .wslconfig set to 56 GB / 16 GB swap / 20 procs; verified 54 Gi visible.
- Task 1a   Project tree + git init + empty scaffold files.
- Task 1b   GitHub remote linked (Nashidamio/identitygate, private).

## Verified components
- WSL2 kernel + GPU passthrough.
- Disk: / = 955 GB free (WSL vhdx on C:), /mnt/c = 597 GB free.
- Git remote push works.

## Environment (locked once decided)
- Python: not yet installed.
- PyTorch: not yet installed. Target: build matching a supported CUDA (12.x), NOT the raw 13.1 the driver advertises.
- Conda / venv: not yet decided.
- SAM 3.1 repo: not yet cloned.
- SAM3_Tracking_Zoo: not yet cloned.

## Datasets
- MOSEv2 (primary): NOT downloaded. ~100+ GB. Storage target TBD.
- DAVIS-17, LVOS v2, SA-V: not planned this week.
- P2 contaminated 200-video split + P2 600 cache videos: HARD EXCLUDE (assertion required in code).

## Checkpoints
- SAM 3.1: not downloaded. Requires HuggingFace access / token.
- SAM 3 fallback: not downloaded.

## Experiments completed
None.

## Experiments pending
See EXPERIMENT_REGISTRY.md.

## Important decisions
- 2026-08-15  OS choice: WSL2 Ubuntu 22.04 (Linux tooling for SAM3 repos).
- 2026-08-15  WSL memory pinned to 56 GB via .wslconfig.
- 2026-08-15  GitHub remote: Nashidamio/identitygate (private now, public at Week 7 release).
- OPEN  F1 (B2-vs-B3 as co-primary) — supervisor decision required before Week 6 freeze. Not blocking Week 0/1.
- OPEN  Storage target for MOSEv2 (~100 GB): / (WSL vhdx on C:) vs /mnt/d or /mnt/f. Decide before Week 2 dataset download.

## Problems / solutions
- 2026-08-15  Initial wsl --status reported virtualization disabled. Cause: Virtual Machine Platform enabled but reboot pending. Fix: reboot. Verified after reboot.
- 2026-08-15  First PAT accidentally pasted in chat by user. Fix: revoked immediately, second PAT created.
- 2026-08-15  Second PAT initially had no repository access / permissions. Fix: added Nashidamio/identitygate + Contents Read/Write; push succeeded.

## Current results
None (no experiments yet).

## Open questions
- SAM 3.1 repo state, Object Multiplex semantics, checkpoint availability — Claude's training data ends May 2026; all facts must be verified against the actual repo at audit time.
- Pointer/embedding per-object addressability under Object Multiplex — Week-1 kill-rule gate (F4).

## Next required action
Task 1c: install Miniconda; create thesis conda env; install PyTorch matching a supported CUDA build.
