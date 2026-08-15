# IdentityGate — Experiment Registry

Every meaningful experimental run appended here. One entry per run. Never delete; append corrections as new dated notes.

Fields per entry:
- Experiment ID
- Purpose
- Hypothesis
- Code version (git commit / tag)
- Dataset (name + version + split)
- Checkpoint (model + version)
- Configuration file
- Seed
- Command
- Output location
- Result (raw metrics)
- Status (PLANNED / RUNNING / COMPLETED / FAILED / SUPERSEDED)
- Interpretation (what the evidence supports, honestly)

---

## EXP000 — Environment sanity (bookkeeping only)
- Purpose: Placeholder to verify registry format.
- Status: PLANNED (no run).
- Notes: First real experiment will be a single-video SAM 3.1 smoke test after Task 1 (env + checkpoints) completes.
