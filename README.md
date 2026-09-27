# IdentityGate

Controlled study of supervised memory-write admission for frozen SAM 3 VOS/PVS video object tracking.

## Scientific status

The scientific and experimental execution phase is complete through EXP054.

The final system ladder includes B0, B1, B2, B3-S, B3-R, and B5. The project includes physically verified closed-loop memory-write control, leakage-controlled cohorts, frozen POR@30 and ITR@30 evaluation, matched-write-rate analysis, video-clustered BCa inference, and a one-touch final TEST campaign.

No additional TEST run, TEST threshold retuning, confirmatory retraining, endpoint change, or outcome-driven subgroup search is permitted after scientific closure.

## Locked research question

> What information should a memory-write gate use - quality signals, temporal signals, or identity signals?

The contribution is a controlled study of memory-write admission rather than SAM fine-tuning.

## Core substrate

Core experiments use:

- frozen SAM 3
- VOS/PVS tracking
- `build_sam3_video_model()`
- `facebook/sam3/sam3.pt`
- SAM source commit `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`
- no SAM gradient updates, LoRA, detector replacement, or external re-identification model

True SAM 3.1 Object Multiplex was retained only as transfer and feasibility context because the measured path exceeded the 16 GB experimental GPU budget. It does not carry the core statistical claim.

## Final primary interpretation boundary

For the frozen HARD_TEST80 B3-S versus B2 POR@30 contrast, the frozen-threshold estimate is `+0.007905` with video-clustered BCa 95% CI `[-0.002037, +0.020882]`.

The pooled write-rate difference is `0.032703`, which exceeds the frozen matched-rate tolerance of `0.02`.

Therefore the final primary status is:

`RATE_MISMATCH_NO_MATCHED_RATE_PRIMARY_INTERPRETATION`

No TEST threshold retuning or TEST rerun is used to repair this mismatch.

## Repository map

- `PREREGISTRATION.md` - frozen confirmatory protocol and interpretation rules
- `EXPERIMENT_REGISTRY.md` - experiment-by-experiment provenance
- `RESEARCH_STATE.md` - chronological research state and amendments
- `configs/` - frozen experiment and protocol configurations
- `scripts/` - experiment, evaluation, and reporting code
- `experiments/` - retained scientific outputs and summaries
- `docs/` - environment, installation, audit, and signal-schema documentation
- `thesis_assets/` - final report tables, figures, evidence packs, and derived reporting artifacts
- `thesis_assets/drafts/SCIENTIFIC_EXECUTION_CLOSURE.md` - scientific execution closure record
- `requirements.lock.txt` and `environment.yml` - realized environment records

Historical audit documents are preserved for provenance and may contain superseded terminology. Current locked amendments and final experiment records take precedence.

## Data and model weights

MOSEv2 data are not vendored in this repository. Obtain MOSEv2 from the official upstream distribution and follow its upstream license and usage conditions.

MOSEv2 is distributed by its maintainers under CC BY-NC-SA 4.0 for non-commercial research use.

SAM checkpoints are also not vendored in this repository. Core experiments use the upstream `facebook/sam3/sam3.pt` checkpoint.

The repository license applies only to original IdentityGate material for which the authors hold the relevant rights. It does not replace or override licenses for third-party datasets, model weights, software, or third-party-derived media.

## Reproducibility

Start with:

1. `docs/ENVIRONMENT.md`
2. `docs/SAM3_INSTALL.md`
3. `environment.yml`
4. `requirements.lock.txt`
5. `PREREGISTRATION.md`
6. `EXPERIMENT_REGISTRY.md`

The Git commit, configuration, command, output location, and interpretation are recorded for scientific experiments in the registry and research-state record.

## License

See `LICENSE`.

## Citation

See `CITATION.cff`.
