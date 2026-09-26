# Checkpoint 20 - Figure and Table QA

Status: TECHNICAL PRE-PDF QA

## Conceptual figures

- `fig01_controlled_memory_write_admission.png` - 1918x820 px
- `fig02_memory_management_related_work_landscape.png` - 1774x887 px
- `fig03_controlled_memory_write_evaluation_pipeline.png` - 1672x941 px
- `fig04_signal_composition_and_write_decision_architecture.png` - 1774x887 px
- `fig05_leakage_controlled_cohort_construction.png` - 1448x1086 px

Known content blockers:
- Figure 2 requires DAMSAM -> DAM4SAM and removal of remaining SIGMA prominence.
- Figure 3 must show the TEST comparison rule `|r_A-r_B| <= 0.02`, not `|r_test-r*|`.

## Quantitative plots

- `chapter5_hard_test_bca_effects.svg` - 4416 bytes - PASS
- `chapter5_hard_test_write_rates.svg` - 3045 bytes - PASS
- `chapter5_por30_hard_vs_representative.svg` - 4825 bytes - PASS
- `chapter5_por30_vs_write_rate.svg` - 7034 bytes - PASS

## Tables

- `chapter5_headline_results.csv` - 12 data rows
- `chapter5_key_inference.csv` - 13 data rows
- `chapter5_rate_status.csv` - 16 data rows
- `chapter5_statistical_comparisons.csv` - 13 data rows

CHECKPOINT20_STATUS=PARTIAL_FIG2_FIG3_AND_FINAL_PDF_QA
