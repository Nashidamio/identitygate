# EXP014 synchronized-reappearance visual audit

Purpose: visually inspect GT-timing synchronized reappearance clusters before any dataset exclusion or reclassification.

Important: these clusters are candidates only. No cluster is classified as whole-scene occlusion by this script.

Event-level criterion: at least 5 reappearance events within +/-10 frames.
Event-level synchronized events: 78.
Candidate clusters: 5.

Each cluster directory contains:
- events.csv: qualifying event rows
- timeline.csv: per-frame GT visibility counts
- keyframes_raw.jpg: raw visual evidence
- keyframes_gt_overlay.jpg: same frames with GT masks overlaid

Classification is intentionally left OPEN pending human inspection.
