# Stage 11 Validation Report

Status: PASS

This report records identity, completeness, and deterministic-export gates only. It does not state a paper result conclusion.

## Identity gates

- Stage 10 baseline ZIP SHA256: `fb3666b9a9cd74d2a52978df98b2e4f08b7ddcb11bff0134bbdaed6cc09fb389`
- Stage 10 baseline archive members: 5401
- Stage 10 baseline tree SHA256: `0b9cd18dda5c8bb3dcc8e5ca4a752329bd0fd8599145ba70960d448aa3868f3d`
- Stage 11 science contract SHA256: `9fc2141695a6e73697bbee2e7590eb54678be5ca094ae06e753a36a38f3282a1`
- All twelve frozen source SHA256 gates: PASS
- Stage 1-10 tree boundary: PASS

## Evidence row-count gates

- `paper_data/F_only_individual.csv`: 80
- `paper_data/F_only_paired.csv`: 32
- `paper_data/H300_width_strata_summary.csv`: 15
- `paper_data/HPRF_individual.csv`: 100
- `paper_data/HPRF_paired.csv`: 40
- `paper_data/NI_consistency.csv`: 5
- `paper_data/composite_individual.csv`: 180
- `paper_data/composite_paired.csv`: 72
- `paper_precision_audit_master.csv`: 216
- `paper_results_individual_master.csv`: 360
- `paper_results_paired_master.csv`: 144
- `paper_statistical_precision_limitations.csv`: 6
- `registry/condition_registry_export.csv`: 19
- `registry/contrast_registry_export.csv`: 2
- `registry/method_registry_export.csv`: 5
- `registry/metric_registry_export.csv`: 4
- `paper_statistical_precision_limitations.csv`: 6
- `paper_evidence_registry.csv`: 16

## Prohibited-execution gates

- New Bootstrap, RNG, CI, p-value, contrast, physical simulation, and Formal rerun: absent
- Condition dropping, parameter change, and result-driven selection: absent
- Paper figure rendering and paper result conclusion writing: absent
- Stage 12: absent

STOP after Stage 11.
