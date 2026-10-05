# Phase 1R-S — Blinded model-validity salvage

## Verdict

**`MODEL_VALIDITY_NO_GO_FOR_LINEAGE_INFERENCE`**

The model did not satisfy the prespecified reliability prerequisites before lineage labels were examined. The 4→6 support failure persisted after the time axis, latent coordinates, support gate, and solver path were checked. Seed-to-seed dynamics remain inconsistent. This is a model-admission result; no lineage prediction was evaluated.

Lineage and clone/fate ground truth remained locked throughout Phase 1R-S. No pre-unblinding manifest was created, and lineage unblinding is not authorized.

## Freeze, provenance, and scope

Phase 1R-S was frozen before its numerical diagnostics. The five Phase 1R AE/RUOT checkpoints, inputs, model outputs, frozen classifier, taxonomy, solver-independent metrics, and the 5% support threshold were preserved. No model was retrained, no failed seed was removed, and no threshold or data subset was changed. No Phase 1R population QC was rerun because no Phase 1R implementation bug was found.

The Phase 1R-S protocol is in [`PHASE1RS_FROZEN_PROTOCOL.yaml`](../configs/PHASE1RS_FROZEN_PROTOCOL.yaml). The run manifest inherits the Phase 1R input, checkpoint, output, source, and environment hashes and adds hashes for the Phase 1R-S scripts, results, and reports: [`PHASE1RS_RUN_MANIFEST.tsv`](../manifests/PHASE1RS_RUN_MANIFEST.tsv). The diagnostics used Python 3.10.20, NumPy 1.26.4, SciPy 1.10.1, scikit-learn 1.2.2, pandas 1.5.3, and PyTorch 1.13.1; the environment record is [`phase1rs_diagnostic_environment.json`](../results/phase1RS/phase1rs_diagnostic_environment.json).

## Technical audits

### Time axis and latent coordinates

| Check | Result |
|---|---|
| Biological time mapping | Days 2/4/6 map to model times 0/1/2 |
| 2→4 and 4→6 intervals | Both are 2 biological days and 1 model-time unit |
| Frozen snapshot integration | Float32 Euler–Maruyama, dt=0.01, 100 left-endpoint steps per interval |
| Training/inference time mismatch | None found; no endpoint extrapolation |
| Latent scale mismatch | None found; each paired RUOT uses the same seed's two-dimensional `latent_ae` coordinates for training, inference, support, STA, and classifier input |
| ID/order mismatch | None found |

Detailed records: [`time_parameterization_audit.tsv`](../results/phase1RS/time_parameterization_audit.tsv) and [`latent_scaling_registry.tsv`](../results/phase1RS/latent_scaling_registry.tsv).

### Support gate

The implementation matches the frozen definition: endpoint nearest-target Euclidean distance compared with the q99 radius of the target cloud's leave-one-out 30th-neighbor distances. It uses the paired seed's target cloud and coordinates and does not include simulated points in its reference radius. The threshold remains 5%.

The implementation checks passed: target self-comparison produced 0% out of support, held-out cells from the same synthetic target distribution produced 0%, and distant synthetic points produced 100%. The gate also exactly reproduced the frozen Phase 1R rates for all five seeds and both intervals. See [`support_radius_validation.tsv`](../results/phase1RS/support_radius_validation.tsv) and the three support-gate unit tests in [`test_phase1rs_support_gate.py`](../tests/test_phase1rs_support_gate.py).

### Solver sensitivity and population metrics

The dt=0.01 full-SDE endpoints matched the saved Phase 1R endpoints exactly for all ten seed-by-interval comparisons; every saved solver endpoint was finite. The custom log-domain Sinkhorn implementation matched the frozen POT debiased Sinkhorn values exactly on the seed-11 2→4 model, identity, and random-null comparison. Solver changes used common Brownian paths; the original frozen dt=0.01 path was reproduced, then conditionally subdivided with Brownian bridges.

| Interval / arm | dt | Seeds within 5% support | Seeds better than both nulls (Sinkhorn, MMD) |
|---|---:|---:|---:|
| 2→4 full SDE | 0.01 | 5/5 | 5/5, 5/5 |
| 2→4 full SDE | 0.005 | 5/5 | 5/5, 5/5 |
| 2→4 full SDE | 0.0025 | 5/5 | 5/5, 5/5 |
| 4→6 full SDE | 0.01 | 0/5 | 5/5, 5/5 |
| 4→6 full SDE | 0.005 | 0/5 | 5/5, 5/5 |
| 4→6 full SDE | 0.0025 | 0/5 | 5/5, 5/5 |

At dt=0.0025, 4→6 out-of-support fractions were 18.94%, 10.57%, 12.12%, 15.54%, and 13.02% for seeds 11, 23, 47, 71, and 101. The maximum change from dt=0.01 was 0.06 percentage points. Endpoint changes decreased with each refinement: for the full SDE 4→6 arm, median-across-seed median displacement fell from 0.000746 at dt=0.005 to 0.000373 at dt=0.0025; the corresponding median p95 displacement fell from 0.00510 to 0.00256. Numerical refinement therefore converged without changing the support conclusion.

The d=0 drift-only diagnostic also failed the 4→6 support gate for all five seeds (12.57%–19.13% at dt=0.0025). Full SDE added a median 0.32 percentage points of support escape relative to drift-only at that step size. Diffusion contributes modestly for some seeds, but removing it does not restore target support.

Machine-readable outputs: [`solver_sensitivity.tsv`](../results/phase1RS/solver_sensitivity/solver_sensitivity.tsv), [`solver_sensitivity_summary.json`](../results/phase1RS/solver_sensitivity/solver_sensitivity_summary.json), and [`solver_output_integrity.tsv`](../results/phase1RS/solver_sensitivity/solver_output_integrity.tsv).

### Diffusion scale

For latent dimension 2 and Δt=1, the theoretical diffusion RMS displacement was 0.064–0.127. This equals 2.80–6.47 times the median empirical target 30NN radius, 0.42–0.74 times the q99 support radius, and 0.09–0.56 times the frozen model's median source-to-endpoint displacement. The learned noise scale is large relative to dense local target-manifold structure, although smaller than the q99 support boundary. Values by seed and interval are in [`diffusion_scale_audit.tsv`](../results/phase1RS/diffusion_scale_audit.tsv).

## Cross-seed stability and growth

The corrected neighborhood audit excluded self-neighbors and aligned canonical cell IDs. At k=30, median pairwise-distance Spearman correlation was 0.675 and median trustworthiness was 0.952, but neighborhood Jaccard and overlap fraction remained exactly zero for every non-reference seed. Mutual kNN edge Jaccard was 0.0083 and local-neighbor rank correlation was −0.739. At k=50, median neighborhood Jaccard was only 0.0101 and overlap was 2%. The zero k=30 result is not an indexing or self-neighbor artifact.

Against reference seed 11, the median cross-seed raw velocity cosine was 0.478 overall, 0.265 on day 2, 0.466 on day 4, and 0.880 on day 6. Overall velocity-norm rank correlation was 0.707, but day 2 fell to 0.579. Local averaging did not resolve early directional disagreement. The frozen seed-stability thresholds therefore fail on velocity direction and verified neighborhood overlap.

Growth ranks were also seed-dependent: median rank correlation was 0.349 overall and −0.007 on day 6; local 30NN growth correlation was −0.093 overall and −0.225 on day 6. Library ID explained a median 25.5% of growth variance across the five seeds (η² range 21.8%–49.0%); this is a descriptive technical-coupling signal. The expression-burden comparison uses the sum of frozen top-3000 log1p expression values, a proxy rather than raw UMI library size. Full results are in [`seed_stability_metrics.tsv`](../results/phase1RS/seed_stability_metrics.tsv), [`velocity_stability.tsv`](../results/phase1RS/velocity_stability.tsv), and [`growth_stability.tsv`](../results/phase1RS/growth_stability.tsv).

## Score term

All five final checkpoints have effectively zero score output: median norms range from 1.23×10⁻⁴¹ to 3.65×10⁻⁴⁰, and every audited row is below 10⁻⁸. Float32 score norms and the score-objective parameter gradient are zero after underflow. The corresponding pretraining checkpoints had score norms of 0.69–0.94 and nonzero objective gradients, indicating collapse during subsequent fitting rather than checkpoint loading or serialization. Restoring the fitted score component requires retraining or changing the fitted model, which Phase 1R-S prohibited. The frozen trajectory sampler itself uses `dx = v(x,t)dt + sqrt(2d)dW` and has no score drift; the collapse concerns the trained score component/objective. See [`SCORE_TERM_AUDIT.md`](SCORE_TERM_AUDIT.md) and [`score_term_audit.tsv`](../results/phase1RS/score_term_audit.tsv).

## STA and terminal-classifier reliability

The original frozen 10k STA runs converged for 3/5 seeds. Across the original runs plus two independent 10k replicates per seed, 11/15 runs converged; four seeds had at least one converged 10k run, while seed 101 failed in all three 10k runs. The single bounded confirmation run for each originally unstable seed met the frozen criterion: seed 71 at 20k and seed 101 at its 10k grid point within the 20k run. Thus all five seeds reached the criterion by the permitted ceiling, but seed 101 remained unstable at 10k across independent repeats.

Within-seed Monte Carlo variance in class probabilities was 2.48×10⁻⁷ pooled, compared with 0.0166 between seeds, a ratio of approximately 66,836. Replicate endpoint Gaussian MMD² estimates were near zero; the main STA reproducibility problem is model-seed variation rather than within-model Monte Carlo noise. The endpoint MMD is descriptive, computed from 2,000 sampled endpoints per replicate and starting day. Detailed outputs are in [`sta_replicate_status.tsv`](../results/phase1RS/sta_variance/sta_replicate_status.tsv) and [`sta_variance_decomposition.tsv`](../results/phase1RS/sta_variance/sta_variance_decomposition.tsv).

The terminal classifier remains frozen. Grouped OOF macro-F1 ranged from 0.478 to 0.578. Nineteen seed-by-class results fell below F1=0.30 across Eosinophil, Erythroid, Lymphoid, Mast, and Other/Unresolved. OOF calibration values and per-class precision/recall/F1 are in [`classifier_reliability.tsv`](../results/phase1RS/classifier_reliability.tsv); the pooled five-seed confusion matrix is [`classifier_pooled_oof_confusion.tsv`](../results/phase1RS/classifier_pooled_oof_confusion.tsv). No classifier or taxonomy replacement was made.

## Admission decision

The audit found no time-axis bug, latent-scale bug, support-gate defect, row-indexing problem, or solver divergence. Numerical refinement reproduced the frozen baseline and did not rescue 4→6 support. All five seeds continue to fail the frozen 4→6 support gate, satisfying no-go condition C1. Cross-seed velocity and neighborhood criteria also fail, and the score component has collapsed in all frozen checkpoints. Meeting the full gate requires a new fit or a scientific-model change; none was made in Phase 1R-S.

**Final status: `MODEL_VALIDITY_NO_GO_FOR_LINEAGE_INFERENCE`. Lineage unblinding: NO.** The conclusion is that the model did not satisfy the prespecified reliability prerequisites before lineage labels were examined; it does not claim that lineage predictions are wrong.
