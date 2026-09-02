# CACIE protocol

This is the stable entry point for the PACO v1.1 CACIE protocol. The complete
normative specification is maintained in
[`CACIE_VALIDATION_PROTOCOL.md`](CACIE_VALIDATION_PROTOCOL.md); do not create a
second, divergent set of method parameters here.

The frozen paper-facing path is:

1. fail-closed track and source-frame provenance audit, including explicit
   exclusion of every point-count-detected partial scan;
2. a 200-frame main validation using 9 x 9, 17 x 17, and 33 x 33 projected
   ray--OBB grids, with 33 x 33 as the within-dynamic-OBB reference;
3. post-development calibration of the sparse rule, added after the any-ray
   diagnostic was found to be optimistic, on 100 time-by-density-stratified
   frames and one evaluation on 100 untouched held-out frames, yielding at
   least 3 of 15 visible rays as the subsequently frozen low-cost rule;
4. a separate 25-frame-subset 17 x 17/33 x 33/65 x 65 convergence analysis
   (the subset is drawn from the 200 main validation frames, not independently);
5. `scripts/compute_full_record_observability.py` for the full-record stream;
6. ratio-of-sums inference with circular moving blocks of 1,200 consecutive
   retained-frame observations (nominally 120 seconds at 10 Hz; source-index
   gaps may extend elapsed time) via `scripts/bootstrap_observability.py`;
7. `scripts/rank_residual_demand.py` for current hotspot ranking;
8. declared block-duration, vertical-coordinate, decimation, and partial-scan
   sensitivity summaries;
9. source-audited validation and full-results figures; and
10. a final immutable executed-notebook audit plus explicit release manifest.

`scripts/run_canonical.sh` implements this order. A disabled `RUN_*` analytical
stage means reuse of a checked artifact; it does not permit a missing dependency
to be silently skipped.

The tracked notebook is source only. The runner writes
`outputs/canonical/notebook/cacie_validation.executed.ipynb` and a separate
`notebook_audit_receipt.json`. Reuse re-hashes the source notebook, executed
notebook, and every audited artifact and requires all stages to be `VERIFIED`
with exactly one `CACIE_NOTEBOOK_AUDIT_PASS_V1` output. Release-manifest mode
also requires clean Git and includes the producing source, scripts, lockfile,
documentation, runtime receipts, and internal detector-validation evidence.
Stage-local source-lineage receipts bind reused validation bytes and every
cheap strata/figure derivative to their exact inputs; reuse verifies these
receipts and never refreshes them.

`scripts/optimize_external_coverage.py` is an illustrative future extension,
not a canonical CACIE result or a pole-siting recommendation. Analytic ray--OBB
visibility is a computational reference within the detected dynamic-box
abstraction, not physical visibility ground truth.
