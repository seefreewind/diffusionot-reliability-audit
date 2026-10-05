#!/usr/bin/env python3
"""Audit canonical-row, AE-row, RUOT-row, and saved-output cell identity links.

This script reads only the model-visible expression metadata and per-seed
latent inputs/outputs. It never reads clone assignments, lineage barcodes, or
fate tables.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "results/phase1R/REIMPLEMENTED_ID_PRESERVING_ARM"
SEEDS = [11, 23, 47, 71, 101]
TIMES = [2, 4, 6]
OUTPUT_NAME = "dynamics_outputs.npz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run() -> dict:
    index_path = ARM / "cell_index.tsv"
    index = pd.read_csv(index_path, sep="\t")
    ids = index["original_cell_id"].astype(str).to_numpy()
    source_rows = index["source_row_index"].to_numpy(dtype=np.int64)
    times = index["timepoint"].to_numpy(dtype=np.int64)
    if len(ids) != 130887 or len(np.unique(ids)) != 130887:
        raise ValueError("Canonical source ID vector does not contain 130,887 unique IDs")
    if not np.array_equal(source_rows, np.arange(len(ids))):
        raise ValueError("Source row index is not the canonical 0..N-1 order")
    if sorted(np.unique(times).tolist()) != TIMES:
        raise ValueError(f"Unexpected timepoints in cell index: {np.unique(times).tolist()}")

    ruot_order = np.concatenate([np.flatnonzero(times == time) for time in TIMES])
    if len(ruot_order) != len(ids) or len(np.unique(ruot_order)) != len(ids):
        raise ValueError("RUOT time-group concatenation is not a permutation of canonical rows")
    ruot_row_by_tensor = np.empty(len(ids), dtype=np.int64)
    ruot_row_by_tensor[ruot_order] = np.arange(len(ids), dtype=np.int64)

    seed_summaries = {}
    all_outputs_present = True
    for seed in SEEDS:
        seed_dir = ARM / f"seed_{seed}"
        latent_path = seed_dir / "latent_input.npz"
        with np.load(latent_path, allow_pickle=False) as archive:
            latent_ids = archive["original_cell_id"].astype(str)
            latent_source_rows = archive["source_row_index"].astype(np.int64)
            latent_times = archive["time_label"].astype(str).astype(np.int64)
            latent = archive["latent_ae"]
        if not np.array_equal(latent_ids, ids):
            raise ValueError(f"Seed {seed}: AE latent cell IDs differ from canonical input order")
        if not np.array_equal(latent_source_rows, source_rows):
            raise ValueError(f"Seed {seed}: AE source row indices differ from canonical input order")
        if not np.array_equal(latent_times, times):
            raise ValueError(f"Seed {seed}: AE time labels differ from canonical input order")
        if latent.shape != (len(ids), 2):
            raise ValueError(f"Seed {seed}: unexpected latent shape {latent.shape}")

        identity = pd.DataFrame({
            "tensor_row": source_rows,
            "canonical_cell_id": ids,
            "timepoint": times,
            "AE_row": np.arange(len(ids), dtype=np.int64),
            "RUOT_row": ruot_row_by_tensor,
            "saved_output_row": source_rows,
        })
        out_path = ROOT / "results/phase1R" / f"seed{seed}_cell_identity.tsv"
        identity.to_csv(out_path, sep="\t", index=False)
        summary = {
            "seed": seed,
            "n_cells": len(ids),
            "unique_cell_ids": int(identity["canonical_cell_id"].nunique()),
            "duplicate_ids": int(identity["canonical_cell_id"].duplicated().sum()),
            "missing_ids": int(identity["canonical_cell_id"].isna().sum()),
            "ae_order_matches_canonical": True,
            "ruot_order_is_documented_time_concatenation": True,
            "identity_tsv": str(out_path.relative_to(ROOT)),
            "identity_tsv_sha256": sha256(out_path),
            "latent_input_sha256": sha256(latent_path),
        }

        output_path = seed_dir / OUTPUT_NAME
        if output_path.exists():
            with np.load(output_path, allow_pickle=False) as archive:
                output_ids = archive["original_cell_id"].astype(str)
                output_rows = archive["source_row_index"].astype(np.int64)
                output_times = archive["timepoint"].astype(np.int64)
                output_latent = archive["latent_ae"]
                velocity = archive["velocity"]
                growth = archive["growth"]
                score = archive["score"]
                diffusion = archive["diffusion_shared_d"]
            summary["saved_output_exists"] = True
            summary["saved_output_ids_match_canonical"] = bool(np.array_equal(output_ids, ids))
            summary["saved_output_rows_match_canonical"] = bool(np.array_equal(output_rows, source_rows))
            summary["saved_output_timepoint_matches_canonical"] = bool(np.array_equal(output_times, times))
            summary["saved_output_latent_matches_ae"] = bool(np.array_equal(output_latent, latent))
            summary["saved_dynamics_shapes_valid"] = bool(
                velocity.shape == (len(ids), 2) and growth.shape == (len(ids),)
                and score.shape == (len(ids), 2) and diffusion.shape == ()
            )
            summary["saved_dynamics_finite"] = bool(
                np.isfinite(output_latent).all() and np.isfinite(velocity).all()
                and np.isfinite(growth).all() and np.isfinite(score).all()
                and np.isfinite(diffusion).all()
            )
            summary["saved_output_sha256"] = sha256(output_path)
            if not (
                summary["saved_output_ids_match_canonical"]
                and summary["saved_output_rows_match_canonical"]
                and summary["saved_output_timepoint_matches_canonical"]
                and summary["saved_output_latent_matches_ae"]
                and summary["saved_dynamics_shapes_valid"]
                and summary["saved_dynamics_finite"]
            ):
                raise ValueError(f"Seed {seed}: saved dynamics output row identity mismatch")
        else:
            summary["saved_output_exists"] = False
            all_outputs_present = False
        seed_summaries[str(seed)] = summary

    result = {
        "n_cells": len(ids),
        "n_unique_canonical_ids": int(len(np.unique(ids))),
        "timepoint_counts": {str(t): int(np.count_nonzero(times == t)) for t in TIMES},
        "ruot_tensor_order": "time 2, then 4, then 6; source order retained within each timepoint",
        "lineage_information_opened": False,
        "all_five_seed_outputs_present": all_outputs_present,
        "seeds": seed_summaries,
    }
    summary_path = ROOT / "results/phase1R/cell_id_input_audit.json"
    identity_pass = all_outputs_present and all(
        s["duplicate_ids"] == 0 and s["missing_ids"] == 0
        and s.get("saved_output_ids_match_canonical")
        and s.get("saved_output_rows_match_canonical")
        and s.get("saved_output_timepoint_matches_canonical")
        and s.get("saved_output_latent_matches_ae")
        and s.get("saved_dynamics_shapes_valid")
        and s.get("saved_dynamics_finite")
        for s in seed_summaries.values()
    )
    result["status"] = "CELL_ID_INTEGRITY_PASS" if identity_pass else ("HOLD_ID_INTEGRITY" if all_outputs_present else "INPUT_AE_RUOT_INPUT_ORDER_PASS_OUTPUTS_PENDING")
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    if identity_pass:
        marker = ROOT / "results/phase1R/CELL_ID_INTEGRITY_PASS"
        marker.write_text(json.dumps({
            "status": "CELL_ID_INTEGRITY_PASS",
            "n_cells_per_seed": 130887,
            "unique_ids_per_seed": 130887,
            "duplicates_per_seed": 0,
            "missing_ids_per_seed": 0,
            "untracked_reorders": 0,
            "evidence": str(summary_path.relative_to(ROOT)),
            "evidence_sha256": sha256(summary_path),
            "lineage_information_opened": False,
        }, indent=2) + "\n")
    else:
        stale_marker = ROOT / "results/phase1R/CELL_ID_INTEGRITY_PASS"
        if stale_marker.exists():
            stale_marker.unlink()
    print(json.dumps({"status": result["status"], "n_cells": len(ids), "all_five_seed_outputs_present": all_outputs_present}, indent=2))
    return result


if __name__ == "__main__":
    run()
