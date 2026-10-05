#!/usr/bin/env python3
"""Lineage-blind five-seed stability checks on matched canonical cell IDs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "results/phase1R/REIMPLEMENTED_ID_PRESERVING_ARM"
SEEDS = [11, 23, 47, 71, 101]
PAIR_SEED = 20260927


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)).statistic)


def icc_2_1(values: np.ndarray) -> float:
    """Two-way random, absolute agreement, single-measure ICC(2,1)."""
    x = np.asarray(values, dtype=np.float64)
    n, k = x.shape
    grand = x.mean()
    ms_subject = k * np.square(x.mean(axis=1) - grand).sum() / (n - 1)
    ms_seed = n * np.square(x.mean(axis=0) - grand).sum() / (k - 1)
    residual = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_error = np.square(residual).sum() / ((n - 1) * (k - 1))
    denominator = ms_subject + (k - 1) * ms_error + k * (ms_seed - ms_error) / n
    return float((ms_subject - ms_error) / denominator) if denominator else float("nan")


def main() -> None:
    data = {}
    for seed in SEEDS:
        path = ARM / f"seed_{seed}" / "dynamics_outputs.npz"
        if not path.exists():
            raise FileNotFoundError(f"Seed {seed} dynamics output is missing")
        with np.load(path, allow_pickle=False) as archive:
            data[seed] = {key: archive[key] for key in ["original_cell_id", "source_row_index", "latent_ae", "velocity", "growth", "score", "diffusion_shared_d"]}
    reference = data[11]
    ref_ids = reference["original_cell_id"].astype(str)
    n = len(ref_ids)
    rng = np.random.default_rng(PAIR_SEED)
    pair_i = rng.integers(0, n, size=25000)
    pair_j = rng.integers(0, n, size=25000)
    while np.any(pair_i == pair_j):
        mask = pair_i == pair_j
        pair_j[mask] = rng.integers(0, n, size=int(mask.sum()))
    reference_latent = reference["latent_ae"].astype(np.float64)
    tree_ref = cKDTree(reference_latent)
    _, ref_nn = tree_ref.query(reference_latent, k=31, workers=-1)
    metrics = []
    aligned_d = {11: float(reference["diffusion_shared_d"])}
    scale_by_seed = {11: 1.0}

    for seed in SEEDS:
        d = data[seed]
        ids = d["original_cell_id"].astype(str)
        rows = d["source_row_index"].astype(np.int64)
        if not np.array_equal(ids, ref_ids) or not np.array_equal(rows, np.arange(n)):
            raise ValueError(f"Seed {seed}: saved outputs do not align to canonical IDs")
        x = d["latent_ae"].astype(np.float64)
        velocity = d["velocity"].astype(np.float64)
        growth = d["growth"].astype(np.float64)
        target = reference_latent
        mean_x, mean_y = x.mean(axis=0), target.mean(axis=0)
        xc, yc = x - mean_x, target - mean_y
        u, singular, vt = np.linalg.svd(xc.T @ yc)
        rotation = u @ vt
        aligned_scale = float(singular.sum() / np.square(xc).sum())
        aligned = xc @ rotation * aligned_scale + mean_y
        aligned_velocity = velocity @ rotation * aligned_scale
        d_aligned = float(d["diffusion_shared_d"]) * aligned_scale**2
        aligned_d[seed] = d_aligned
        scale_by_seed[seed] = aligned_scale

        distances = np.linalg.norm(x[pair_i] - x[pair_j], axis=1)
        ref_distances = np.linalg.norm(reference_latent[pair_i] - reference_latent[pair_j], axis=1)
        tree = cKDTree(x)
        _, neighbours = tree.query(x, k=31, workers=-1)
        knn_overlap = np.empty(n, dtype=np.float64)
        for i, row in enumerate(neighbours):
            current_set = set(row[row != i][:30].tolist())
            reference_set = set(ref_nn[i][ref_nn[i] != i][:30].tolist())
            knn_overlap[i] = len(current_set.intersection(reference_set)) / 30
        cos = np.sum(aligned_velocity * reference["velocity"], axis=1) / (
            np.linalg.norm(aligned_velocity, axis=1) * np.linalg.norm(reference["velocity"], axis=1) + 1e-12
        )
        vnorm = np.linalg.norm(aligned_velocity, axis=1)
        ref_vnorm = np.linalg.norm(reference["velocity"], axis=1)
        row = {
            "seed": seed,
            "latent_pairwise_distance_spearman": spearman(distances, ref_distances),
            "latent_knn_overlap_k30_median": float(np.median(knn_overlap)),
            "latent_procrustes_rmse": float(np.sqrt(np.mean(np.square(aligned - target)))),
            "velocity_cosine_median": float(np.median(cos)),
            "velocity_norm_spearman": spearman(vnorm, ref_vnorm),
            "growth_spearman": spearman(growth, reference["growth"]),
            "diffusion_shared_d_native": float(d["diffusion_shared_d"]),
            "diffusion_shared_d_reference_scale": d_aligned,
            "procrustes_scale_to_reference": aligned_scale,
        }
        metrics.append(row)

    growth_matrix = np.column_stack([data[s]["growth"].astype(np.float64) for s in SEEDS])
    for row in metrics:
        row["growth_icc_2_1_all_seeds"] = icc_2_1(growth_matrix)
    diffusion_values = np.asarray([aligned_d[s] for s in SEEDS], dtype=np.float64)
    diffusion_cv = float(diffusion_values.std(ddof=1) / abs(diffusion_values.mean())) if diffusion_values.mean() != 0 else float("inf")
    result = {
        "status": "COMPLETE_LINEAGE_BLIND",
        "reference_seed": 11,
        "n_cells": n,
        "matched_ids": int(n),
        "lineage_information_opened": False,
        "pairwise_distance_pairs": len(pair_i),
        "pairwise_distance_sampling_seed": PAIR_SEED,
        "diffusion_reference_scale_values": {str(k): float(v) for k, v in aligned_d.items()},
        "diffusion_coefficient_of_variation_reference_scale": diffusion_cv,
        "metrics": metrics,
    }
    out = ROOT / "results/phase1R/five_seed_stability.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(metrics).to_csv(ROOT / "results/phase1R/five_seed_stability.tsv", sep="\t", index=False)
    result["summary_path"] = str(out.relative_to(ROOT))
    result["summary_sha256"] = sha256(out)
    print(json.dumps({"status": result["status"], "diffusion_cv": diffusion_cv, "summary_sha256": result["summary_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
