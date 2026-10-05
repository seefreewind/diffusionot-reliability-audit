#!/usr/bin/env python3
"""Lineage-blind snapshot reconstruction checks for the five RUOT seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "results/phase1R/REIMPLEMENTED_ID_PRESERVING_ARM"
SEEDS = [11, 23, 47, 71, 101]
N_SAMPLE = 10000
N_METRIC = 2000
DT = 0.01
TIME_LABELS = [(2, 4), (4, 6)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stratified_sample(index: pd.DataFrame, day: int, n: int, seed: int) -> np.ndarray:
    pool = index.index[index.timepoint.eq(day)].to_numpy(dtype=np.int64)
    if len(pool) < n:
        n = len(pool)
    subset = index.loc[pool]
    strata = subset["library"].astype(str) + "\x1f" + subset["cell_type"].astype(str)
    groups = {key: np.asarray(rows, dtype=np.int64) for key, rows in subset.groupby(strata, sort=True).groups.items()}
    sizes = {key: len(rows) for key, rows in groups.items()}
    raw = {key: n * size / sum(sizes.values()) for key, size in sizes.items()}
    allocation = {key: min(sizes[key], int(np.floor(raw[key]))) for key in groups}
    remaining = n - sum(allocation.values())
    ranked = sorted(groups, key=lambda key: (-(raw[key] - np.floor(raw[key])), key))
    for key in ranked:
        if remaining == 0:
            break
        if allocation[key] < sizes[key]:
            allocation[key] += 1
            remaining -= 1
    rng = np.random.default_rng(seed)
    chosen = []
    for key in sorted(groups):
        k = allocation[key]
        if k:
            chosen.extend(rng.choice(groups[key], size=k, replace=False).tolist())
    chosen = np.asarray(sorted(chosen), dtype=np.int64)
    if len(chosen) != n or len(np.unique(chosen)) != n:
        raise RuntimeError(f"Day {day}: stratified sampler returned {len(chosen)} rows, expected {n}")
    return chosen


def sinkhorn_squared(x: np.ndarray, y: np.ndarray, reg: float, n_iter: int = 500) -> float:
    pot_site = ROOT / ".venv_phase1r_analysis/lib/python3.10/site-packages"
    if pot_site.is_dir() and str(pot_site) not in sys.path:
        sys.path.append(str(pot_site))
    import ot
    a = np.full(len(x), 1 / len(x), dtype=np.float64)
    b = np.full(len(y), 1 / len(y), dtype=np.float64)
    cxy = cdist(x, y, metric="sqeuclidean")
    cxx = cdist(x, x, metric="sqeuclidean")
    cyy = cdist(y, y, metric="sqeuclidean")
    xy = float(ot.sinkhorn2(a, b, cxy, reg, numItermax=n_iter, method="sinkhorn_log"))
    xx = float(ot.sinkhorn2(a, a, cxx, reg, numItermax=n_iter, method="sinkhorn_log"))
    yy = float(ot.sinkhorn2(b, b, cyy, reg, numItermax=n_iter, method="sinkhorn_log"))
    return xy - 0.5 * xx - 0.5 * yy


def gaussian_mmd2(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    scale = 2.0 * bandwidth * bandwidth
    kxx = np.exp(-(cdist(x, x, metric="sqeuclidean")) / scale)
    kyy = np.exp(-(cdist(y, y, metric="sqeuclidean")) / scale)
    kxy = np.exp(-(cdist(x, y, metric="sqeuclidean")) / scale)
    nx, ny = len(x), len(y)
    term_x = (kxx.sum() - np.trace(kxx)) / (nx * (nx - 1))
    term_y = (kyy.sum() - np.trace(kyy)) / (ny * (ny - 1))
    return float(term_x + term_y - 2 * kxy.mean())


def simulate(func, starts: np.ndarray, t_start: float, n_steps: int, d: float, rng: np.random.Generator, batch_size: int = 2048) -> np.ndarray:
    if d < 0:
        raise ValueError(f"Negative learned diffusion parameter d={d}")
    out = np.empty_like(starts, dtype=np.float32)
    sigma = np.sqrt(2.0 * d * DT)
    for lo in range(0, len(starts), batch_size):
        z = torch.from_numpy(starts[lo:lo + batch_size].astype(np.float32, copy=True))
        for step in range(n_steps):
            t = torch.tensor(t_start + step * DT, dtype=torch.float32)
            with torch.no_grad():
                v = func.hyper_net1(t, z)
            noise = rng.normal(0.0, sigma, size=z.shape).astype(np.float32)
            z = z + v * DT + torch.from_numpy(noise)
        out[lo:lo + len(z)] = z.numpy()
    return out


def support_radii(endpoints: np.ndarray, targets_full: np.ndarray) -> tuple[np.ndarray, float]:
    nn = NearestNeighbors(n_neighbors=31, algorithm="kd_tree", n_jobs=1).fit(targets_full)
    loo, loo_idx = nn.kneighbors(targets_full, n_neighbors=32)
    loo30 = np.empty(len(targets_full), dtype=np.float64)
    for i, (distances, indices) in enumerate(zip(loo, loo_idx)):
        other = distances[indices != i]
        loo30[i] = other[29]
    empirical_r99 = float(np.quantile(loo30, 0.99))
    predicted, _ = nn.kneighbors(endpoints, n_neighbors=1)
    return predicted[:, 0], empirical_r99


def evaluate_pair(model_x: np.ndarray, identity_x: np.ndarray, random_x: np.ndarray, target_x: np.ndarray, target_full: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(6047 + seed)
    xi = rng.choice(len(model_x), N_METRIC, replace=False)
    yi = rng.choice(len(target_x), N_METRIC, replace=False)
    x_model, x_id, x_rand, y = model_x[xi], identity_x[xi], random_x[xi], target_x[yi]
    median_sq = float(np.median(pdist(y, metric="sqeuclidean")))
    median_dist = float(np.median(pdist(y, metric="euclidean")))
    reg = 0.05 * median_sq
    model_sink = sinkhorn_squared(x_model, y, reg)
    identity_sink = sinkhorn_squared(x_id, y, reg)
    random_sink = sinkhorn_squared(x_rand, y, reg)
    model_mmd = gaussian_mmd2(x_model, y, median_dist)
    identity_mmd = gaussian_mmd2(x_id, y, median_dist)
    random_mmd = gaussian_mmd2(x_rand, y, median_dist)
    radii, radius99 = support_radii(model_x, target_full)
    return {
        "seed": seed,
        "n_sources": int(len(model_x)),
        "n_targets": int(len(target_x)),
        "metric_subsample": N_METRIC,
        "sinkhorn_regularization": reg,
        "mmd_bandwidth": median_dist,
        "model_debiased_sinkhorn": model_sink,
        "identity_debiased_sinkhorn": identity_sink,
        "random_matched_debiased_sinkhorn": random_sink,
        "model_mmd2": model_mmd,
        "identity_mmd2": identity_mmd,
        "random_matched_mmd2": random_mmd,
        "sinkhorn_improvement_over_identity": 1 - model_sink / identity_sink if identity_sink != 0 else float("nan"),
        "sinkhorn_improvement_over_random_matched": 1 - model_sink / random_sink if random_sink != 0 else float("nan"),
        "mmd_improvement_over_identity": identity_mmd - model_mmd,
        "mmd_improvement_over_random_matched": random_mmd - model_mmd,
        "endpoint_radius99_exceed_fraction": float(np.mean(radii > radius99)),
        "empirical_leave_one_out_30nn_radius_p99": radius99,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "external/DiffusionOT"))
    import utility

    seed = args.seed
    sd = ARM / f"seed_{seed}"
    run = json.loads((sd / "ruot_run.json").read_text())
    if run.get("status") != "complete":
        raise RuntimeError(f"Seed {seed} is not complete")
    ckpt = torch.load(sd / "ckpt_Mouse.pth", map_location="cpu")
    func = utility.RUOT(in_out_dim=2, hidden_dim=16, n_hiddens=4, activation="Tanh", d=0.001).cpu()
    func.load_state_dict(ckpt["func_state_dict"])
    func.eval()
    d = float(func.d.detach().item())
    with np.load(sd / "latent_input.npz", allow_pickle=False) as data:
        latent = data["latent_ae"].astype(np.float32)
        ids = data["original_cell_id"].astype(str)
        times = data["time_label"].astype(str).astype(int)
    index = pd.read_csv(ARM / "cell_index.tsv", sep="\t")
    if not np.array_equal(ids, index.original_cell_id.astype(str).to_numpy()):
        raise ValueError(f"Seed {seed}: latent cell IDs are not canonical")
    out_dir = ROOT / "results/phase1R/snapshot_qc" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for (day_a, day_b), t_start in zip(TIME_LABELS, [0.0, 1.0]):
        src_rows = stratified_sample(index, day_a, N_SAMPLE, 6047)
        target_rows = stratified_sample(index, day_b, N_SAMPLE, 6047 + day_b)
        starts = latent[src_rows]
        targets = latent[target_rows]
        target_full = latent[times == day_b]
        n_steps = int(round((day_b - day_a) / ((4 - 2) if day_a == 2 else (6 - 4)) / DT))
        rng_model = np.random.default_rng(seed * 100000 + day_a)
        pred = simulate(func, starts, t_start, n_steps, d, rng_model)
        rng_random = np.random.default_rng(6047 + seed * 10 + day_a)
        neighbor_model = NearestNeighbors(n_neighbors=30, algorithm="kd_tree", n_jobs=1).fit(target_full)
        _, neighbors = neighbor_model.kneighbors(starts, n_neighbors=30)
        random_choice = rng_random.integers(0, neighbors.shape[1], size=len(starts))
        random_endpoints = target_full[neighbors[np.arange(len(starts)), random_choice]]
        metrics = evaluate_pair(pred, starts, random_endpoints, targets, target_full, seed)
        metrics.update({"pairing": f"{day_a}_to_{day_b}", "status": "evaluated"})
        all_metrics.append(metrics)
        np.savez_compressed(
            out_dir / f"{day_a}_to_{day_b}_endpoints.npz",
            source_row_index=src_rows,
            source_cell_id=ids[src_rows],
            target_row_index=target_rows,
            target_cell_id=ids[target_rows],
            model_endpoints=pred,
            identity_endpoints=starts,
            random_matched_endpoints=random_endpoints,
            learned_shared_d=np.asarray(d, dtype=np.float32),
            seed=np.asarray(seed, dtype=np.int32),
        )
    result = {"seed": seed, "status": "COMPLETE_LINEAGE_BLIND", "n_sample_per_timepoint": N_SAMPLE, "dt": DT, "learned_shared_diffusion": d, "metrics": all_metrics, "lineage_information_opened": False}
    path = out_dir / "snapshot_qc.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"seed": seed, "status": result["status"], "snapshot_qc": str(path.relative_to(ROOT)), "sha256": sha256(path)}, indent=2))


if __name__ == "__main__":
    main()
