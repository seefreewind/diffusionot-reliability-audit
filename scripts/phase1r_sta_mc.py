#!/usr/bin/env python3
"""Frozen lineage-blind STA Monte Carlo convergence audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import entropy

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "results/phase1R/REIMPLEMENTED_ID_PRESERVING_ARM"
OUT = ROOT / "results/phase1R/sta_mc"
SEEDS = [11, 23, 47, 71, 101]
GRID = np.asarray([100, 300, 1000, 3000, 10000], dtype=np.int64)
AUDIT_N = 100
DT = 0.01
CELL_SET_SEED = 101
PROB_TOL = 0.01
ENTROPY_TOL = 0.01


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fixed_audit_rows(index: pd.DataFrame, day: int) -> np.ndarray:
    pool = index.index[index.timepoint.eq(day)].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(CELL_SET_SEED + day)
    selected = np.sort(rng.choice(pool, size=AUDIT_N, replace=False))
    return selected


def simulate_cell(func, mapper, start: np.ndarray, t_start: float, n_steps: int, d: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if d < 0:
        raise ValueError(f"Negative learned diffusion coefficient d={d}")
    states = np.repeat(start.reshape(1, 2).astype(np.float32), int(GRID[-1]), axis=0)
    sigma = np.sqrt(2.0 * d * DT)
    with torch.no_grad():
        for step in range(n_steps):
            t = torch.tensor(t_start + step * DT, dtype=torch.float32)
            z = torch.from_numpy(states)
            velocity = func.hyper_net1(t, z).numpy()
            noise = rng.normal(0.0, sigma, size=states.shape).astype(np.float32)
            states = states + velocity * DT + noise
    probability_curves = np.empty((len(GRID), len(mapper["classes"])), dtype=np.float64)
    class_probabilities = mapper["model"].predict_proba(states)
    fitted_classes = mapper["model"][-1].classes_
    class_to_column = {name: i for i, name in enumerate(mapper["classes"])}
    ordered_probabilities = np.zeros((len(states), len(mapper["classes"])), dtype=np.float64)
    for j, category in enumerate(fitted_classes):
        ordered_probabilities[:, class_to_column[category]] = class_probabilities[:, j]
    for i, n in enumerate(GRID):
        probability_curves[i] = ordered_probabilities[:n].mean(axis=0)
    return probability_curves, states[GRID - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    seed = args.seed
    sys.path.insert(0, str(ROOT / "external/DiffusionOT"))
    import utility

    sd = ARM / f"seed_{seed}"
    run = json.loads((sd / "ruot_run.json").read_text())
    if run.get("status") != "complete":
        raise RuntimeError(f"Seed {seed} RUOT is not complete")
    checkpoint = torch.load(sd / "ckpt_Mouse.pth", map_location="cpu")
    func = utility.RUOT(in_out_dim=2, hidden_dim=16, n_hiddens=4, activation="Tanh", d=0.001).cpu()
    func.load_state_dict(checkpoint["func_state_dict"])
    func.eval()
    d = float(func.d.detach().item())
    mapper_path = ROOT / "results/phase1R/terminal_classifier" / f"seed_{seed}" / "day6_state_mapper.joblib"
    mapper = joblib.load(mapper_path)
    index = pd.read_csv(ARM / "cell_index.tsv", sep="\t")
    with np.load(sd / "latent_input.npz", allow_pickle=False) as data:
        latent = data["latent_ae"].astype(np.float32)
        ids = data["original_cell_id"].astype(str)
        rows = data["source_row_index"].astype(np.int64)
        times = data["time_label"].astype(str).astype(np.int64)
    if not np.array_equal(ids, index.original_cell_id.astype(str).to_numpy()) or not np.array_equal(rows, np.arange(len(rows))):
        raise ValueError(f"Seed {seed}: STA inputs fail canonical ID checks")

    all_curves, all_ids, all_days = [], [], []
    for day, t_start, n_steps in [(2, 0.0, 200), (4, 1.0, 100)]:
        audit_rows = fixed_audit_rows(index, day)
        if not np.all(times[audit_rows] == day):
            raise ValueError("Fixed audit set contains wrong timepoint rows")
        for cell_i, row in enumerate(audit_rows):
            rng = np.random.default_rng(20260927 + seed * 100000 + day * 1000 + cell_i)
            curves, _ = simulate_cell(func, mapper, latent[row], t_start, n_steps, d, rng)
            all_curves.append(curves)
            all_ids.append(ids[row])
            all_days.append(day)

    curves = np.stack(all_curves)
    max_probability = curves.max(axis=2)
    norm_entropy = entropy(curves, axis=2) / np.log(curves.shape[2])
    delta_max_p = np.abs(np.diff(max_probability, axis=1))
    delta_entropy = np.abs(np.diff(norm_entropy, axis=1))
    global_pass = (delta_max_p.max(axis=0) < PROB_TOL) & (delta_entropy.max(axis=0) < ENTROPY_TOL)
    converged_index = next((i + 1 for i, passed in enumerate(global_pass) if passed), None)
    per_cell_first = []
    for i in range(len(curves)):
        passed = (delta_max_p[i] < PROB_TOL) & (delta_entropy[i] < ENTROPY_TOL)
        per_cell_first.append(next((j + 1 for j, value in enumerate(passed) if value), None))
    summary = {
        "seed": seed,
        "status": "STA_CONVERGED" if converged_index is not None else "STA_MONTE_CARLO_UNSTABLE",
        "grid": GRID.tolist(),
        "primary_trajectory_count": int(GRID[converged_index]) if converged_index is not None else None,
        "fixed_audit_cells": AUDIT_N,
        "fixed_cell_set_seed": CELL_SET_SEED,
        "cells_by_day": {"2": AUDIT_N, "4": AUDIT_N},
        "max_probability_change_by_step": delta_max_p.max(axis=0).tolist(),
        "max_normalized_entropy_change_by_step": delta_entropy.max(axis=0).tolist(),
        "all_cells_first_converged_grid_index": [int(x) if x is not None else None for x in per_cell_first],
        "learned_shared_diffusion": d,
        "lineage_information_opened": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_npz = OUT / f"seed_{seed}_sta_mc_curves.npz"
    np.savez_compressed(
        out_npz,
        audit_cell_id=np.asarray(all_ids),
        audit_timepoint=np.asarray(all_days, dtype=np.int8),
        trajectory_grid=GRID,
        class_probability_curves=curves.astype(np.float32),
        normalized_entropy=norm_entropy.astype(np.float32),
        maximum_class_probability=max_probability.astype(np.float32),
        class_order=np.asarray(mapper["classes"]),
        learned_shared_diffusion=np.asarray(d, dtype=np.float32),
    )
    summary["curves_path"] = str(out_npz.relative_to(ROOT))
    summary["curves_sha256"] = sha256(out_npz)
    summary_path = OUT / f"seed_{seed}_sta_mc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"seed": seed, "status": summary["status"], "primary_trajectory_count": summary["primary_trajectory_count"], "summary_sha256": sha256(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
