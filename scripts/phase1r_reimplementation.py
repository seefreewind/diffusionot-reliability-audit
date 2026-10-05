#!/usr/bin/env python3
"""Build and train an ID-preserving DiffusionOT-like arm with lineage hidden.

Allowed data inputs are the public expression matrix, gene names and non-lineage
cell metadata. No clone assignment or fate table is imported or opened.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/model_visible/phase1r_source"
OUT = ROOT / "results/phase1R/REIMPLEMENTED_ID_PRESERVING_ARM"
CONFIG = ROOT / "configs/phase1r_reimplementation.json"
SEEDS = [11, 23, 47, 71, 101]
EXPR_PATH = RAW / "stateFate_inVitro_normed_counts.mtx.gz"
META_PATH = RAW / "stateFate_inVitro_metadata.txt.gz"
GENE_PATH = RAW / "stateFate_inVitro_gene_names.txt.gz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def to_plain_list(value):
    """Detach nested tensors before storing diagnostic histories."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [to_plain_list(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in (EXPR_PATH, META_PATH, GENE_PATH)}


def load_source_metadata() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    # The released source metadata contains no clone/fate columns. The barcode
    # is retained only in the ID sidecar and never passed to either model.
    metadata = pd.read_csv(META_PATH, sep="\t")
    required = ["Library", "Cell barcode", "Time point", "Cell type annotation"]
    if any(column not in metadata.columns for column in required):
        raise ValueError(f"Required source metadata columns missing: {required}")
    forbidden = [column for column in metadata.columns if any(term in column.casefold() for term in ("clone", "fate", "lineage"))]
    if forbidden:
        raise ValueError(f"Refusing metadata with lineage-related columns: {forbidden}")
    identifiers = (metadata["Library"].astype(str) + "|" + metadata["Cell barcode"].astype(str)).to_numpy(dtype=str)
    if pd.Index(identifiers).duplicated().any():
        raise ValueError("Composite source cell IDs are not unique")
    times = metadata["Time point"].astype(int).to_numpy()
    labels = metadata["Cell type annotation"].astype(str).to_numpy(dtype=str)
    return metadata, identifiers, times, labels


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    hashes = source_hashes()
    metadata, identifiers, times, labels = load_source_metadata()
    gene_names = pd.read_csv(GENE_PATH, header=None, compression="gzip", dtype=str)[0].to_numpy(dtype=str)
    if len(metadata) != 130887 or len(gene_names) != 25289:
        raise ValueError(f"Unexpected source dimensions: cells={len(metadata)}, genes={len(gene_names)}")

    expression = scipy.io.mmread(str(EXPR_PATH)).tocsr()
    expression.sum_duplicates()
    expression.eliminate_zeros()
    if expression.shape != (len(metadata), len(gene_names)):
        raise ValueError(f"Expression shape {expression.shape} does not match metadata/gene names")
    if expression.data.size and np.min(expression.data) < 0:
        raise ValueError("Expression contains negative values; log1p preprocessing is not valid")

    min_detected_cells = int(math.ceil(0.01 * len(metadata)))
    detected = np.asarray(expression.getnnz(axis=0)).reshape(-1)
    gene_filter = detected >= min_detected_cells
    expression = expression[:, gene_filter].tocsr()
    kept_gene_names = gene_names[gene_filter]
    expression.data = np.log1p(expression.data).astype(np.float32, copy=False)
    expression.eliminate_zeros()

    import anndata as ad
    import scanpy as sc

    obs = pd.DataFrame(
        {"timepoint": times, "cell_type": labels},
        index=pd.Index(identifiers, name="original_cell_id"),
    )
    var = pd.DataFrame(index=pd.Index(kept_gene_names, name="gene_name"))
    adata = ad.AnnData(X=expression, obs=obs, var=var)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        n_top_genes=3000,
        batch_key=None,
        subset=False,
        inplace=True,
    )
    hvg_mask = adata.var["highly_variable"].to_numpy(dtype=bool)
    if int(hvg_mask.sum()) != 3000:
        raise ValueError(f"HVG selection returned {int(hvg_mask.sum())}, expected 3000")
    selected_indices = np.flatnonzero(hvg_mask)
    selected_genes = kept_gene_names[selected_indices]
    selected = adata.X[:, selected_indices].tocsr()

    ids_path = OUT / "cell_index.tsv"
    index_frame = pd.DataFrame({
        "source_row_index": np.arange(len(metadata), dtype=np.int64),
        "original_cell_id": identifiers,
        "library": metadata["Library"].astype(str).to_numpy(),
        "cell_barcode": metadata["Cell barcode"].astype(str).to_numpy(),
        "timepoint": times,
        "cell_type": labels,
    })
    index_frame.to_csv(ids_path, sep="\t", index=False)
    genes_path = OUT / "selected_genes.tsv"
    pd.DataFrame({"selected_feature_order": np.arange(3000), "gene_name": selected_genes}).to_csv(genes_path, sep="\t", index=False)

    matrix_path = OUT / "expression_log1p_top3000.npy"
    dense = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(len(metadata), 3000))
    chunk = 512
    for start in range(0, len(metadata), chunk):
        end = min(start + chunk, len(metadata))
        dense[start:end] = selected[start:end].toarray().astype(np.float32, copy=False)
    dense.flush()
    del dense, selected, adata, expression
    gc.collect()

    payload = {
        "arm": "REIMPLEMENTED_ID_PRESERVING_ARM",
        "config": config,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes_sha256": hashes,
        "source_shape": [130887, 25289],
        "retained_cells": int(len(metadata)),
        "cell_filter": "none; all source rows retained in canonical source order because the paper notebook does not disclose its rare-cell-type threshold or selected row indices",
        "source_cell_id_definition": "Library|Cell barcode",
        "unique_cell_ids": int(pd.Index(identifiers).nunique()),
        "timepoint_counts": {str(int(k)): int(v) for k, v in pd.Series(times).value_counts().sort_index().items()},
        "cell_type_counts": {str(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()},
        "gene_filter": f"nonzero in at least {min_detected_cells} cells (ceil(1% of 130887))",
        "n_genes_after_detection_filter": int(gene_filter.sum()),
        "transform": "log1p applied to distributed normalized expression values",
        "hvg": {"implementation": "Scanpy", "version": sc.__version__, "flavor": "seurat", "n_top_genes": 3000, "batch_key": None, "selected_order": "source feature order among selected HVGs"},
        "scaling": "none; no explicit scaling is shown in the official notebook's NPZ-writing cell",
        "lineage_inputs_opened": [],
        "model_features_exclude_cell_ids_and_metadata": True,
        "outputs": {
            "cell_index_tsv": str(ids_path.relative_to(ROOT)),
            "selected_genes_tsv": str(genes_path.relative_to(ROOT)),
            "expression_npy": str(matrix_path.relative_to(ROOT)),
        },
    }
    payload["outputs_sha256"] = {k: sha256(ROOT / v) for k, v in payload["outputs"].items()}
    (OUT / "preparation_metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: payload[k] for k in ["retained_cells", "n_genes_after_detection_filter", "hvg", "outputs_sha256"]}, indent=2))


class MatrixRows(Dataset):
    def __init__(self, matrix: np.ndarray, indices: np.ndarray):
        self.matrix = matrix
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.matrix[self.indices[index]], dtype=np.float32, copy=True))


def train_ae_seed(seed: int) -> Path:
    ae_dir = OUT / f"seed_{seed}"
    ae_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ae_dir / "autoencoder.pt"
    latent_path = ae_dir / "latent_input.npz"
    marker_path = ae_dir / "autoencoder_run.json"
    if marker_path.exists() and checkpoint_path.exists() and latent_path.exists():
        old = json.loads(marker_path.read_text())
        if old.get("status") == "complete" and old.get("seed") == seed:
            return latent_path

    import scanpy as sc
    sys.path.insert(0, str(ROOT / "external/DiffusionOT"))
    from AE.models import AutoEncoder

    matrix = np.load(OUT / "expression_log1p_top3000.npy", mmap_mode="r")
    index_table = pd.read_csv(OUT / "cell_index.tsv", sep="\t")
    ids = index_table["original_cell_id"].astype(str).to_numpy(dtype=str)
    row_ids = index_table["source_row_index"].to_numpy(dtype=np.int64)
    times = index_table["timepoint"].to_numpy(dtype=np.int64)
    types = index_table["cell_type"].astype(str).to_numpy(dtype=str)
    train_idx, test_idx = train_test_split(np.arange(len(ids)), test_size=0.1, random_state=seed)
    np.savez_compressed(ae_dir / "split_indices.npz", train_indices=train_idx, validation_indices=test_idx)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cpu")
    model = AutoEncoder(
        in_dim=matrix.shape[1], n_latent=2, n_hidden=300, n_layers=1,
        activate_type="relu", dropout=0.2, norm=True, seed=seed,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(MatrixRows(matrix, train_idx), batch_size=128, shuffle=True, generator=train_gen, num_workers=0)
    val_loader = DataLoader(MatrixRows(matrix, test_idx), batch_size=128, shuffle=False, num_workers=0)
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best = float("inf")
    patience = 0
    started = datetime.now(timezone.utc).isoformat()
    log_path = ae_dir / "autoencoder.log"
    with log_path.open("w") as log, contextlib.redirect_stdout(log):
        for epoch in range(500):
            model.train()
            train_total = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                train_total += loss.item()
                optimizer.step()
            train_loss = train_total / len(train_idx)
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    val_total += model(batch.to(device)).item()
            val_loss = val_total / len(test_idx)
            history["epoch"].append(epoch)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            print(f"Epoch {epoch}: train loss = {train_loss:.6f}, val error = {val_loss:.6f}", flush=True)
            if best - val_loss >= 0.001:
                best = val_loss
                patience = 0
            else:
                patience += 1
            if patience >= 30:
                print("Early stopping after 30 epochs without a 0.001 improvement", flush=True)
                break

    model.eval()
    latent = np.empty((len(ids), 2), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(ids), 4096):
            end = min(start + 4096, len(ids))
            batch = torch.from_numpy(np.asarray(matrix[start:end], dtype=np.float32).copy()).to(device)
            latent[start:end] = model.get_latent_representation(batch).cpu().numpy()
    np.savez_compressed(
        latent_path,
        latent_ae=latent,
        time_label=times.astype(str),
        type_label=types,
        original_cell_id=ids,
        source_row_index=row_ids,
    )
    torch.save({
        "func_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": history,
        "seed": seed,
        "input_feature_count": int(matrix.shape[1]),
    }, checkpoint_path)
    marker = {
        "status": "complete",
        "seed": seed,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "epochs_run": len(history["epoch"]),
        "best_validation_loss": float(best),
        "checkpoint_sha256": sha256(checkpoint_path),
        "latent_input_sha256": sha256(latent_path),
        "lineage_information_opened": False,
    }
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    return latent_path


def train_ruot_seed(seed: int) -> None:
    seed_dir = OUT / f"seed_{seed}"
    latent_path = seed_dir / "latent_input.npz"
    done_path = seed_dir / "ruot_run.json"
    if not latent_path.exists():
        raise FileNotFoundError(latent_path)
    if done_path.exists() and json.loads(done_path.read_text()).get("status") == "complete":
        return

    started = datetime.now(timezone.utc).isoformat()
    script_digest = sha256(Path(__file__).resolve())
    run_record = {
        "status": "running", "seed": seed, "started_at_utc": started,
        "script_sha256": script_digest, "lineage_information_opened": False,
    }
    done_path.write_text(json.dumps(run_record, indent=2) + "\n")

    sys.path.insert(0, str(ROOT / "external/DiffusionOT"))
    import utility

    args = SimpleNamespace(
        dataset="Mouse", timepoints=[0.0, 1.0, 2.0], niters=5000, lr=3e-3,
        num_samples=100, hidden_dim=16, n_hiddens=4, activation="Tanh",
        gpu=0, input_dir=str(seed_dir), save_dir=str(seed_dir), seed=seed, d=0.001,
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cpu")
    z = np.load(latent_path, allow_pickle=False)
    latent = z["latent_ae"]
    labels = z["time_label"].astype(str)
    data_train, data_type = [], []
    for time_label in ["2", "4", "6"]:
        idx = np.flatnonzero(labels == time_label)
        data_train.append(torch.from_numpy(latent[idx].copy()).type(torch.float32).to(device))
        data_type.append(z["type_label"][idx])
    if [len(x) for x in data_train] != [28249, 48498, 54140]:
        raise ValueError(f"Unexpected reimplementation timepoint counts: {[len(x) for x in data_train]}")
    time_points = range(len(data_train))
    train_time = list(time_points)

    func = utility.RUOT(
        in_out_dim=data_train[0].shape[1], hidden_dim=args.hidden_dim,
        n_hiddens=args.n_hiddens, activation=args.activation, d=args.d,
    ).to(device)
    func.apply(utility.initialize_weights)
    options = {"method": "Dopri5", "h": None, "rtol": 1e-3, "atol": 1e-5, "print_neval": False, "neval_max": 1000000, "safety": None}
    optimizer = torch.optim.Adam(func.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.niters - 400, args.niters - 200], gamma=0.5, last_epoch=-1)
    mse = torch.nn.MSELoss()
    loss_hist, trans, l2_1, l2_2, l2_3, l2_4, sigmas, d_values = [], [], [], [], [], [], [], []
    log_path = seed_dir / "ruot.log"
    with log_path.open("w") as log, contextlib.redirect_stdout(log):
        # Match the public training.py three-stage optimization. Only the
        # time-stratified, ID-free latent arrays are available to this routine.
        func.d.requires_grad = False
        for layer in [func.hyper_net1, func.hyper_net2]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, func.parameters()), lr=args.lr, weight_decay=0.01)
        sigma_now = 1
        for itr in range(1, 201):
            optimizer.zero_grad()
            loss_score = utility.pre_train_score(mse, func, args, data_train, train_time, args.timepoints, sigma_now, device, itr)
            loss_score.backward(); optimizer.step()
            print(f"Pre_train_score_Iter: {itr}, loss: {loss_score.item():.6f}", flush=True)
        torch.save({"func_state_dict": func.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, seed_dir / "ckpt_Mouse_score.pth")

        for parameter in func.hyper_net3.parameters():
            parameter.requires_grad = False
        for layer in [func.hyper_net1, func.hyper_net2]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, func.parameters()), lr=args.lr, weight_decay=0.01)
        sigma_now = 1
        for itr in range(1, 101):
            optimizer.zero_grad()
            loss, sigma_now, l2v3, l2v4 = utility.pre_train_model(mse, func, args, data_train, train_time, args.timepoints, sigma_now, options, device, itr)
            loss.backward(); optimizer.step(); scheduler.step()
            loss_hist.append(float(loss.item())); sigmas.append(float(sigma_now)); l2_3.append(to_plain_list(l2v3))
            print(f"Pre_train_model_Iter: {itr}, loss: {loss.item():.6f}", flush=True)
        torch.save({"func_state_dict": func.state_dict()}, seed_dir / "ckpt_Mouse_pre.pth")

        for layer in [func.hyper_net1, func.hyper_net2, func.hyper_net3]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, func.parameters()), lr=args.lr, weight_decay=0.01)
        sigma_now = 0.5
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            loss, loss1, sigma_now, l2v1, l2v2, l2v3, l2v4 = utility.train_model(mse, func, args, data_train, train_time, args.timepoints, sigma_now, options, device, itr)
            loss.backward(); optimizer.step(); scheduler.step()
            loss_hist.append(float(loss.item()))
            trans.append(float(loss1[-1].mean(0).item()))
            sigmas.append(float(sigma_now))
            l2_1.append(to_plain_list(l2v1)); l2_2.append(to_plain_list(l2v2)); l2_3.append(to_plain_list(l2v3)); l2_4.append(to_plain_list(l2v4))
            if itr % 100 == 0:
                print(f"Iter: {itr}, loss: {loss.item():.6f}", flush=True)
            if itr % 500 == 0:
                dt = utility.diffusion_fit(func, args, data_train, train_time, args.timepoints, device, time_tt=0.01)
                D = torch.mean(dt)
                d_values.append(float(D.detach().cpu().item()))
                torch.save({"func_state_dict": func.state_dict()}, seed_dir / f"ckpt_Mouse_itr{itr}.pth")
            if itr % 100 == 0 and itr < 500:
                dt = utility.diffusion_fit(func, args, data_train, train_time, args.timepoints, device, time_tt=0.01)
                func.d = torch.nn.Parameter(torch.mean(dt))
        print(f"Training complete after {args.niters} iterations.", flush=True)

    final_ckpt = seed_dir / "ckpt_Mouse.pth"
    torch.save({
        "func_state_dict": func.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "LOSS": loss_hist, "TRANS": trans, "L2_1": l2_1, "L2_2": l2_2,
        "L2_3": l2_3, "L2_4": l2_4, "Sigma": sigmas, "D": d_values,
    }, final_ckpt)
    done_path.write_text(json.dumps({
        "status": "complete", "seed": seed, "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "iterations": args.niters, "pretraining_iterations": 300,
        "checkpoint_sha256": sha256(final_ckpt), "lineage_information_opened": False,
        "script_sha256": script_digest,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "ae", "ruot", "all"])
    args = parser.parse_args()
    if args.stage in {"prepare", "all"}:
        prepare()
    if args.stage in {"ae", "all"}:
        for seed in SEEDS:
            path = train_ae_seed(seed)
            print(f"AE complete for seed={seed}: {path.relative_to(ROOT)}", flush=True)
    if args.stage in {"ruot", "all"}:
        for seed in SEEDS:
            try:
                train_ruot_seed(seed)
            except Exception as exc:
                marker = OUT / f"seed_{seed}" / "ruot_run.json"
                record = json.loads(marker.read_text()) if marker.exists() else {"seed": seed}
                record.update({
                    "status": "failed", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "lineage_information_opened": False,
                })
                marker.write_text(json.dumps(record, indent=2) + "\n")
                raise
            print(f"RUOT complete for seed={seed}", flush=True)


if __name__ == "__main__":
    main()
