#!/usr/bin/env python3
"""Train and freeze the non-lineage Day-6 state mapper for each AE seed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "results/phase1R/REIMPLEMENTED_ID_PRESERVING_ARM"
OUT = ROOT / "results/phase1R/terminal_classifier"
SEEDS = [11, 23, 47, 71, 101]
CLASSES = [
    "Monocyte", "Neutrophil", "Basophil", "Megakaryocyte", "Mast",
    "Erythroid", "Eosinophil", "Lymphoid", "Other/Unresolved",
    "Undifferentiated/Unresolved progenitor",
]
MAPPING = {
    "Monocyte": "Monocyte",
    "Neutrophil": "Neutrophil",
    "Baso": "Basophil",
    "Meg": "Megakaryocyte",
    "Mast": "Mast",
    "Erythroid": "Erythroid",
    "Eos": "Eosinophil",
    "Lymphoid": "Lymphoid",
    "Ccr7_DC": "Other/Unresolved",
    "pDC": "Other/Unresolved",
    "Undifferentiated": "Undifferentiated/Unresolved progenitor",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mapper() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000, random_state=8103),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cell_index = pd.read_csv(ARM / "cell_index.tsv", sep="\t")
    expected_ids = cell_index["original_cell_id"].astype(str).to_numpy()
    source_rows = cell_index["source_row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(source_rows, np.arange(len(source_rows))):
        raise ValueError("Cell index is not in canonical source order")
    if cell_index["original_cell_id"].duplicated().any():
        raise ValueError("Cell index contains duplicate cell IDs")
    libraries = cell_index["library"].astype(str).to_numpy()
    timepoints = cell_index["timepoint"].to_numpy(dtype=np.int64)
    raw_labels = cell_index["cell_type"].astype(str).to_numpy()
    if not set(raw_labels).issubset(MAPPING):
        raise ValueError(f"Unfrozen cell-type annotations found: {sorted(set(raw_labels)-set(MAPPING))}")
    labels = np.array([MAPPING[x] for x in raw_labels], dtype=str)

    all_runs = {}
    for seed in SEEDS:
        latent_path = ARM / f"seed_{seed}" / "latent_input.npz"
        with np.load(latent_path, allow_pickle=False) as data:
            ids = data["original_cell_id"].astype(str)
            rows = data["source_row_index"].astype(np.int64)
            times = data["time_label"].astype(str).astype(np.int64)
            x_all = data["latent_ae"].astype(np.float32)
        if not np.array_equal(ids, expected_ids) or not np.array_equal(rows, source_rows) or not np.array_equal(times, timepoints):
            raise ValueError(f"Seed {seed}: latent input failed the canonical ID/time order check")
        d6 = timepoints == 6
        x = x_all[d6]
        y = labels[d6]
        groups = libraries[d6]
        if set(y) != set(CLASSES):
            raise ValueError(f"Seed {seed}: Day-6 classes differ from frozen taxonomy: {sorted(set(y))}")

        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=8103)
        oof = np.zeros((len(y), len(CLASSES)), dtype=np.float64)
        fold_id = np.full(len(y), -1, dtype=np.int8)
        for fold, (train, test) in enumerate(splitter.split(x, y, groups=groups)):
            model = mapper()
            model.fit(x[train], y[train])
            probs = model.predict_proba(x[test])
            model_classes = model[-1].classes_
            for j, category in enumerate(model_classes):
                oof[test, CLASSES.index(category)] = probs[:, j]
            fold_id[test] = fold
        if np.any(fold_id < 0) or not np.allclose(oof.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError(f"Seed {seed}: incomplete or invalid out-of-fold probabilities")
        pred = np.asarray(CLASSES, dtype=object)[oof.argmax(axis=1)]
        matrix = confusion_matrix(y, pred, labels=CLASSES)
        report = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)

        final = mapper()
        final.fit(x, y)
        seed_out = OUT / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        model_path = seed_out / "day6_state_mapper.joblib"
        joblib.dump({"model": final, "classes": CLASSES, "seed": seed}, model_path)
        oof_path = seed_out / "day6_oof_predictions.npz"
        np.savez_compressed(
            oof_path,
            source_row_index=np.flatnonzero(d6),
            original_cell_id=expected_ids[d6],
            true_state=y,
            predicted_probability=oof.astype(np.float32),
            predicted_state=pred,
            fold_id=fold_id,
            classes=np.asarray(CLASSES),
            library=groups,
        )
        cm_path = seed_out / "day6_oof_confusion.tsv"
        pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(cm_path, sep="\t", index_label="true_state")
        payload = {
            "seed": seed,
            "status": "FROZEN_PRE_LINEAGE_UNBLIND",
            "n_day6_cells": int(d6.sum()),
            "n_library_groups": int(pd.Index(groups).nunique()),
            "grouped_cv": {"splitter": "StratifiedGroupKFold", "n_splits": 5, "shuffle": True, "seed": 8103, "group": "Library"},
            "classes": CLASSES,
            "accuracy_oof": float(report["accuracy"]),
            "macro_f1_oof": float(report["macro avg"]["f1-score"]),
            "per_class_oof": {c: report[c] for c in CLASSES},
            "confusion_matrix_path": str(cm_path.relative_to(ROOT)),
            "confusion_matrix_sha256": sha256(cm_path),
            "model_path": str(model_path.relative_to(ROOT)),
            "model_sha256": sha256(model_path),
            "oof_predictions_path": str(oof_path.relative_to(ROOT)),
            "oof_predictions_sha256": sha256(oof_path),
            "input_latent_sha256": sha256(latent_path),
            "lineage_information_opened": False,
            "interpretation": "cell-state mapping performance only; not lineage fidelity",
        }
        meta_path = seed_out / "freeze_metadata.json"
        meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        payload["freeze_metadata_sha256"] = sha256(meta_path)
        all_runs[str(seed)] = payload

    summary = {
        "status": "FROZEN_PRE_LINEAGE_UNBLIND",
        "taxonomy_mapping": MAPPING,
        "class_order": CLASSES,
        "model": "StandardScaler + multinomial LogisticRegression(C=1, lbfgs, max_iter=3000)",
        "validation": "5-fold StratifiedGroupKFold grouped by Library, shuffle=true, seed=8103",
        "training_cells": "Day-6 cells only",
        "uses_lineage": False,
        "warning": "Classifier performance maps latent endpoint states to annotation categories; it is not a measure of clone prediction accuracy.",
        "seeds": all_runs,
    }
    summary_path = OUT / "classifier_freeze_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": summary["status"], "macro_f1_oof": {s: all_runs[s]["macro_f1_oof"] for s in all_runs}, "summary": str(summary_path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
