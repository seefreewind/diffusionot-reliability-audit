# DiffusionOT reliability audit

Frozen aggregate data, author-written computational code and figure regeneration sources for a pre-lineage reliability audit of a stochastic optimal-transport implementation in single-cell hematopoiesis.

## Reproduce the four main figures

Use Python 3.12 (or a compatible version) and install `environment/requirements-figures.txt` in a virtual environment. Run from the repository root:

```sh
python -m pip install -r environment/requirements-figures.txt
python figure_source_data/Figure1/generate_figure.py
python figure_source_data/Figure2/generate_figure.py
python figure_source_data/Figure3/generate_figure.py
python figure_source_data/Figure4/generate_figure.py
```

The generators read only the included frozen aggregates. They export vector PDF, SVG and 300 dpi PNG into `figures/`; the PDF/PNG basenames retain the manuscript figure-version suffix. Regeneration does not fit models, run trajectories or access lineage outcomes. Exact PDF bytes can vary with timestamps and library versions; `provenance/SHA256SUMS.tsv` records the archived bytes.

## Contents and units

- `figure_source_data/Figure1/`: machine-readable evidence schematic and provenance counts.
- `figure_source_data/Figure2/`: five seeds × two transitions × three step sizes, measured as out-of-support percentages.
- `figure_source_data/Figure3/`: stored paired full-SDE/drift-only support and five stored step-refinement curves.
- `figure_source_data/Figure4/`: variance components, reference-seed dynamics and the 15 initial sampling-run statuses. Two confirmation runs are separated in the full aggregate status file.
- `aggregate_results/`: seed/time-stratum diagnostics and aggregate classifier confusion counts; no cell-level identities or predictions.
- `tables/`: editable TSV exports of the two main tables and eleven supplementary tables (Table S8 has two parts) and the DOME-ML checklist.
- `scripts/` and `configs/`: original non-lineage computational implementation and frozen specifications, retained for review. These are not invoked by figure regeneration.
- `environment/`: figure dependency pins and historical software provenance.

## Computational scope

The original training and checkpoint diagnostics require the public expression inputs from GEO GSE140802, permitted non-lineage metadata, original checkpoints and the external DiffusionOT source at https://github.com/liujuntan/DiffusionOT (recorded commit `bb3c5bd0c18066e90929961e92b79ffd9abb5e86`). Obtain third-party code from its own repository under its own terms. Third-party code, raw expression matrices, cell identifiers, lineage assignments, clone/fate outcomes and model checkpoints are not redistributed here. Consequently, this release directly reproduces the figures and exposes the computational specification; it is not a self-contained full retraining release. It contains no direct lineage accuracy result and no cross-method performance ranking.

Full pairwise seed matrices were not available among the frozen aggregates. Figure 4 therefore uses comparisons against reference seed 11 and reports the corresponding medians (0.478 and 0.349). The support gate uses each seed's frozen two-dimensional latent state space and the unchanged 5% threshold. The variance ratio is a descriptive pipeline diagnostic, not calibrated trajectory-model uncertainty.

## Licenses and citation

Author-written code: MIT (`LICENSE`). Author-generated aggregate data, schematics, tables and figure assets: CC BY 4.0 (`DATA_LICENSE.md`). These licenses do not extend to third-party inputs or software. Cite this version using `CITATION.cff` and its Zenodo archive. Original data: Weinreb et al., Science 2020, DOI 10.1126/science.aaw3381, GEO GSE140802.
