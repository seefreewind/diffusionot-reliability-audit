# Figure 2 frozen source package

Run `generate_figure.py` with Python, pandas, numpy and matplotlib. Outputs go to `figures/`. The generator reads this folder only. Input SHA-256 values are in `input_hashes.tsv`; exported PNG/PDF hashes are in `figure_hash.txt`.

Scientific recalculation: NO. No raw trajectories, lineage data, fitting, new seeds or uncertainty estimates are used. Values are existing frozen aggregates. Figures use DejaVu Sans (sans-serif equivalent), a white background, blue/orange/gray, vector PDF text and 300 dpi PNG. No inferential tests or confidence intervals are added because only frozen descriptive observations are available.

Panel selections: All 30 seed × transition × dt rows.

Source files are copied without replacing the frozen originals. `source_data.tsv` is the tidy plotted-point inventory (Figure 1 uses `figure_structure.tsv`). Rounding of display labels and categorical x positions are presentation transformations only.
