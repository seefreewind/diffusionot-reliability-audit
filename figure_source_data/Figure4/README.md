# Figure 4 frozen source package

Run `generate_figure.py` with Python, pandas, numpy and matplotlib. Outputs go to `figures/`. The generator reads this folder only. Input SHA-256 values are in `input_hashes.tsv`; exported PNG/PDF hashes are in `figure_hash.txt`.

Scientific recalculation: NO. No raw trajectories, lineage data, fitting, new seeds or uncertainty estimates are used. Values are existing frozen aggregates. Figures use DejaVu Sans (sans-serif equivalent), a white background, blue/orange/gray, vector PDF text and 300 dpi PNG. No inferential tests or confidence intervals are added because only frozen descriptive observations are available.

Panel selections: A: two positive stored variance components on a log axis. B/C: four non-reference seeds vs seed 11; self-comparison is omitted. Frozen 0.478/0.349 are medians against seed 11, not a complete 5×5 all-pairs matrix. D: the 15 initial 10,000-sample runs, replicates 1–3. Two bounded confirmation rows are excluded explicitly; seed 71 failed replicate 1.

Source files are copied without replacing the frozen originals. `source_data.tsv` is the tidy plotted-point inventory (Figure 1 uses `figure_structure.tsv`). Rounding of display labels and categorical x positions are presentation transformations only.

Historical frozen summary note: its drift-only range sentence reads 12.57–19.13%; the actual frozen paired table and manuscript report 10.65–19.13%. The preserved historical summary is not used for those plotted points.
