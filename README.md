# Pulse-selection stability under HPRF and false-return interference

Research code and frozen numerical evidence prepared for a proposed *Optical
Engineering* manuscript. The final manuscript title and citation will be
added after author confirmation.

## What is in this repository

- `code/simulation/`: complete Stage 1–11 Python source, build scripts,
  contracts, tests, taskbooks, and documentation.
- `data/simulation/artifacts/`: all configuration, Pilot, Formal, Bootstrap,
  export, chunk, manifest, and audit artifacts from the frozen Stage 11 tree.
- `code/plot_results.py`: plotting code for seven manuscript figures.
- `data/figure_source_data/`: eight graph-ready tables and their SHA256
  manifest; each table traces to a frozen Stage 11 artifact.
- `data/figure_qa_reference/`: original figure QA records and previews.
- `code/reproduce.py`: checks every data file and regenerates all seven figures.

Historical Stage 1–10 code and data are contained in the Stage 11 project.
Duplicate dated ZIP/RAR snapshots, virtual environments, bytecode, and caches
are not tracked. The packaged plotting script changes only input/output paths
so data are read from `data/` and new figures are saved in `results/`.

## Reproduce the figures

Use Python 3.12. Install exact plotting dependencies once:

```bash
python -m pip install -r environment/requirements.txt
python code/reproduce.py --root .
```

The script verifies 5,385 data-file SHA256 hashes, 23 Stage 11 artifact
hashes, and eight figure-source provenance links. It writes seven PDF/SVG/PNG/
TIFF figure sets, a combined PDF, previews, and a JSON verification report to
`results/`. GitHub Actions runs the same command on Linux for each push and
pull request. The first GitHub Actions run must be checked after upload.

The default command **does not** rerun the 190,000-row Formal simulation or
the B=2000 Bootstrap. It regenerates figures from the frozen evidence. Source
implementations, contracts, raw metrics, distributions, and audit data are
present for separate full recomputation. Stage 11 was expressly designed as
a read-only evidence export after the earlier computations were frozen.

## Evidence and limitations

- `data/simulation/artifacts/stage11/status.json` records Stage 11 PASS and
  20/20 Stage 11 tests passing. It also records 423 passing historical tests
  and three authorized legacy stage-boundary failures. This is not a clean
  full-suite pass.
- Six of 216 preregistered precision cells miss their targets. The source
  audit and supplementary precision figure preserve those outcomes.
- Repository checks establish file integrity and figure regeneration. They
  do not establish that every numerical claim and caption in a manuscript
  agrees with the frozen results; that comparison still needs completion.

## Citation and reuse

Author names, affiliations, final title, and code/data licenses require
author confirmation before a public release. Cite a versioned GitHub release
or commit in the manuscript. GitHub does not itself mint a DOI; if a DOI is
required, archive the released version with a DOI issuing repository such as
Zenodo.
