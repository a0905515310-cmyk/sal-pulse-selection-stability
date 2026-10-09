# Improving guidance-pulse selection stability across cycles using a hybrid-coding-assisted dual-feature criterion for semi-active laser lock-on tracking

Research code and frozen numerical evidence for a proposed *Optical Engineering*
manuscript.

## Authors

| Author | Affiliation |
| --- | --- |
| Zhen Zhang | a |
| Ailing Tian (corresponding author) | a |
| Changyuan Wang | b |
| Bingcai Liu | a |
| Hongjun Wang | a |
| Xueliang Zhu | a |
| Xinmeng Fang | a |
| Juan Du | c |
| Jialin Dang | a |
| Jintao Xu | c |

Correspondence: Ailing Tian, [ailintian@xatu.edu.cn](mailto:ailintian@xatu.edu.cn).

## Affiliations

- a. Shaanxi Province Key Laboratory of Thin Films Technology and Optical Test,
  Xi’an Technological University, Xi’an, Shaanxi Province 710021, China
- b. College of Computer Science, Xi’an Technological University, Xi’an 710021,
  China
- c. Xi’an Zhongke Xunjie Optoelectronic Technology Co., Ltd., Xi’an, Shaanxi
  Province 710000, China

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
pull request.

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

Code, scripts, workflow files, and documentation are licensed under the
[MIT License](LICENSE). Files under `data/` are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-DATA).
These scopes are separate; MIT does not govern the datasets, and CC BY 4.0
does not govern the source code. When redistributing data, credit the authors,
link to the license, and indicate if changes were made.

Cite a versioned GitHub release or commit in the manuscript. GitHub does not
itself mint a DOI; if a DOI is required, archive the released version with a
DOI issuing repository such as Zenodo. The repository's `CITATION.cff` records
the author list and affiliations supplied for the manuscript.
