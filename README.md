# Code and results for "When does adapting an imported prediction model help?"

Companion archive for the article *When does adapting an imported prediction model help?
Persistence-anchored transfer of longitudinal ECG prediction across departments and devices*
(Elfatouaki H, Latif A; manuscript under review).

Archive DOI (all versions): [10.5281/zenodo.22998860](https://doi.org/10.5281/zenodo.22998860)

It contains the open-source package **ptransfer**, every analysis script, the parametric
simulation, the code that draws each figure, and the aggregated results behind every table
and figure. **No patient-level data are included** (see *Data access*).

## Contents

| Folder | What it holds |
|---|---|
| `ptransfer_package/` | The Python package (MIT licence): persistence-anchored evaluation, updating strategies with patient-grouped shrinkage, two-level bootstrap, durable thresholds, source-R² ranking. Tests in `ptransfer_package/tests`, worked example in `ptransfer_package/examples/demo_synthetic.py`. |
| `analysis/` | Scripts run on ECG-ViEW II in Kaggle notebooks: `analyse_B_v2.py` (strategies, confidence intervals, thresholds), `analyse_B_v2_intra.py` (within-department sensitivity analysis; set `INTRA_SERVICE_SEULEMENT = True`), `analyse_A_v2.py` (linear references, source R², local linear curves), `cellule_A8.py`, `cellule_S9_erreurs_absolues.py` (absolute errors in ms and degrees), `check_within_department.py`. |
| `simulation/` | `simulation.py` (20 scenarios × 3 seeds), `fig7_simulation.py` (Fig. 7), raw results `res_*.json` and `simulation_results.csv`. |
| `figures_code/` | Code for every figure of the manuscript (see the table below). |
| `data/` | Aggregated results read by the figure code (no patient-level data). |
| `figs/` | The figures, as vector PDF and PNG. |

## Figures

| Manuscript figure | Script | Data |
|---|---|---|
| Fig. 1, study design | `figures_code/fig1_study_design.py` | numbers in the script |
| Fig. 2, evaluation protocol and strategies | `figures_code/fig2_protocol_strategies.py` | none |
| Figs. 3–6 and Supplementary Fig. S2 | `figures_code/figs.py` | `data/` |
| Fig. 7, parametric simulation | `simulation/fig7_simulation.py` | `simulation/res_*.json` |
| Supplementary Fig. S1 | `figures_code/figS1.py` | `data/ev_curves.csv`, `data/mimic_curves.csv` |

## Reproduce

```
pip install -e ptransfer_package          # Python >= 3.9; numpy, scipy, pandas, scikit-learn
pytest ptransfer_package/tests            # 10 tests
python ptransfer_package/examples/demo_synthetic.py
python figures_code/fig1_study_design.py  # run from this folder; writes figs/
python figures_code/fig2_protocol_strategies.py
python figures_code/figs.py               # reads data/, writes figs/
python figures_code/figS1.py
cd simulation && python fig7_simulation.py                                   # figure from the saved runs
cd simulation && OMP_NUM_THREADS=1 python simulation.py && python fig7_simulation.py   # rerun the simulation (~20 min on 2 cores)
```

The figure scripts use the Liberation Sans font (metric-compatible with Arial); matplotlib
falls back to its default font if it is not installed.

The scripts in `analysis/` expect the ECG-ViEW II CSV under `/kaggle/input` and write to
`/kaggle/working`. Set `SMOKE_TEST = True` to run them on synthetic data first.
Analyses were run with Python 3.12 and scikit-learn 1.6.1 in Kaggle notebooks on CPU.

## Data access

- **ECG-ViEW II**: available from its maintainers (Kim et al., PLoS One 2017;12:e0176222).
- **MIMIC-IV-ECG**: available on PhysioNet to credentialed users under the PhysioNet data use agreement; it is not redistributed here.
- **PF12RED**: openly available under a CC BY 4.0 licence (Muñoz, Zenodo, doi:10.5281/zenodo.10864238; data descriptor: Munoz-Macho et al., Data Brief 2024;54:110444).

## Changes

- **1.1.0**: figures restyled and numbered as in the current manuscript; metadata corrected (authors, PF12RED access). Package and results unchanged.
- **1.0.0**: first release.

## Licence

MIT (see `LICENSE`). Please cite the article and this archive (see `CITATION.cff`).
