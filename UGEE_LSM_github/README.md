# Pairwise Discordance Regression for Joint Mean-Variability Inference

Code accompanying *U-statistics-based Pairwise Discordance Regression for Joint
Mean-Variability Inference in Longitudinal Studies with Missing Data*.

## Layout

| Folder | Contents |
|---|---|
| `simulation/` | Simulation studies: Tables 2, 3, 4 and Supplemental Tables 6, 7, 8 |
| `realdata/` | CHAP blood-pressure application: Table 5, Figure 1, Supplemental Table 10 |
| `python/` | Standalone Python implementation of the complete-data UGEE solver (illustration only; see its README) |

## Data availability

The blood pressure data analysed in the paper contain protected health information and
**cannot be shared**. `realdata/` expects a local file `CHAP_visit_level_data.csv` that is not
distributed here; see `realdata/README.md` for the expected format.

The simulation studies generate their own data and require no external input.

## Reproducing the reported results

Saved fitted objects are **not** included, so the tables are produced by rerunning the
pipelines. See `simulation/README.md` for run order and timings; the efficiency run takes
several hours on a multi-core machine.

The runtime comparison in Supplemental Table 9 has no accompanying script in this repository.

## Requirements

R with `Rcpp`, `RcppEigen` and `glmmTMB`; Python 3.x for `python/`.
