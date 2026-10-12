# Simulation studies

## Read this first

`fit_lsm2.R` does **not** correct for the glmmTMB `dispformula` scale change. Its LSM `tau`
estimates are on the log-SD scale under current glmmTMB builds, which is half the
data-generating value. The published numbers are right because `make_tables.R` applies the
correction during assembly (`LSM_DISP_SCALE`, `rescale_lsm()`, `apply_lsm_scale()`).

**Never read `tau` from a `fit_lsm2.R` fit directly.** Go through `make_tables.R`, or use
`fit_lsm3.R`, which detects the scale at load and stops if it cannot identify it.

## Which script makes which table

| Manuscript object | Chain |
|---|---|
| Table 2, calibration | `code/calibration/`: `WUGEE_Subj_MAR.R` (weighted) and `LSM_Normal_v2.R` (complete data) |
| Table 3, efficiency | `run_part3.R` -> `part3.rds` -> `build_fig1()` -> `fig1_data.csv` -> `emit_final.R`, paired with `build_lsm_coverage()` |
| Table 4, non-Gaussian | `run_part2_chunked.R` -> `part2_B500.rds`, plus `rerun_lsm_se.R` -> `part2_lsm_se.rds`, then `emit_final.R` |
| Supp Table 6, N = 300 | `code/calibration/` chain |
| Supp Table 7, dropout | `code/calibration/` chain |
| Supp Table 8, working correlations | `run_part2_chunked.R` chain, then `emit_supp.R` |

## Run order, production chain

```bash
cd code/production
Rscript preflight.R                                  # ~2 min, must print PREFLIGHT PASSED
Rscript run_part2_chunked.R                          # ~45 min on 10 cores -> part2_B500.rds
Rscript run_part3.R "500,200,40" "" "5,10,50"        # ~5 h on 10 cores   -> part3.rds
Rscript rerun_lsm_se.R                               # -> part2_lsm_se.rds, part3_lsm_se.rds
Rscript -e 'source("make_tables.R"); build_table2("part2_B500.rds"); build_fig1("part3.rds")'
Rscript emit_final.R
Rscript emit_supp.R
```

Replication counts are 500 at K = 5, 200 at K = 10, 40 at K = 50. The paper reports K = 5 and 10
only; the K = 50 cell is kept in the run command because `build_fig1()` expects the three-cell
structure.
