# CHAP Real-Data UGEE Analysis

This folder contains the code for the final real-data analysis of longitudinal systolic blood pressure (SBP) in the CHAP application.

## Primary Reported Analysis

The final manuscript primary WUGEE analysis uses the expanded pooled linear unstabilized dropout weighting model:

```r
r_cur ~ k + last_sbp + ckd + age_c + baseline_weight_c + last_weight
```

The WUGEE analysis uses the full monotone-eligible analytic sample of `N = 558` subjects. Complete-case comparison analyses use `N = 288` subjects.

## Data

Raw patient-level CHAP data are not included. The expected input file for local reruns is:

```text
CHAP_visit_level_data.csv
```

Place that file in the project working directory before rerunning the analysis scripts.

## Contents

- `code/CHAP_revised_realdata_pipeline.R`: broader revised real-data pipeline used for LSM and WUGEE specifications.
- `code/WUGEE_subj_final_pooled_external_weights.cpp`: C++ solver used for the saved pooled dropout-weight WUGEE fits, including the primary expanded pooled linear unstabilized model.
- `code/WUGEE_subj_final.cpp`: original WUGEE solver.
- `code/CC_UGEE_final.cpp`: complete-case UGEE solver used for supplemental comparison.
- `code/CHAP_final_Figs.R`: real-data figure generation script.
- `code/table3_weight_sensitivity.R`: script used to construct diagnostic sensitivity weights.
- `code/finish_table3_weight_sensitivity_from_saved.R`: script used to format saved converged sensitivity results.
