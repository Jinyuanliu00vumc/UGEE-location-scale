# Python implementation (complete-data UGEE)

A Python implementation of the complete-data UGEE estimator for the Gaussian location-scale model, with the sandwich standard errors. It is a standalone illustration of the estimator. The simulation and real-data results in the paper were produced by the R/C++ code in `simulation/` and `realdata/`, not by this folder.

## Files

- `Dist_UGEE_func_v4_SE.py`: defines `ugee_solve_FRM(y_ki_all, ui, t_star, eps=1e-5, max_iter=449)`. It first updates the mean parameters, then the variance parameters with the mean fixed, and returns the sandwich variance.
- `LSM_II_Data_Gene_Once_v4.py`: demonstration driver that simulates data, calls the solver in a loop, and writes `results300itr.json` to the current working directory.

## Inputs and outputs

- `y_ki_all`: list of K arrays of length n, one per visit, holding the outcome at visit k.
- `ui`: length-n array of a single binary between-subject covariate.
- `t_star`: knot of the piecewise-linear time trend.

The function returns a dict with:

- `theta_est`: stacked estimate of length 8, ordered (beta0, beta1, beta2, alpha0, alpha1, tau0, tau1, tau2).
- `sigma_est`: estimated variances.
- `Sigma_est`: estimated covariance matrix of `theta_est`.
- `iter_beta`, `iter_eta`: iteration counts for the mean and variance steps.
- `converged`: boolean convergence flag.

## Requirements

Python 3 with `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `statsmodels` and `IPython` (imported by the driver).

## Running the demonstration

The solver file contains no imports of its own, so it must be executed in the same namespace as the driver:

```python
exec(open("Dist_UGEE_func_v4_SE.py").read())
exec(open("LSM_II_Data_Gene_Once_v4.py").read())
```

Demo settings (set at the top of the driver): n = 100, K = 3, `t_star` = 2, beta = (1, -2, 1), alpha = (-1, 2), tau = (-0.5, 2), subject-level scale effect sd 0.5, 150 repetitions. In our test a single repetition took about 45 seconds, so the full 150 repetitions take roughly two hours. Lower `num_repetitions` for a quick check.
