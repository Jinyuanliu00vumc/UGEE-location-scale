# LSM_Normal_v2.R
# ============================================================
# Monte Carlo simulation study for complete-data UGEE.
#
# Data-generating process — IDENTICAL to WUGEE (LSM_Missing_v2.R):
#   y_{ik} = beta0 + beta1*k + beta2*(k - t*)_+ + b_i + eps_{ik}
#   sigma^2_{b_i}    = exp(alpha0 + alpha1 * u_i)
#   sigma^2_{eps,ik} = exp(tau0 + tau1*k + tau2*(k-t*)_+ + omega_i)
#   omega_i ~ N(-0.5*sigma_omega^2, sigma_omega^2)  [centered: E[exp(omega)]=1]
#
# The ONLY difference from WUGEE: no missing data (all pairs observed).
#
# TRUE PARAMETERS used as starting values to accelerate convergence.
#
# Parallel computing:
#   Unix/Mac : parallel::mclapply  (fork, shares compiled C++)
#   Windows  : parallel::parLapply (PSOCK cluster, re-sources Cpp on each worker)
#
# Summary table columns (convergent runs only):
#   param | true | mean_hat | bias | emp_var | mean_sand |
#   ratio_sand_to_emp | coverage | n_used | conv_rate
#   — matches LSM_Missing_v2.R column-for-column
#
# All numeric results rounded to 3 decimal places.
# ============================================================

## ---- 0. Setup -----------------------------------------------
cpp_file <- file.path(dirname(rstudioapi::getActiveDocumentContext()$path),
                      "UGEE_v2.cpp")
Rcpp::sourceCpp(cpp_file)

library(parallel)
library(MASS)

## ---- 1. Data-generating process (same as WUGEE) -------------
simulate_ugee_data <- function(
    n,
    K,
    t_star,
    Beta_true,    # length 3: (beta0, beta1, beta2)
    Eta_true,     # length 5: (alpha0, alpha1, tau0, tau1, tau2)
    p_u         = 0.5,
    sigma_omega = 0.2
) {
  stopifnot(length(Beta_true) == 3)
  stopifnot(length(Eta_true)  == 5)

  beta0  <- Beta_true[1]; beta1 <- Beta_true[2]; beta2 <- Beta_true[3]
  alpha0 <- Eta_true[1];  alpha1 <- Eta_true[2]
  tau0   <- Eta_true[3];  tau1  <- Eta_true[4]; tau2  <- Eta_true[5]

  # u_i ~ Bernoulli(p_u)
  ui <- rbinom(n, 1, p_u)

  # omega_i centred so E[exp(omega)] = 1
  mu_omega <- -0.5 * sigma_omega^2
  omega_i  <- rnorm(n, mean = mu_omega, sd = sigma_omega)

  # Between-subject random effect
  sigma_b2 <- exp(alpha0 + alpha1 * ui)
  b_i      <- rnorm(n, 0, sqrt(sigma_b2))

  # Outcomes — same formula as simulate_wugee_data()
  Y <- matrix(NA_real_, n, K)
  for (k in seq_len(K)) {
    mu_k       <- beta0 + beta1 * k + beta2 * max(0, k - t_star)
    sig2_eps_k <- exp(tau0 + tau1 * k + tau2 * max(0, k - t_star) + omega_i)
    Y[, k]     <- mu_k + b_i + rnorm(n, 0, sqrt(sig2_eps_k))
  }

  list(
    y_ki = lapply(seq_len(K), function(k) Y[, k]),
    ui   = ui
  )
}

## ---- 2. Single-replicate wrapper ----------------------------
one_rep_ugee <- function(b, n, K, t_star, Beta_true, Eta_true,
                         sigma_omega, eps, max_iter, cpp_file, seed_offset) {
  set.seed(b + seed_offset)

  dat <- simulate_ugee_data(
    n           = n,
    K           = K,
    t_star      = t_star,
    Beta_true   = Beta_true,
    Eta_true    = Eta_true,
    sigma_omega = sigma_omega
  )

  fit <- tryCatch(
    ugee_solve_v2(
      y_ki_all  = dat$y_ki,
      ui_R      = dat$ui,
      t_star    = t_star,
      Beta_init = Beta_true,   # start at truth
      Eta_init  = Eta_true,    # start at truth
      eps       = eps,
      max_iter  = max_iter
    ),
    error = function(e) NULL
  )

  if (is.null(fit)) return(list(converged = FALSE))

  p <- 3 + 5   # Beta(3) + Eta(5)
  theta_hat <- as.numeric(fit$theta.est)
  if (length(theta_hat) != p || !all(is.finite(theta_hat)))
    return(list(converged = FALSE))

  var_hat <- diag(as.matrix(fit$Sigma.est))
  if (!all(is.finite(var_hat)) || any(var_hat <= 0))
    return(list(converged = FALSE))

  list(
    converged = isTRUE(fit$converged),
    theta_hat = theta_hat,
    var_hat   = var_hat,
    iter_beta = fit$iter_beta,
    iter_eta  = fit$iter_eta
  )
}

## ---- 3. Summary table helper (same format as WUGEE) ---------
make_summary_table_ugee <- function(theta_hat_mat, var_hat_mat,
                                    theta_true, param_names, nsim_total) {
  n_used    <- nrow(theta_hat_mat)
  mean_hat  <- colMeans(theta_hat_mat)
  bias      <- mean_hat - theta_true
  emp_var   <- apply(theta_hat_mat, 2, var)
  mean_sand <- colMeans(var_hat_mat)
  ratio     <- mean_sand / emp_var

  z975     <- qnorm(0.975)
  se_mat   <- sqrt(var_hat_mat)
  lower    <- theta_hat_mat - z975 * se_mat
  upper    <- theta_hat_mat + z975 * se_mat
  coverage <- colMeans(
    sweep(lower, 2, theta_true, "<=") &
    sweep(upper, 2, theta_true, ">=")
  )

  tab <- data.frame(
    param             = param_names,
    true              = theta_true,
    mean_hat          = mean_hat,
    bias              = bias,
    emp_var           = emp_var,
    mean_sand         = mean_sand,
    ratio_sand_to_emp = ratio,
    coverage          = coverage,
    n_used            = n_used,
    conv_rate         = round(n_used / nsim_total, 3),
    stringsAsFactors  = FALSE,
    row.names         = NULL
  )

  num_cols <- sapply(tab, is.numeric)
  tab[num_cols] <- lapply(tab[num_cols], round, digits = 3)
  tab
}

## ---- 4. Main MC runner --------------------------------------
run_mc_ugee_v2 <- function(
    nsim        = 500,
    n           = 150,
    K           = 5,
    t_star      = 2,
    Beta_true   = c(1, -2, 1),
    Eta_true    = c(log(0.7), 1, log(0.4), 0.5, -0.5),
    sigma_omega = 0.2,
    eps         = 1e-5,
    max_iter    = 500,
    n_cores     = max(1L, parallel::detectCores() - 1L),
    seed        = 123,
    cpp_file_path = NULL
) {
  set.seed(seed)

  p           <- 8   # Beta(3) + Eta(5)
  theta_true  <- c(Beta_true, Eta_true)
  param_names <- c("beta0", "beta1", "beta2",
                   "alpha0", "alpha1", "tau0", "tau1", "tau2")

  seed_offset <- sample.int(1e6, 1)

  ## ----- Benchmark on 5 reps -----
  cat("Benchmarking on 5 reps...\n")
  bench_time <- system.time({
    for (bb in 1:5)
      one_rep_ugee(bb, n, K, t_star, Beta_true, Eta_true,
                   sigma_omega, eps, max_iter, cpp_file_path, seed_offset)
  })["elapsed"]
  sec_per_rep <- bench_time / 5
  cat(sprintf(
    "  ~%.2f sec/rep  =>  B=%d sequential: ~%.0f sec  |  parallel (%d cores): ~%.0f sec\n",
    sec_per_rep, nsim, round(sec_per_rep * nsim, 1),
    n_cores, round(sec_per_rep * nsim / n_cores, 1)))

  ## ----- Parallel or sequential -----
  cat(sprintf("Running %d MC replications (n=%d, K=%d)...\n", nsim, n, K))
  t_start   <- proc.time()["elapsed"]
  args_list <- as.list(seq_len(nsim))

  if (.Platform$OS.type == "windows" && n_cores > 1L) {
    if (is.null(cpp_file_path))
      stop("On Windows supply cpp_file_path = path to UGEE_v2.cpp")
    cl <- parallel::makeCluster(n_cores)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    local_env <- environment()
    parallel::clusterExport(
      cl,
      varlist = c("simulate_ugee_data", "one_rep_ugee",
                  "n", "K", "t_star", "Beta_true", "Eta_true",
                  "sigma_omega", "eps", "max_iter", "cpp_file_path", "seed_offset"),
      envir = local_env
    )
    parallel::clusterEvalQ(cl, {
      library(Rcpp); library(RcppEigen)
      Rcpp::sourceCpp(cpp_file_path)
    })
    results <- parallel::parLapply(cl, args_list, function(b)
      one_rep_ugee(b, n, K, t_star, Beta_true, Eta_true,
                   sigma_omega, eps, max_iter, cpp_file_path, seed_offset))
  } else {
    results <- parallel::mclapply(
      args_list,
      FUN = function(b)
        one_rep_ugee(b, n, K, t_star, Beta_true, Eta_true,
                     sigma_omega, eps, max_iter, cpp_file_path, seed_offset),
      mc.cores     = n_cores,
      mc.set.seed  = TRUE
    )
  }

  t_elapsed <- proc.time()["elapsed"] - t_start
  cat(sprintf("Done. Elapsed: %.1f sec\n", t_elapsed))

  ## ----- Extract convergent results -----
  conv_vec      <- vapply(results, function(r) isTRUE(r$converged), logical(1))
  theta_hat_mat <- do.call(rbind, lapply(results[conv_vec], `[[`, "theta_hat"))
  var_hat_mat   <- do.call(rbind, lapply(results[conv_vec], `[[`, "var_hat"))

  n_used    <- sum(conv_vec)
  conv_rate <- round(n_used / nsim, 3)
  cat(sprintf("Converged: %d / %d  (rate = %.3f)\n", n_used, nsim, conv_rate))

  if (n_used < 10)
    warning("Fewer than 10 convergent runs; results may be unreliable.")

  ## ----- Summary table -----
  tbl <- make_summary_table_ugee(theta_hat_mat, var_hat_mat,
                                  theta_true, param_names, nsim)

  cat("\n--- Beta (mean parameters) ---\n")
  print(tbl[tbl$param %in% c("beta0","beta1","beta2"), ], row.names = FALSE)
  cat("\n--- Eta (variance parameters) ---\n")
  print(tbl[tbl$param %in% c("alpha0","alpha1","tau0","tau1","tau2"), ], row.names = FALSE)

  invisible(list(
    table         = tbl,
    n_used        = n_used,
    conv_rate     = conv_rate,
    theta_true    = theta_true,
    param_names   = param_names,
    timing_sec    = t_elapsed,
    theta_hat_mat = theta_hat_mat,
    var_hat_mat   = var_hat_mat
  ))
}

## ============================================================
## 5. Run simulations
## ============================================================

## --- Quick smoke test (1 replicate) -------------------------
cat("=== Smoke test (n=50, K=5, 1 rep) ===\n")
smoke_dat <- simulate_ugee_data(
  n = 50, K = 5, t_star = 2,
  Beta_true = c(1, -2, 1),
  Eta_true  = c(log(0.7), 1, log(0.4), 0.5, -0.5)
)
fit_smoke <- ugee_solve_v2(
  y_ki_all  = smoke_dat$y_ki,
  ui_R      = smoke_dat$ui,
  t_star    = 2,
  Beta_init = c(1, -2, 1),
  Eta_init  = c(log(0.7), 1, log(0.4), 0.5, -0.5),
  eps       = 1e-5,
  max_iter  = 200
)
cat("Converged:", fit_smoke$converged,
    " | Beta:", round(fit_smoke$Beta, 3),
    " | Eta:", round(fit_smoke$Eta, 3), "\n")
cat("SE(Beta+Eta):", round(sqrt(fit_smoke$sigma.est), 3), "\n")

## --- n = 150, B = 500 ----------------------------------------
cat("\n=== n=150, B=500, K=5, t*=2 ===\n")
res_150 <- run_mc_ugee_v2(
  nsim        = 500,
  n           = 150,
  K           = 5,
  t_star      = 2,
  Beta_true   = c(1, -2, 1),
  Eta_true    = c(log(0.7), 1, log(0.4), 0.5, -0.5),
  sigma_omega = 0.2,
  eps         = 1e-5,
  max_iter    = 500,
  seed        = 123
)
saveRDS(res_150, "res_ugee_n150.rds")

## --- n = 500, B = 500 ----------------------------------------
cat("\n=== n=500, B=500, K=5, t*=2 ===\n")
res_500 <- run_mc_ugee_v2(
  nsim        = 500,
  n           = 500,
  K           = 5,
  t_star      = 2,
  Beta_true   = c(1, -2, 1),
  Eta_true    = c(log(0.7), 0.5, log(0.4), 0.5, -0.5),  # alpha1=0.5, matches WUGEE n=500
  sigma_omega = 0.2,
  eps         = 1e-5,
  max_iter    = 500,
  seed        = 456
)
saveRDS(res_500, "res_ugee_n500.rds")

## --- Side-by-side comparison helper --------------------------
## After running both UGEE and WUGEE, use this to print a merged table:
##
## ugee_tbl  <- readRDS("res_ugee_n150.rds")$table
## wugee_tbl <- readRDS("res_wugee_n150.rds")$table
## core_params <- c("beta0","beta1","beta2","alpha0","alpha1","tau0","tau1","tau2")
##
## cat("\n=== Side-by-side: n=150 ===\n")
## cat("-- UGEE (complete data) --\n")
## print(ugee_tbl[ugee_tbl$param %in% core_params, ], row.names = FALSE)
## cat("-- WUGEE (MAR missing) --\n")
## print(wugee_tbl[wugee_tbl$param %in% core_params, ], row.names = FALSE)


# # Supplement: n=300
# ## --- n = 300, B = 500 ----------------------------------------
# cat("\n=== n=300, B=500, K=5, t*=2 ===\n")
# res_300 <- run_mc_ugee_v2(
#   nsim        = 500,
#   n           = 300,
#   K           = 5,
#   t_star      = 2,
#   Beta_true   = c(1, -2, 1),
#   Eta_true    = c(log(0.7), 0.5, log(0.4), 0.5, -0.5),  # alpha1=0.5, matches WUGEE n=500
#   sigma_omega = 0.2,
#   eps         = 1e-5,
#   max_iter    = 500,
#   seed        = 456
# )
# saveRDS(res_300, "res_ugee_n300_v2.rds")
