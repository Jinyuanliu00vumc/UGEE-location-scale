## Part 2 at B=500, run as 10 resumable chunks of 50. Seeds are global (SEED0 + b, b = 1..500),
## so the concatenation is identical to a single B=500 run. Re-running skips finished chunks.
## LSM is fitted serially in the parent, never in a forked worker (glmmTMB/TMB + fork is risky).
## Do NOT cap OMP threads: doing so roughly halved the UGEE fit speed.
suppressMessages(Rcpp::sourceCpp("Dist_UGEE_split.cpp", cacheDir="cache4"))
source("dgp2.R"); source("fit_lsm2.R"); library(parallel)
## cores: first command-line arg, else every physical core. No editing required.
.a <- commandArgs(TRUE)
NC <- if (length(.a) >= 1 && !is.na(suppressWarnings(as.integer(.a[1])))) as.integer(.a[1]) else
        max(1L, parallel::detectCores(logical = FALSE))
CH <- 10L; NCH <- 50L; SEED0 <- 2026001L
cat(sprintf("using %d cores\n", NC))
N <- 300L; K <- 5L; TAU <- tau_for_K(K); TRUTH <- c(BETA, ALPHA, TAU)
SPEC <- list(Ind="independence", MB="model", Exch="exchangeable", AR1="ar1")
dir.create("chunks", showWarnings = FALSE)
fit_ugee <- function(Y, u, cc) {
  base <- list(Y=Y, u=u, t_star=T_STAR, var_basis=1L, estimate_rho=TRUE, tol=1e-6,
               max_step=1.0, beta_init=BETA, Eta_init=c(ALPHA,TAU), use_full_model_cov=FALSE)
  ## Rung 1 is capped at maxit=100. Diagnostics over 225 successful undamped fits: median 7
  ## iterations, 99th pct 62, max 86 -- so the cap diverts nothing that would have converged,
  ## but a doomed rung 1 now burns 100 iterations instead of 200. The middle rung is dropped:
  ## on failure go straight to the robust damped fit, which halves the expensive path again.
  ladder <- list(
    list(corr1=cc,corr2=cc,            corr3=cc, ridge=1e-6, rho_relax=1.00, maxit=100, step="undamped"),
    list(corr1=cc,corr2=cc,            corr3=cc, ridge=1e-4, rho_relax=0.30, maxit=200, step="damped"),
    list(corr1=cc,corr2="independence",corr3=cc, ridge=1e-4, rho_relax=0.30, maxit=200, step="block2ind"))
  for (L in ladder) { st <- L$step; L$step <- NULL
    f <- try(do.call(Dist_UGEE_split_cpp, c(base, L)), silent=TRUE)
    if (!inherits(f,"try-error") && isTRUE(f$converged))
      return(list(ok=TRUE, step=st, est=c(f$beta,f$Eta),
                  v=c(diag(f$vcov_beta),diag(f$vcov_eta)), it=f$iterations, b2=f$corr2_eff)) }
  list(ok=FALSE, step="failed")
}
one_ugee <- function(b) { d <- sim_prod(N, K, "nonnormal", Tau=TAU, seed=SEED0 + b)
  lapply(SPEC, function(cc) fit_ugee(d$Y, d$u, cc)) }
one_lsm  <- function(b) { d <- sim_prod(N, K, "nonnormal", Tau=TAU, seed=SEED0 + b); fit_lsm(d) }
for (ch in 1:NCH) {
  fn <- sprintf("chunks/p2_chunk%02d.rds", ch)
  if (file.exists(fn)) next
  bs <- ((ch-1)*CH + 1):(ch*CH); t0 <- Sys.time()
  r <- mclapply(bs, one_ugee, mc.cores=NC)          # UGEE in parallel
  lsm <- lapply(bs, one_lsm)                        # LSM serial, in the parent
  for (i in seq_along(r)) r[[i]]$LSM <- lsm[[i]]
  saveRDS(list(res=r, b=bs, elapsed=as.numeric(difftime(Sys.time(),t0,units="secs"))), fn)
  cat(sprintf("chunk %02d (reps %d-%d) done in %.0f s\n", ch, min(bs), max(bs),
              as.numeric(difftime(Sys.time(),t0,units="secs"))))
  flush.console()
}
## merge
files <- sprintf("chunks/p2_chunk%02d.rds", 1:NCH)
if (all(file.exists(files))) {
  parts <- lapply(files, readRDS); res <- do.call(c, lapply(parts, `[[`, "res"))
  saveRDS(list(res=res, truth=TRUTH,
    param=c("beta0","beta1","beta2","alpha0","alpha1","tau0","tau1","tau2"),
    N=N, K=K, B=length(res), Tau=TAU, Beta=BETA, Alpha=ALPHA, sigma_omega=SIGMA_W,
    t_star=T_STAR, nu_b=NU_B, nu_e=NU_E, p_u=P_U, seed0=SEED0, spec=SPEC, Zv=ortho_quad(K),
    var_basis="orthogonal quadratic", mean_basis="knot t*=2",
    errors="chi2_4 intercepts, t_5 errors",
    elapsed=sum(sapply(parts, `[[`, "elapsed")), chunked=TRUE,
    sessionInfo=sessionInfo()), "part2_B500.rds")
  cat(sprintf("MERGED -> part2_B500.rds with %d replications\n", length(res)))
}
