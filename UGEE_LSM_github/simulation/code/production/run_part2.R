## PART 2 (robustness). Non-Gaussian DGP, N=300, K=5. Four UGEE working correlations + LSM,
## all fitted to the SAME dataset per replication. SPEC_production_run.md 2026-09-26.
suppressMessages(Rcpp::sourceCpp("Dist_UGEE_split.cpp", cacheDir="cache4"))
source("dgp2.R"); source("fit_lsm2.R"); library(parallel)
args <- commandArgs(TRUE); B <- as.integer(args[1]); NC <- as.integer(args[2]); SEED0 <- 2026001L
N <- 300L; K <- 5L; TAU <- tau_for_K(K); TRUTH <- c(BETA, ALPHA, TAU)
SPEC <- list(Ind="independence", MB="model", Exch="exchangeable", AR1="ar1")

## fallback ladder: genuine -> relax rho -> add ridge -> block-2 independence (recorded)
fit_ugee <- function(Y, u, cc) {
  base <- list(Y=Y, u=u, t_star=T_STAR, var_basis=1L, estimate_rho=TRUE, maxit=200, tol=1e-6,
               max_step=1.0, beta_init=BETA, Eta_init=c(ALPHA,TAU), use_full_model_cov=FALSE)
  ## undamped first: rho_relax=1.0 converges in ~6 iterations vs ~33 at 0.3, to the same
  ## solution. Damping is the fallback for the cases that need it, not the default.
  ladder <- list(
    list(corr1=cc,corr2=cc,           corr3=cc, ridge=1e-6, rho_relax=1.00, step="undamped"),
    list(corr1=cc,corr2=cc,           corr3=cc, ridge=1e-6, rho_relax=0.50, step="damp0.5"),
    list(corr1=cc,corr2=cc,           corr3=cc, ridge=1e-4, rho_relax=0.30, step="damp0.3+ridge"),
    list(corr1=cc,corr2="independence",corr3=cc,ridge=1e-4, rho_relax=0.30, step="block2ind"))
  for (L in ladder) {
    st <- L$step; L$step <- NULL
    f <- try(do.call(Dist_UGEE_split_cpp, c(base, L)), silent=TRUE)
    if (!inherits(f,"try-error") && isTRUE(f$converged))
      return(list(ok=TRUE, step=st, est=c(f$beta,f$Eta),
                  v=c(diag(f$vcov_beta),diag(f$vcov_eta)), it=f$iterations, b2=f$corr2_eff))
  }
  list(ok=FALSE, step="failed")
}
one <- function(b) {
  tb <- Sys.time()
  d <- sim_prod(N, K, "nonnormal", Tau=TAU, seed=SEED0 + b)
  out <- lapply(SPEC, function(cc) fit_ugee(d$Y, d$u, cc))
  out$LSM <- fit_lsm(d)
  cat(sprintf("rep %3d  %5.1f s  rungs: %s\n", b, as.numeric(difftime(Sys.time(),tb,units="secs")),
      paste(sapply(out[names(SPEC)], function(z) if (isTRUE(z$ok)) z$step else "FAIL"), collapse=" ")),
      file="p2_progress.txt", append=TRUE)
  out
}
t0 <- Sys.time(); res <- mclapply(seq_len(B), one, mc.cores=NC)
saveRDS(list(res=res, truth=TRUTH, param=c("beta0","beta1","beta2","alpha0","alpha1","tau0","tau1","tau2"),
             N=N, K=K, B=B, Tau=TAU, Beta=BETA, Alpha=ALPHA, sigma_omega=SIGMA_W, t_star=T_STAR,
             nu_b=NU_B, nu_e=NU_E, p_u=P_U, seed0=SEED0, var_basis="orthogonal quadratic",
             mean_basis="knot t*=2", errors="chi2_4 intercepts, t_5 errors", spec=SPEC,
             Zv=ortho_quad(K), elapsed=as.numeric(difftime(Sys.time(),t0,units="secs")),
             sessionInfo=sessionInfo()), sprintf("part2_B%d.rds", B))
cat(sprintf("PART2 DONE B=%d in %.0f s\n", B, as.numeric(difftime(Sys.time(),t0,units="secs"))))
