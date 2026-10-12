## PART 3 (efficiency). N=300, BOTH DGPs. UGEE (Ind, MB) vs LSM, paired on the same dataset.
## Tapered replication counts: B is a comma-separated list matching KS.
## usage: Rscript run_part3.R "500,200,40" 2 "5,10,50"
suppressMessages(Rcpp::sourceCpp("Dist_UGEE_split.cpp", cacheDir="cache4"))
source("dgp2.R"); source("fit_lsm2.R"); library(parallel)
a  <- commandArgs(TRUE)
BS <- as.integer(strsplit(a[1],",")[[1]])
NC <- if (length(a) >= 2 && !is.na(suppressWarnings(as.integer(a[2])))) as.integer(a[2]) else
        max(1L, parallel::detectCores(logical = FALSE))
cat(sprintf("using %d cores\n", NC))
KS <- as.integer(strsplit(a[3],",")[[1]]); stopifnot(length(BS) == length(KS))
SEED0 <- 3026001L; N <- 300L; cells <- list(); t00 <- Sys.time()
for (ERR in c("gaussian","nonnormal")) for (ki in seq_along(KS)) {
  K <- KS[ki]; B <- BS[ki]; TAU <- tau_for_K(K); TRUTH <- c(BETA, ALPHA, TAU)
  one <- function(b) {
    d <- sim_prod(N, K, ERR, Tau = TAU, seed = SEED0 + 1000L*K + b)
    g <- function(cc) {
      f <- try(Dist_UGEE_split_cpp(Y=d$Y, u=d$u, t_star=T_STAR, var_basis=1L,
           corr1=cc, corr2=cc, corr3=cc, use_full_model_cov=FALSE, estimate_rho=TRUE,
           maxit=200, tol=1e-6, beta_init=BETA, Eta_init=c(ALPHA,TAU),
           ridge=1e-6, rho_relax=1.0), silent=TRUE)
      if (inherits(f,"try-error") || !isTRUE(f$converged)) return(list(ok=FALSE))
      list(ok=TRUE, est=c(f$beta, f$Eta)) }
    list(Ind=g("independence"), MB=g("model"))
  }
  one_lsm <- function(b) fit_lsm(sim_prod(N, K, ERR, Tau = TAU, seed = SEED0 + 1000L*K + b))
  t0 <- Sys.time()
  r <- mclapply(seq_len(B), one, mc.cores=NC)        # UGEE in parallel
  lsm <- lapply(seq_len(B), one_lsm)                  # LSM serial in the parent: glmmTMB/TMB
  for (i in seq_along(r)) r[[i]]$LSM <- lsm[[i]]      # uses OpenMP and is not fork-safe
  el <- as.numeric(difftime(Sys.time(), t0, units="secs"))
  cells[[sprintf("%s_K%d", ERR, K)]] <- list(res=r, truth=TRUTH, K=K, B=B, errors=ERR,
                                             Tau=TAU, Zv=ortho_quad(K), elapsed=el)
  cat(sprintf("  %-9s K=%2d B=%3d  %.0f s\n", ERR, K, B, el))
  saveRDS(list(cells=cells, N=N, BS=BS, KS=KS, seed0=SEED0, Beta=BETA, Alpha=ALPHA,
    sigma_omega=SIGMA_W, t_star=T_STAR, nu_b=NU_B, nu_e=NU_E, p_u=P_U,
    param=c("beta0","beta1","beta2","alpha0","alpha1","tau0","tau1","tau2"),
    var_basis="orthogonal quadratic", mean_basis="knot t*=2",
    sessionInfo=sessionInfo()), "part3.rds")
}
cat(sprintf("PART3 DONE in %.0f s\n", as.numeric(difftime(Sys.time(), t00, units="secs"))))
