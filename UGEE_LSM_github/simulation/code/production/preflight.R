## Preflight. Run this FIRST. Two minutes, catches every setup problem before a multi-hour run.
cat("R", R.version.string, "\n")
need <- c("Rcpp","RcppEigen","glmmTMB","parallel")
miss <- need[!need %in% rownames(installed.packages())]
if (length(miss)) stop("missing packages: ", paste(miss, collapse=", "),
                       "\n  install.packages(c(", paste(sprintf('"%s"',miss), collapse=","), "))")
cat("packages OK:", paste(sprintf("%s %s", need, sapply(need, function(p) as.character(packageVersion(p)))), collapse=" | "), "\n")
## glmmTMB must be built against the installed TMB, or every LSM fit is unreliable.
bt <- tryCatch(as.character(glmmTMB:::.TMB.version), error=function(e) NA)
cur <- as.character(packageVersion("TMB"))
if (!is.na(bt) && bt != cur) stop(sprintf(
  "glmmTMB was built against TMB %s but TMB %s is installed.\n  FIX: install.packages('glmmTMB', type='source')\n  then rerun preflight.", bt, cur))
cat("glmmTMB/TMB build match OK\n")
nc <- parallel::detectCores(logical = FALSE); cat("physical cores:", nc, "\n")
cat("\ncompiling Dist_UGEE_split.cpp (first time: 1-2 min) ...\n")
t0 <- Sys.time(); suppressMessages(Rcpp::sourceCpp("Dist_UGEE_split.cpp", cacheDir="cache"))
cat(sprintf("  compiled in %.0f s\n", as.numeric(difftime(Sys.time(),t0,units="secs"))))
source("dgp2.R"); source("fit_lsm2.R")
cat("\nsmoke test: 1 replication, all four working correlations + LSM ...\n")
TAU <- tau_for_K(5); d <- sim_prod(300, 5, "nonnormal", Tau = TAU, seed = 1)
for (cc in c("independence","model","exchangeable","ar1")) {
  tt <- system.time(f <- Dist_UGEE_split_cpp(Y=d$Y,u=d$u,t_star=T_STAR,var_basis=1L,corr1=cc,
    corr2=cc,corr3=cc,use_full_model_cov=FALSE,estimate_rho=TRUE,maxit=200,tol=1e-6,
    beta_init=BETA,Eta_init=c(ALPHA,TAU),ridge=1e-6,rho_relax=1.0))["elapsed"]
  cat(sprintf("  %-13s %5.1f s  conv=%s  block2=%-13s iter=%d\n", cc, tt, f$converged, f$corr2_eff, f$iterations))
}
r <- fit_lsm(d); cat(sprintf("  %-13s        conv=%s\n", "LSM", r$ok))
if (!isTRUE(r$ok)) stop("LSM comparator failed -- check glmmTMB version")
cat(sprintf("\nESTIMATES (part 2, B=500, %d cores): about %.0f min\n", nc, 500*53/nc/60))
cat(sprintf("ESTIMATES (part 3, tapered,  %d cores): about %.1f h\n", nc, (500*31+200*91+40*1510)*2/nc/3600))
cat("\nPREFLIGHT PASSED\n")
