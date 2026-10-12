## Refit the LSM comparator ONLY, on exactly the same seeds as the production run,
## returning model-based standard errors so LSM coverage can be reported next to
## the UGEE sandwich. UGEE is not touched and part2_B500.rds / part3.rds are not
## modified: this writes new files that the assembly script joins on.
##
## Usage (from Production_v2/):
##   caffeinate -i nohup Rscript rerun_lsm_se.R 8 > lsm_se.log 2>&1 &
## The single optional argument is the number of cores (default: physical cores).
suppressMessages({library(parallel); library(glmmTMB)})
source("dgp2.R"); source("fit_lsm3.R")

a  <- commandArgs(trailingOnly = TRUE)
NC <- if (length(a) >= 1 && !is.na(as.integer(a[1]))) as.integer(a[1]) else
      max(1L, detectCores(logical = FALSE))
cat(sprintf("using %d cores; glmmTMB %s; disp multiplier %d\n",
            NC, as.character(packageVersion("glmmTMB")), DISP_SCALE))

N <- 300L
BETA <- c(1, -2, 1); ALPHA <- c(log(0.4), 1.0)

run <- function(K, B, ERR, seedf, tag) {
  TAU <- tau_for_K(K)
  t0  <- Sys.time()
  r <- mclapply(seq_len(B), function(b)
         fit_lsm(sim_prod(N, K, ERR, Tau = TAU, seed = seedf(b))),
       mc.cores = NC)
  ok <- sum(sapply(r, function(q) isTRUE(q$ok)))
  cat(sprintf("%-16s B=%3d  ok=%3d  %.0f s\n", tag, B, ok,
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  list(res = r, K = K, B = B, errors = ERR, truth = c(BETA, ALPHA, TAU),
       Zv = ortho_quad(K), disp_scale = DISP_SCALE)
}

## ---- part 2: K=5, non-Gaussian, B=500, SEED0 = 2026001 ----------------------
S2 <- 2026001L
p2 <- run(5L, 500L, "nonnormal", function(b) S2 + b, "part2 K=5")
saveRDS(c(p2, list(param = c("beta0","beta1","beta2","alpha0","alpha1","tau0","tau1","tau2"),
                   N = N, seed0 = S2, sessionInfo = sessionInfo())), "part2_lsm_se.rds")

## ---- part 3: both DGPs, K in {5,10,50}, SEED0 = 3026001 ---------------------
S3 <- 3026001L; KS <- c(5L, 10L, 50L); BS <- c(500L, 200L, 40L)
cells <- list()
for (ERR in c("gaussian", "nonnormal")) for (i in seq_along(KS)) {
  K <- KS[i]
  cells[[sprintf("%s_K%d", ERR, K)]] <-
    run(K, BS[i], ERR, local({KK <- K; function(b) S3 + 1000L*KK + b}),
        sprintf("part3 %s K=%d", ERR, K))
  saveRDS(list(cells = cells, param = c("beta0","beta1","beta2","alpha0","alpha1","tau0","tau1","tau2"),
               N = N, KS = KS, BS = BS, seed0 = S3, disp_scale = DISP_SCALE,
               sessionInfo = sessionInfo()), "part3_lsm_se.rds")
}
cat("LSM SE RERUN DONE\n")
