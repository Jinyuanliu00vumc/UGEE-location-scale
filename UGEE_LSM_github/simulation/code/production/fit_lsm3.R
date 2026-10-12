## LSM comparator v3 (2026-09-27).
## Changes from v2:
##   (1) The dispersion scale is DETECTED at load time, not assumed. glmmTMB
##       changed the Gaussian dispersion parameterization between 1.1.8
##       (log-variance) and 1.1.14 (log-SD); v2 hard-coded the 1.1.8 convention
##       and silently halved every tau on a machine running 1.1.14.
##   (2) Model-based standard errors are returned for all eight parameters, so
##       LSM coverage can be compared against the UGEE sandwich.
suppressMessages(library(glmmTMB))

## ---- (1) runtime detection of the dispformula scale ------------------------
.detect_disp_scale <- function(seed = 20260927, n = 3000) {
  set.seed(seed)
  k  <- rep(1:5, n)
  t0 <- -0.9; t1 <- 0.35                       # TRUE LOG-VARIANCE coefficients
  y  <- rnorm(length(k), 0, sqrt(exp(t0 + t1*k)))
  f  <- glmmTMB(y ~ 1, dispformula = ~ k, data = data.frame(y, k), REML = FALSE)
  r  <- c(t0, t1) / unname(fixef(f)$disp)      # 1 => log-variance, 2 => log-SD
  s  <- if (all(abs(r - 1) < 0.15)) 1 else if (all(abs(r - 2) < 0.30)) 2 else NA_real_
  if (is.na(s)) stop(sprintf(
    "fit_lsm3: cannot identify the glmmTMB dispersion scale (ratios %s, version %s). Stop and investigate.",
    paste(round(r, 3), collapse = ", "), as.character(packageVersion("glmmTMB"))))
  message(sprintf("fit_lsm3: glmmTMB %s, dispformula on log-%s scale, tau multiplier = %d",
                  as.character(packageVersion("glmmTMB")), if (s == 1) "variance" else "SD", s))
  s
}
DISP_SCALE <- .detect_disp_scale()
stopifnot(DISP_SCALE %in% c(1, 2))

## ---- (2) the fit -----------------------------------------------------------
fit_lsm <- function(d) {
  f <- try(glmmTMB(y ~ k + post + diag(0 + factor(u) | id),
                   dispformula = ~ c1 + c2, data = d$long, REML = FALSE), silent = TRUE)
  if (inherits(f, "try-error")) return(list(ok = FALSE))
  ok <- isTRUE(f$sdr$pdHess) && f$fit$convergence == 0

  bh  <- unname(fixef(f)$cond)
  dh  <- DISP_SCALE * unname(fixef(f)$disp)
  sdv <- attr(VarCorr(f)$cond$id, "stddev")
  if (length(sdv) != 2 || length(bh) != 3 || length(dh) != 3) return(list(ok = FALSE))
  a0 <- 2*log(sdv[1]); a1 <- 2*log(sdv[2]) - 2*log(sdv[1])

  ## model-based variances
  v <- rep(NA_real_, 8)
  V <- try(vcov(f, full = TRUE), silent = TRUE)
  if (!inherits(V, "try-error")) {
    nmv <- colnames(V)
    ib <- seq_len(3)                                   # cond fixed effects, in order
    id_ <- grep("^d~", nmv); if (!length(id_)) id_ <- which(nmv %in% c("(Intercept)","c1","c2")[0])
    it <- grep("^theta", nmv)
    if (length(ib) == 3) v[1:3] <- diag(V)[ib]
    if (length(id_) == 3) v[6:8] <- DISP_SCALE^2 * diag(V)[id_]
    if (length(it) == 2) {
      ## theta for diag() is on the log-SD scale; assert before using it
      if (max(abs(exp(unname(f$fit$par[names(f$fit$par) == "theta"])) - unname(sdv))) < 1e-6) {
        J  <- matrix(c(2, 0, -2, 2), 2, 2, byrow = TRUE)
        v[4:5] <- diag(J %*% V[it, it] %*% t(J))
      }
    }
  }
  list(ok = ok, est = unname(c(bh, a0, a1, dh)), v = v, disp_scale = DISP_SCALE)
}
