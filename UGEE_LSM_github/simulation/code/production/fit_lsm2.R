## ============================================================================
##  WARNING, READ BEFORE USING THIS FILE'S OUTPUT DIRECTLY
##
##  glmmTMB changed the scale of `dispformula` between versions: 1.1.8 placed it
##  on the log-VARIANCE scale, later versions on the log-SD scale.  This file does
##  not detect or correct for that.  Its LSM `tau` estimates are therefore on
##  whichever scale the installed glmmTMB uses, and under a log-SD build they are
##  HALF the data-generating values.
##
##  The published numbers are correct because `make_tables.R` applies the
##  correction in the assembly layer (`LSM_DISP_SCALE`, `rescale_lsm()`,
##  `apply_lsm_scale()`).  Fitted objects produced here are NOT corrected.
##
##  Do not read tau from these fits directly.  Either go through make_tables.R,
##  or use fit_lsm3.R, which detects the scale at load and stops if it cannot.
## ============================================================================

## LSM comparator, validated Stage 2A. Knot mean, orthogonal-quadratic dispersion.
suppressMessages(library(glmmTMB))
fit_lsm <- function(d) {
  f <- try(glmmTMB(y ~ k + post + diag(0 + factor(u) | id),
                   dispformula = ~ c1 + c2, data = d$long, REML = FALSE), silent = TRUE)
  if (inherits(f, "try-error")) return(list(ok = FALSE))
  ok <- isTRUE(f$sdr$pdHess) && f$fit$convergence == 0
  bh <- unname(fixef(f)$cond); dh <- unname(fixef(f)$disp)   # disp coefs are on log-VARIANCE scale
  sdv <- attr(VarCorr(f)$cond$id, "stddev")                   # cond REs are on the SD scale
  if (length(sdv) != 2 || length(bh) != 3 || length(dh) != 3) return(list(ok = FALSE))
  a0 <- 2*log(sdv[1]); a1 <- 2*log(sdv[2]) - 2*log(sdv[1])    # -> log-variance scale
  list(ok = ok, est = unname(c(bh, a0, a1, dh)))
}
