## Assembly. Reads ONLY the .rds files and writes LaTeX. No number is transcribed by hand.
## v2 (2026-09-27): adds a documented validity screen and a convergence column.
fmt <- function(x, d=3) formatC(x, format="f", digits=d)

## ---- LSM dispersion-scale correction ---------------------------------------
## glmmTMB changed the Gaussian dispersion parameterization between 1.1.8 and
## 1.1.14. In 1.1.8 `dispformula` coefficients are on the LOG-VARIANCE scale
## (verified: fitted/true ratio 1.00). In 1.1.14, the version the production run
## used, they are on the LOG-STANDARD-DEVIATION scale (verified: true/fitted
## ratio 2.00 +/- 0.03 over 18 cell-by-parameter combinations, both DGPs, all K).
## The comparator was validated in Stage 2A against 1.1.8; rebuilding glmmTMB
## from source on the run machine silently moved it to 1.1.14.
## The correction is an exact deterministic rescale of the SAVED fits -- the fits
## themselves are unchanged, only the scale they are read on. It is applied here,
## in the assembly layer, and never to the .rds inputs.
LSM_DISP_SCALE <- 2      # tau_j = LSM_DISP_SCALE * dispformula coefficient
TAU_IDX <- 6:8

rescale_lsm <- function(q, s = LSM_DISP_SCALE) {
  if (is.null(q) || !isTRUE(q$ok) || s == 1) return(q)
  q$est[TAU_IDX] <- s * q$est[TAU_IDX]
  if (!is.null(q$v)) q$v[TAU_IDX] <- s^2 * q$v[TAU_IDX]
  q
}
apply_lsm_scale <- function(res) lapply(res, function(r) { r$LSM <- rescale_lsm(r$LSM); r })

## ---- validity screen -------------------------------------------------------
## A replication is USABLE for a given method only if all three hold:
##   (i)   the solver reported convergence (ok);
##   (ii)  every fitted variance linear predictor satisfies |eta| <= ETA_MAX.
##         The C++ clamps the variance linear predictor at +/-30, so a fit that
##         reaches the clamp is a boundary solution, not a converged interior one.
##         etas: between-subject alpha0, alpha0+alpha1; within-subject Zv %*% tau.
##   (iii) no sandwich variance entry is negative (a negative entry means the
##         meat/bread product failed to be PSD numerically).
## Counts of (ii) and (iii) are reported per method, not silently dropped.
ETA_MAX <- 10

screen_fit <- function(q, Zv, eta_max = ETA_MAX) {
  if (!isTRUE(q$ok)) return(c(ok=FALSE, bnd=NA, neg=NA))
  e   <- q$est
  eta <- c(e[4], e[4] + e[5], as.numeric(Zv %*% e[6:8]))
  bnd <- any(!is.finite(eta)) || max(abs(eta)) > eta_max
  neg <- !is.null(q$v) && any(!is.finite(q$v) | q$v < 0)
  c(ok = !(bnd || neg), bnd = bnd, neg = neg)
}

## ---------- Table 2: robustness ----------
## Join LSM model-based SEs from the rerun, if it has been done. The rerun uses
## the same seeds, so fit b in the rerun IS fit b in the production run; this is
## asserted on the point estimates before the variances are adopted.
join_lsm_se <- function(res, f = "part2_lsm_se.rds", tol = 1e-3) {
  if (!file.exists(f)) return(res)
  y <- readRDS(f); stopifnot(length(y$res) == length(res))
  d <- max(abs(sapply(seq_along(res), function(b) {
         a <- res[[b]]$LSM; c <- y$res[[b]]
         if (!isTRUE(a$ok) || !isTRUE(c$ok)) 0 else max(abs(a$est - c$est)) })))
  if (d > tol) stop(sprintf("join_lsm_se: rerun does not reproduce the saved LSM fits (max diff %.2e)", d))
  cat(sprintf("  joined LSM SEs from %s (max estimate discrepancy %.2e)\n", f, d))
  for (b in seq_along(res)) if (isTRUE(y$res[[b]]$ok)) res[[b]]$LSM$v <- y$res[[b]]$v
  res
}

## Swap the Exch/AR1 columns for the rho-capped rerun, when it exists. The rerun uses
## the same seeds, so replication b pairs with replication b; Ind, MB and LSM are left
## alone because the cap provably cannot move them (gate 1 in run_part2_cap.R).
join_capped <- function(res, f = "part2_corr_capped.rds") {
  if (!file.exists(f)) return(res)
  y <- readRDS(f); stopifnot(length(y$res) == length(res))
  for (b in seq_along(res)) for (nm in names(y$spec)) res[[b]][[nm]] <- y$res[[b]][[nm]]
  cat(sprintf("  swapped %s from %s (rho cap %.2f)\n",
      paste(names(y$spec), collapse=", "), f, y$rho_cap))
  res
}

build_table2 <- function(f = "part2_B500.rds", out = "table2.tex",
                         lsm_se = "part2_lsm_se.rds", capped = "part2_corr_capped.rds") {
  x <- readRDS(f); TR <- x$truth; PN <- x$param; Zv <- x$Zv
  lab <- c(beta0="$\\beta_0$", beta1="$\\beta_1$", beta2="$\\beta_2$",
           alpha0="$\\alpha_0$", alpha1="$\\alpha_1$",
           tau0="$\\tau_0$", tau1="$\\tau_1$", tau2="$\\tau_2$")
  x$res <- join_capped(join_lsm_se(apply_lsm_scale(x$res), lsm_se), capped)
  meth <- c(names(x$spec), "LSM")
  S <- lapply(meth, function(nm) {
    z  <- lapply(x$res, `[[`, nm)
    sc <- t(sapply(z, screen_fit, Zv = Zv))
    ok <- sc[, "ok"] %in% TRUE
    raw <- sapply(z, function(q) isTRUE(q$ok))
    E <- t(sapply(z[ok], `[[`, "est"))
    V <- if (all(sapply(z[ok], function(q) is.null(q$v)))) NULL else t(sapply(z[ok], function(q) if (is.null(q$v)) rep(NA_real_, 8) else q$v))
    r <- list(conv = mean(ok), n = sum(ok),
              n_solver = sum(raw),
              n_bnd = sum(sc[, "bnd"] %in% TRUE),
              n_neg = sum(sc[, "neg"] %in% TRUE),
              bias = colMeans(E) - TR,
              emp = apply(E, 2, var),          # empirical VARIANCE
              esd = apply(E, 2, sd))           # empirical SD
    if (!is.null(V)) {
      ## Reported on the SD scale. mean(SE) is used rather than sqrt(mean(Var))
      ## because the sandwich variance is right-skewed under the correlated
      ## working structures: a single AR(1) replication with a boundary rho-hat
      ## contributes 62% of mean(Var) but only 7% of mean(SE). The two differ by
      ## Jensen's gap, which is itself the diagnostic, so both are computed and
      ## the gap is written into the table header.
      r$est <- colMeans(V, na.rm = TRUE)                  # mean VARIANCE
      r$ase <- colMeans(sqrt(pmax(V, 0)), na.rm = TRUE)   # mean SE
      r$jensen <- sqrt(r$est) / r$ase
      lo <- E - 1.96*sqrt(pmax(V, 0)); hi <- E + 1.96*sqrt(pmax(V, 0))
      r$cov <- colMeans(sweep(lo, 2, TR, "<=") & sweep(hi, 2, TR, ">="), na.rm = TRUE)
    } else { r$est <- r$ase <- r$jensen <- r$cov <- rep(NA, 8) }
    r })
  names(S) <- meth
  L <- c("% generated by make_tables.R v2 -- do not edit by hand",
    sprintf("%% source: %s   B=%d  N=%d  K=%d", f, x$B, x$N, x$K),
    sprintf("%% validity screen: solver convergence AND max|eta| <= %g AND no negative sandwich entry", ETA_MAX),
    paste0("% excluded: ", paste(sprintf("%s(solver %d, boundary %d, negvar %d)",
           meth, x$B - sapply(S, `[[`, "n_solver"),
           sapply(S, `[[`, "n_bnd"), sapply(S, `[[`, "n_neg")), collapse="; ")),
    sprintf("%% max Jensen gap sqrt(mean Var)/mean(SE) over all cells: %s",
      paste(sprintf("%s %.2f", meth, sapply(S, function(z) max(z$jensen, na.rm = TRUE))), collapse="; ")),
    "\\begin{tabular}{l l S S S S S S S}", "\\toprule",
    "Param & Method & {Truth} & {Bias} & {ESD} & {ASE} & {ASE/ESD} & {Cov} & {Conv} \\\\", "\\midrule")
  for (j in 1:8) {
    if (j==1) L <- c(L, "\\multicolumn{9}{@{}l}{\\textit{Mean parameters}} \\\\")
    if (j==4) L <- c(L, "\\addlinespace", "\\multicolumn{9}{@{}l}{\\textit{Variability parameters}} \\\\")
    for (i in seq_along(meth)) { nm <- meth[i]; s <- S[[nm]]
      L <- c(L, sprintf("%s & %s & %s & %s & %s & %s & %s & %s & %s \\\\",
        if (i==1) lab[PN[j]] else "", nm, if (i==1) fmt(TR[j]) else "",
        fmt(s$bias[j]), fmt(s$esd[j]),
        if (is.na(s$ase[j])) "{--}" else fmt(s$ase[j]),
        if (is.na(s$ase[j])) "{--}" else fmt(s$ase[j]/s$esd[j], 2),
        if (is.na(s$cov[j])) "{--}" else fmt(s$cov[j]), fmt(s$conv, 3))) }
    if (j<8) L <- c(L, "\\addlinespace") }
  L <- c(L, "\\bottomrule", "\\end{tabular}")
  writeLines(L, out)
  cat(sprintf("wrote %s\n", out))
  for (nm in meth) { s <- S[[nm]]
    cat(sprintf("  %-4s usable %3d/%d (%.3f)  solver-fail %2d  boundary %2d  negvar %2d  max Jensen gap %.2f\n",
        nm, s$n, x$B, s$conv, x$B - s$n_solver, s$n_bnd, s$n_neg,
        if (all(is.na(s$jensen))) NA else max(s$jensen, na.rm = TRUE))) }
  invisible(S)
}

## ---------- Figure 1 data: RE (Gaussian) and relative MSE (non-Gaussian) ----------
build_fig1 <- function(f = "part3.rds", out = "fig1_data.csv") {
  x <- readRDS(f); rows <- list()
  for (cn in names(x$cells)) { cl <- x$cells[[cn]]; TR <- cl$truth
    cl$res <- apply_lsm_scale(cl$res)
    Zv <- if (!is.null(cl$Zv)) cl$Zv else x$Zv
    sok <- function(nm) { z <- lapply(cl$res, `[[`, nm)
      if (is.null(Zv)) sapply(z, function(q) isTRUE(q$ok))
      else t(sapply(z, screen_fit, Zv = Zv))[, "ok"] %in% TRUE }
    Lok <- sok("LSM")
    for (ug in c("Ind","MB")) { Uok <- sok(ug)
      both <- Lok & Uok; if (sum(both) < 5) next
      Eu <- t(sapply(lapply(cl$res,`[[`,ug)[both], `[[`, "est"))
      El <- t(sapply(lapply(cl$res,`[[`,"LSM")[both], `[[`, "est"))
      vu <- apply(Eu,2,var); vl <- apply(El,2,var)
      mu <- (colMeans(Eu)-TR)^2 + vu; ml <- (colMeans(El)-TR)^2 + vl
      rows[[length(rows)+1]] <- data.frame(errors=cl$errors, K=cl$K, B=cl$B,
        n_used=sum(both), n_excl_ugee=sum(!Uok), n_excl_lsm=sum(!Lok),
        ugee=ug, param=x$param, RE=vl/vu, relMSE=ml/mu,
        bias_ugee=colMeans(Eu)-TR, bias_lsm=colMeans(El)-TR) } }
  d <- do.call(rbind, rows); write.csv(d, out, row.names=FALSE)
  cat(sprintf("wrote %s  (%d rows)\n", out, nrow(d))); invisible(d)
}

## ---------- Diagnostic table: LSM coverage, Gaussian vs non-Gaussian ----------
## Establishes that the LSM's variability-block under-coverage is caused by
## non-Gaussianity and not by omitting the random scale effect: under the
## Gaussian DGP (omega still present, still omitted from the fit) coverage is
## at nominal. UGEE variances were not stored in part 3, so this table is
## LSM-only; the UGEE/LSM coverage comparison lives in Table 2.
build_lsm_coverage <- function(f = "part3_lsm_se.rds", out = "table_lsm_coverage.tex") {
  y <- readRDS(f); PN <- y$param
  lab <- c(beta0="$\\beta_0$", beta1="$\\beta_1$", beta2="$\\beta_2$",
           alpha0="$\\alpha_0$", alpha1="$\\alpha_1$",
           tau0="$\\tau_0$", tau1="$\\tau_1$", tau2="$\\tau_2$")
  cell <- function(e, K) { cl <- y$cells[[sprintf("%s_K%d", e, K)]]
    z <- cl$res; ok <- sapply(z, function(q) isTRUE(q$ok)); TR <- cl$truth
    E <- t(sapply(z[ok], `[[`, "est")); V <- t(sapply(z[ok], `[[`, "v"))
    lo <- E - 1.96*sqrt(V); hi <- E + 1.96*sqrt(V)
    list(cov = colMeans(sweep(lo,2,TR,"<=") & sweep(hi,2,TR,">="), na.rm=TRUE),
         r = colMeans(V, na.rm=TRUE)/apply(E,2,var), B = sum(ok)) }
  KS <- y$KS
  G <- lapply(KS, function(K) cell("gaussian", K)); N_ <- lapply(KS, function(K) cell("nonnormal", K))
  L <- c("% generated by make_tables.R v2 -- do not edit by hand",
    sprintf("%% source: %s   replications per cell: %s", f, paste(sapply(G, `[[`, "B"), collapse=", ")),
    paste0("\\begin{tabular}{l", strrep(" S", 2*length(KS)), "}"), "\\toprule",
    sprintf("& \\multicolumn{%d}{c}{Gaussian} & \\multicolumn{%d}{c}{Non-Gaussian} \\\\", length(KS), length(KS)),
    sprintf("\\cmidrule(lr){2-%d}\\cmidrule(lr){%d-%d}", 1+length(KS), 2+length(KS), 1+2*length(KS)),
    paste0("Param & ", paste(sprintf("{$K=%d$}", c(KS, KS)), collapse=" & "), " \\\\"), "\\midrule")
  for (j in 1:8) { if (j==4) L <- c(L, "\\addlinespace")
    L <- c(L, sprintf("%s & %s \\\\", lab[PN[j]],
      paste(c(sapply(G, function(g) fmt(g$cov[j])), sapply(N_, function(g) fmt(g$cov[j]))), collapse=" & "))) }
  L <- c(L, "\\bottomrule", "\\end{tabular}")
  writeLines(L, out); cat(sprintf("wrote %s\n", out))
  invisible(list(gaussian=G, nonnormal=N_))
}
