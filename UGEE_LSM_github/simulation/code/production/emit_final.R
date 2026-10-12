source("make_tables.R")
f3 <- function(x) formatC(x, format="f", digits=3); f2 <- function(x) formatC(x, format="f", digits=2)
lab <- c(beta0="$\\beta_0$",beta1="$\\beta_1$",beta2="$\\beta_2$",alpha0="$\\alpha_0$",
         alpha1="$\\alpha_1$",tau0="$\\tau_0$",tau1="$\\tau_1$",tau2="$\\tau_2$")
M <- c("Ind","MB","LSM")
S <- build_table2(out=tempfile()); x <- readRDS("part2_B500.rds"); TR <- x$truth; PN <- x$param

## ---------- TABLE 2: wide, non-Gaussian, 14 columns ----------
L <- character()
for (j in 1:8) {
  if (j==1) L <- c(L,"\\multicolumn{14}{@{}l}{\\textit{Mean parameters}} \\\\")
  if (j==4) L <- c(L,"\\addlinespace","\\multicolumn{14}{@{}l}{\\textit{Variability parameters}} \\\\")
  cells <- c(f3(TR[j]),
    sapply(M, function(m) f3(S[[m]]$bias[j])),
    sapply(M, function(m) f3(S[[m]]$esd[j])),
    sapply(M, function(m) f2(S[[m]]$ase[j]/S[[m]]$esd[j])),
    sapply(M, function(m) f3(S[[m]]$cov[j])))
  L <- c(L, sprintf("%-10s & %s \\\\", lab[PN[j]], paste(cells, collapse=" & ")))
}
writeLines(L, "t2_body.tex")

## ---------- TABLE 3: efficiency, published subset ----------
## TRIMMED 2026-10-11 to match the final manuscript: Gaussian mechanism only,
## K = 5 and 10.  The K = 50 column and the non-Gaussian relative-MSE columns were
## removed from the paper and are no longer emitted.  The published Table 3 pairs
## these relative-efficiency values with LSM coverage produced separately by
## build_lsm_coverage("part3_lsm_se.rds") in make_tables.R.
## NOTE: this block was edited after the last production run and has NOT been rerun.
## Verify against the published Table 3 before relying on it.
d <- read.csv("fig1_data.csv"); d <- subset(d, ugee=="MB"); KS <- c(5,10)
RE  <- sapply(KS, function(k){ s <- subset(d, errors=="gaussian" & K==k); s$RE[match(PN,s$param)] })
L3 <- character()
for (j in 1:8) {
  if (j==1) L3 <- c(L3,"\\multicolumn{3}{@{}l}{\\textit{Mean parameters}} \\\\")
  if (j==4) L3 <- c(L3,"\\addlinespace","\\multicolumn{3}{@{}l}{\\textit{Variability parameters}} \\\\")
  L3 <- c(L3, sprintf("%-10s & %s & %s \\\\", lab[PN[j]], f3(RE[j,1]), f3(RE[j,2])))
}
writeLines(L3, "t3_re_body.tex")

## ---------- cross-run replication check for the supplement caption ----------
CV <- S$LSM$cov   # non-Gaussian LSM coverage, K=5, for the replication check below
y <- readRDS("part3_lsm_se.rds"); cl <- y$cells[["nonnormal_K5"]]
z <- cl$res; ok <- sapply(z,function(q) isTRUE(q$ok)); T3 <- cl$truth
E <- t(sapply(z[ok],`[[`,"est")); V <- t(sapply(z[ok],`[[`,"v"))
c3 <- colMeans(sweep(E-1.96*sqrt(V),2,T3,"<=") & sweep(E+1.96*sqrt(V),2,T3,">="), na.rm=TRUE)
p <- (CV+c3)/2; zs <- (CV-c3)/sqrt(2*p*(1-p)/sum(ok))
cat(sprintf("replication check: max |diff| %.3f, max |z| %.2f\n", max(abs(CV-c3)), max(abs(zs))))
cat(sprintf("T2 aSE/eSE ranges: UGEE %.2f-%.2f | LSM %.2f-%.2f (variability block LSM %.2f-%.2f)\n",
  min(sapply(c("Ind","MB"), function(m) min(S[[m]]$ase/S[[m]]$esd))),
  max(sapply(c("Ind","MB"), function(m) max(S[[m]]$ase/S[[m]]$esd))),
  min(S$LSM$ase/S$LSM$esd), max(S$LSM$ase/S$LSM$esd),
  min((S$LSM$ase/S$LSM$esd)[4:8]), max((S$LSM$ase/S$LSM$esd)[4:8])))
