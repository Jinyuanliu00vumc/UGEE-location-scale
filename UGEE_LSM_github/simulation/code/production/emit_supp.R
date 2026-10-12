source("make_tables.R")
f3 <- function(x) formatC(x,format="f",digits=3); f2 <- function(x) formatC(x,format="f",digits=2)
lab <- c(beta0="$\\beta_0$",beta1="$\\beta_1$",beta2="$\\beta_2$",alpha0="$\\alpha_0$",
         alpha1="$\\alpha_1$",tau0="$\\tau_0$",tau1="$\\tau_1$",tau2="$\\tau_2$")
S <- build_table2(out=tempfile()); x <- readRDS("part2_B500.rds"); TR <- x$truth; PN <- x$param
M <- c("Ind","MB","Exch","AR1"); nm2 <- c(Ind="UGEE-Ind",MB="UGEE-MB",Exch="UGEE-Exch",AR1="UGEE-AR1")
L <- character()
for (j in 1:8) {
  if (j==1) L <- c(L,"\\multicolumn{8}{@{}l}{\\textit{Mean parameters}} \\\\")
  if (j==4) L <- c(L,"\\addlinespace","\\multicolumn{8}{@{}l}{\\textit{Variability parameters}} \\\\")
  for (i in seq_along(M)) { m <- M[i]; s <- S[[m]]
    L <- c(L, sprintf("%-10s & %-10s & %-7s & %7s & %6s & %5s & %6s & %5s \\\\",
      if (i==1) lab[PN[j]] else "", nm2[m], if (i==1) f3(TR[j]) else "",
      f3(s$bias[j]), f3(s$esd[j]), f2(s$ase[j]/s$esd[j]), f3(s$cov[j]), f3(s$conv))) }
  if (j<8) L <- c(L,"\\addlinespace") }
writeLines(L, "supp_workcorr_body.tex")
cat("emitted supplement working-correlation body\n")
