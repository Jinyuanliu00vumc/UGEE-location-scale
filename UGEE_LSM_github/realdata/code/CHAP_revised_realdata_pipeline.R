## ============================================================
## CHAP revised real-data pipeline: LSM + WUGEE specifications
## ============================================================
## Primary manuscript analysis uses baseline weight + change from baseline.
## The personal-mean/deviation specification is retained only as an internal
## sensitivity output and is not used for manuscript interpretation.
## Both UGEE solvers use absolute between-subject age/weight differences
## through WUGEE_subj_v.cpp and WUGEE_v3_v.cpp.
## ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(Rcpp)
  library(glmmTMB)
  library(ggplot2)
  library(patchwork)
  library(parallel)
})

if (requireNamespace("rstudioapi", quietly = TRUE)) {
  ctx <- tryCatch(rstudioapi::getActiveDocumentContext(), error = function(e) NULL)
  if (!is.null(ctx) && nzchar(ctx$path)) setwd(dirname(ctx$path))
}

K_WIN <- 10L
EPS_UGEE <- 1e-5
MAX_ITER <- 999L
N_CORES <- max(1L, parallel::detectCores(logical = FALSE) - 1L)
OUT_DIR <- "revised_outputs"
dir.create(OUT_DIR, showWarnings = FALSE)

fmt_p <- function(p) ifelse(is.na(p), NA_character_, ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))

build_summary <- function(theta_est, se_vec, param_names, component) {
  z_stat <- theta_est / se_vec
  p_val <- 2 * pnorm(abs(z_stat), lower.tail = FALSE)
  data.frame(
    Component = component,
    Term = param_names,
    Estimate = theta_est,
    Std.Error = se_vec,
    Wald_z = z_stat,
    P_value = p_val,
    stringsAsFactors = FALSE
  )
}

classify_pattern <- function(obs_vec) {
  if (all(obs_vec == 1L)) return("complete")
  if (all(obs_vec == 0L)) return("all_missing")
  first_miss <- min(which(obs_vec == 0L))
  if (any(obs_vec[seq(first_miss, length(obs_vec))] == 1L)) return("intermittent")
  "monotone_dropout"
}

to_ki_list <- function(panel, col_sym, n, K = K_WIN) {
  wide <- panel %>%
    select(episode, visit_num, val = {{ col_sym }}) %>%
    pivot_wider(names_from = visit_num, values_from = val) %>%
    arrange(episode)
  mat <- as.matrix(wide[, -1])
  stopifnot(nrow(mat) == n, ncol(mat) == K)
  lapply(seq_len(K), function(j) as.numeric(mat[, j]))
}

get_subj_vec <- function(panel, col_sym) {
  panel %>% filter(visit_num == 1L) %>% arrange(episode) %>% pull({{ col_sym }})
}

prepare_panel <- function() {
  dat_raw <- read.csv("CHAP_visit_level_data.csv", stringsAsFactors = FALSE) %>%
    mutate(
      visitdt = as.Date(visitdt),
      ckd = as.integer(ckd),
      post_chap = as.integer(post_chap),
      sbp_mono_miss = as.integer(sbp_mono_miss),
      obs_sbp = as.integer(!is.na(sbp))
    )

  miss_class <- dat_raw %>%
    arrange(episode, visit_num) %>%
    group_by(episode) %>%
    summarize(miss_pattern = classify_pattern(obs_sbp), .groups = "drop")

  mono_ids <- miss_class %>%
    filter(miss_pattern %in% c("complete", "monotone_dropout")) %>%
    pull(episode)

  panel_raw <- dat_raw %>%
    filter(episode %in% mono_ids) %>%
    arrange(episode, visit_num)

  ep_order_vec <- sort(unique(panel_raw$episode))
  panel_raw %>%
    mutate(episode = factor(episode, levels = ep_order_vec)) %>%
    arrange(episode, visit_num)
}

add_weight_spec <- function(panel_raw, spec) {
  age_grand_mean <- mean(panel_raw$age_bl[panel_raw$visit_num == 1L], na.rm = TRUE)
  ep_weight_mean <- panel_raw %>%
    filter(!is.na(weight_lb)) %>%
    group_by(episode) %>%
    summarize(ep_mean_wt = mean(weight_lb, na.rm = TRUE), .groups = "drop")
  wt_grand_mean <- mean(ep_weight_mean$ep_mean_wt, na.rm = TRUE)

  baseline_weight <- panel_raw %>%
    arrange(episode, visit_num) %>%
    group_by(episode) %>%
    summarize(baseline_weight_raw = weight_lb[visit_num == min(visit_num)][1], .groups = "drop")
  baseline_wt_grand_mean <- mean(baseline_weight$baseline_weight_raw, na.rm = TRUE)

  out <- panel_raw %>%
    left_join(ep_weight_mean, by = "episode") %>%
    left_join(baseline_weight, by = "episode") %>%
    mutate(
      age_c = age_bl - age_grand_mean,
      mean_weight_lb = ep_mean_wt - wt_grand_mean,
      dev_weight_lb = weight_lb - ep_mean_wt,
      baseline_weight_lb = baseline_weight_raw - baseline_wt_grand_mean,
      weight_change_from_baseline_lb = weight_lb - baseline_weight_raw
    )

  if (spec == "personal_mean") {
    out <- out %>% mutate(weight_between = mean_weight_lb, weight_within = dev_weight_lb)
  } else if (spec == "baseline_change") {
    out <- out %>% mutate(weight_between = baseline_weight_lb, weight_within = weight_change_from_baseline_lb)
  } else {
    stop("Unknown spec")
  }

  attr(out, "centering") <- list(
    age_grand_mean = age_grand_mean,
    wt_grand_mean = wt_grand_mean,
    baseline_wt_grand_mean = baseline_wt_grand_mean
  )
  out
}

validate_spec <- function(panel, spec) {
  subject_summary <- panel %>%
    group_by(episode) %>%
    summarize(
      baseline_weight_n = n_distinct(na.omit(baseline_weight_raw)),
      baseline_change_at_v1 = weight_change_from_baseline_lb[visit_num == 1L][1],
      mean_dev = mean(dev_weight_lb, na.rm = TRUE),
      .groups = "drop"
    )

  pair_check <- panel %>% filter(visit_num == 1L) %>% arrange(episode)
  age_abs <- as.vector(dist(pair_check$age_c, method = "manhattan"))
  weight_abs <- as.vector(dist(pair_check$weight_between, method = "manhattan"))

  checks <- data.frame(
    check = c(
      "absolute age differences nonnegative",
      "absolute weight differences nonnegative",
      "baseline weight constant within subject",
      "change from baseline zero at visit 1",
      "personal-mean deviations average zero within subject",
      "finite model covariates"
    ),
    passed = c(
      all(age_abs >= 0, na.rm = TRUE),
      all(weight_abs >= 0, na.rm = TRUE),
      all(subject_summary$baseline_weight_n == 1L),
      max(abs(subject_summary$baseline_change_at_v1), na.rm = TRUE) < 1e-8,
      max(abs(subject_summary$mean_dev), na.rm = TRUE) < 1e-8,
      all(is.finite(panel$age_c[panel$visit_num == 1L])) &&
        all(is.finite(panel$weight_between[panel$visit_num == 1L])) &&
        all(is.finite(panel$ckd[panel$visit_num == 1L]))
    ),
    detail = c(
      sprintf("min=%.6f max=%.6f", min(age_abs, na.rm = TRUE), max(age_abs, na.rm = TRUE)),
      sprintf("min=%.6f max=%.6f", min(weight_abs, na.rm = TRUE), max(weight_abs, na.rm = TRUE)),
      sprintf("max distinct baseline weights=%d", max(subject_summary$baseline_weight_n, na.rm = TRUE)),
      sprintf("max abs baseline change at visit 1=%.6g", max(abs(subject_summary$baseline_change_at_v1), na.rm = TRUE)),
      sprintf("max abs subject mean deviation=%.6g", max(abs(subject_summary$mean_dev), na.rm = TRUE)),
      sprintf("rows=%d episodes=%d", nrow(panel), n_distinct(panel$episode))
    ),
    stringsAsFactors = FALSE
  )
  write.csv(checks, file.path(OUT_DIR, paste0("validation_", spec, ".csv")), row.names = FALSE)
  if (!all(checks$passed)) stop(sprintf("Validation failed for %s", spec))
  checks
}

fit_lsm_spec <- function(panel, spec) {
  dat_final <- panel %>% filter(!is.na(sbp))
  form_mu <- sbp ~ visit_num + I(visit_num^2) + age_c + weight_between + weight_within + ckd + (1 | mrn)
  form_disp <- ~ visit_num + I(visit_num^2) + age_c + weight_between + weight_within + ckd

  fit <- glmmTMB(
    formula = form_mu,
    dispformula = form_disp,
    data = dat_final,
    family = t_family(link = "identity")
  )

  coef_summary <- summary(fit)$coefficients
  cond <- as.data.frame(coef_summary$cond)
  disp <- as.data.frame(coef_summary$disp)
  cond$Term <- rownames(cond)
  disp$Term <- rownames(disp)
  names(cond)[1:4] <- c("Estimate", "Std.Error", "Wald_z", "P_value")
  names(disp)[1:4] <- c("Estimate", "Std.Error", "Wald_z", "P_value")
  lsm_summary <- bind_rows(
    cond %>% mutate(Component = "Location"),
    disp %>% mutate(Component = "Scale")
  ) %>% select(Component, Term, Estimate, Std.Error, Wald_z, P_value)

  write.csv(lsm_summary, file.path(OUT_DIR, paste0("lsm_summary_", spec, ".csv")), row.names = FALSE)
  saveRDS(fit, file.path(OUT_DIR, paste0("lsm_fit_", spec, ".rds")))

  fitted_sbp <- as.numeric(predict(fit, type = "response"))
  res_raw <- dat_final$sbp - fitted_sbp
  disp_hat <- as.numeric(predict(fit, type = "disp"))
  sd_hat <- exp(disp_hat)
  diag_df <- dat_final %>%
    mutate(fitted_val = fitted_sbp, res_raw = res_raw, res_std = res_raw / sd_hat, obs_sbp = sbp)
  write.csv(diag_df, file.path(OUT_DIR, paste0("lsm_diagnostics_", spec, ".csv")), row.names = FALSE)

  sqrt_abs_res <- sqrt(abs(diag_df$res_std))
  r2_val <- cor(diag_df$obs_sbp, diag_df$fitted_val, use = "complete.obs")^2

  fig_d1 <- ggplot(diag_df, aes(fitted_val, res_std)) +
    geom_point(alpha = 0.25, size = 0.8, colour = "steelblue") +
    geom_hline(yintercept = 0, colour = "red") +
    geom_smooth(method = "loess", se = TRUE, colour = "darkorange", fill = "orange") +
    labs(title = "D1 - Standardized residuals vs fitted", x = "Fitted SBP", y = "Standardized residuals")

  fig_d2 <- ggplot(diag_df, aes(sample = res_std)) +
    stat_qq(alpha = 0.35, size = 0.8, colour = "steelblue") +
    stat_qq_line(colour = "red") +
    labs(title = "D2 - Q-Q plot", x = "Theoretical quantiles", y = "Sample quantiles")

  fig_d3 <- ggplot(diag_df, aes(fitted_val, sqrt_abs_res)) +
    geom_point(alpha = 0.25, size = 0.8, colour = "steelblue") +
    geom_smooth(method = "loess", se = TRUE, colour = "darkorange", fill = "orange") +
    labs(title = "D3 - Scale-location", x = "Fitted SBP", y = expression(sqrt("|Standardized residuals|")))

  fig_d4 <- ggplot(diag_df, aes(factor(visit_num), res_std)) +
    geom_jitter(alpha = 0.20, size = 0.6, width = 0.18, colour = "steelblue") +
    geom_boxplot(alpha = 0, outlier.shape = NA, colour = "grey35") +
    geom_hline(yintercept = 0, colour = "red") +
    labs(title = "D4 - Residuals by visit", x = "Visit", y = "Standardized residuals")

  fig_d5 <- ggplot(diag_df, aes(res_std)) +
    geom_histogram(aes(y = after_stat(density)), bins = 45, fill = "steelblue", colour = "white") +
    stat_function(fun = dnorm, args = list(mean = mean(diag_df$res_std), sd = sd(diag_df$res_std)), colour = "red") +
    labs(title = "D5 - Residual distribution", x = "Standardized residuals", y = "Density")

  fig_d6 <- ggplot(diag_df, aes(fitted_val, obs_sbp)) +
    geom_point(alpha = 0.20, size = 0.7, colour = "steelblue") +
    geom_abline(slope = 1, intercept = 0, colour = "red") +
    annotate("text", x = -Inf, y = Inf, label = sprintf("r^2 = %.3f", r2_val), hjust = -0.15, vjust = 1.3, size = 3.5) +
    labs(title = "D6 - Observed vs fitted", x = "Fitted SBP", y = "Observed SBP")

  fig_diag <- (fig_d1 | fig_d2) / (fig_d3 | fig_d4) / (fig_d5 | fig_d6) +
    plot_annotation(
      title = paste("LSM diagnostics:", spec),
      subtitle = paste0(
        "Location: SBP ~ visit + visit^2 + age_c + ",
        ifelse(spec == "baseline_change", "baseline_weight + weight_change", "mean_weight + weight_deviation"),
        " + CKD + (1|mrn)\n",
        "Scale: same fixed-effect covariates"
      )
    )
  ggsave(file.path(OUT_DIR, paste0("lsm_diagnostics_", spec, ".pdf")), fig_diag, width = 7, height = 11)
  if (spec == "baseline_change") {
    ggsave(file.path(OUT_DIR, "glmmTMBDiag_baseline_change.pdf"), fig_diag, width = 7, height = 11)
  }

  list(fit = fit, summary = lsm_summary, diagnostics = diag_df)
}

fit_wugee_spec <- function(panel, spec) {
  n <- n_distinct(panel$episode)
  y_ki_all <- to_ki_list(panel, sbp, n)
  r_indiv_all <- to_ki_list(panel, obs_sbp, n)
  R_pair_all <- lapply(seq_len(K_WIN), function(j) {
    ri <- r_indiv_all[[j]]
    outer(ri, ri, FUN = function(a, b) as.numeric(a == 1 & b == 1))
  })

  age_c_R <- get_subj_vec(panel, age_c)
  wt_between_R <- get_subj_vec(panel, weight_between)
  ckd_R <- get_subj_vec(panel, ckd)

  wt_within_ki_all <- to_ki_list(panel, weight_within, n)
  wt_within_ki_all <- lapply(wt_within_ki_all, function(v) { v[is.na(v)] <- 0.0; v })

  panel_cc <- panel %>% filter(!is.na(sbp))
  lm_init <- lm(sbp ~ visit_num + I(visit_num^2) + age_c + weight_between + weight_within + ckd, data = panel_cc)
  Beta_init <- as.numeric(coef(lm_init))
  if (length(Beta_init) != 7L) Beta_init <- c(Beta_init, rep(0, 7L - length(Beta_init)))
  var_sbp <- var(panel_cc$sbp, na.rm = TRUE)
  Eta_init <- c(log(var_sbp / 2), rep(0, 4), log(var_sbp / 4), rep(0, 3))
  Xi_init <- rep(0, 2L * (K_WIN - 1L))
  Gamma_init <- rep(0, 2L * (K_WIN - 1L))

  t_subj <- system.time({
    fit_subj <- wugee_subj_v(
      y_ki_all = y_ki_all,
      r_indiv_all = r_indiv_all,
      age_c_R = age_c_R,
      wt_mean_R = wt_between_R,
      wt_dev_ki_all = wt_within_ki_all,
      ckd_R = ckd_R,
      Beta_init = Beta_init,
      Eta_init = Eta_init,
      Xi_init = Xi_init,
      eps = EPS_UGEE,
      max_iter = MAX_ITER
    )
  })

  t_pair <- system.time({
    fit_pair <- wugee_pair_v(
      y_ki_all = y_ki_all,
      R_pair_all = R_pair_all,
      age_c_R = age_c_R,
      wt_mean_R = wt_between_R,
      wt_dev_ki_all = wt_within_ki_all,
      ckd_R = ckd_R,
      Beta_init = Beta_init,
      Eta_init = Eta_init,
      Gamma_init = Gamma_init,
      eps = EPS_UGEE,
      max_iter = MAX_ITER
    )
  })

  term_between <- if (spec == "personal_mean") "mean_weight_lb" else "baseline_weight_lb"
  term_within <- if (spec == "personal_mean") "dev_weight_lb" else "weight_change_from_baseline_lb"
  param_names <- c(
    "beta0_intercept", "beta1_visit_num", "beta2_visit_num_sq", "beta3_age_c",
    paste0("beta4_", term_between), paste0("beta5_", term_within), "beta6_ckd",
    "alpha0", "alpha1_abs_age_diff", paste0("alpha2_abs_", term_between, "_diff"),
    "alpha3_ckd_discordant", "alpha4_both_ckd",
    "tau0", "tau1_visit_num", "tau2_visit_num_sq", paste0("tau3_", term_within)
  )
  component <- c(rep("Marginal Mean", 7), rep("Between-subject Heterogeneity", 5), rep("Within-subject Variability", 4))
  p_BE <- length(param_names)

  summary_subj <- build_summary(c(fit_subj$Beta, fit_subj$Eta), sqrt(diag(fit_subj$Sigma.theta)), param_names, component)
  summary_pair <- build_summary(c(fit_pair$Beta, fit_pair$Eta), sqrt(diag(fit_pair$Sigma.theta)[seq_len(p_BE)]), param_names, component)
  comparison <- summary_subj %>%
    transmute(Component, Term, Subj_Est = Estimate, Subj_SE = Std.Error, Subj_p = P_value) %>%
    left_join(summary_pair %>% transmute(Term, Pair_Est = Estimate, Pair_SE = Std.Error, Pair_p = P_value), by = "Term")

  write.csv(summary_subj, file.path(OUT_DIR, paste0("wugee_subject_summary_", spec, ".csv")), row.names = FALSE)
  write.csv(summary_pair, file.path(OUT_DIR, paste0("wugee_pair_summary_", spec, ".csv")), row.names = FALSE)
  write.csv(comparison, file.path(OUT_DIR, paste0("wugee_subject_pair_comparison_", spec, ".csv")), row.names = FALSE)
  saveRDS(fit_subj, file.path(OUT_DIR, paste0("wugee_subject_fit_", spec, ".rds")))
  saveRDS(fit_pair, file.path(OUT_DIR, paste0("wugee_pair_fit_", spec, ".rds")))

  conv <- data.frame(
    spec = spec,
    model = c("subject", "pairwise"),
    converged = c(fit_subj$converged, fit_pair$converged),
    iter = c(fit_subj$iter, fit_pair$iter),
    seconds = c(unname(t_subj["elapsed"]), unname(t_pair["elapsed"])),
    stringsAsFactors = FALSE
  )
  write.csv(conv, file.path(OUT_DIR, paste0("wugee_convergence_", spec, ".csv")), row.names = FALSE)

  list(subject = fit_subj, pair = fit_pair, subject_summary = summary_subj, pair_summary = summary_pair, comparison = comparison, convergence = conv)
}

write_latex_table <- function(df, file, caption, label) {
  rows <- apply(df, 1, function(r) {
    paste0(r[["Component"]], " & ", r[["Parameter"]], " & ", r[["Estimate"]], " & ", r[["Std. Error"]], " & ", r[["Wald z"]], " & ", r[["p-value"]], "\\\\")
  })
  out <- c(
    "\\begin{table}[ht]",
    "\\centering",
    paste0("\\caption{", caption, "}"),
    paste0("\\label{", label, "}"),
    "\\begin{tabular}{llrrrr}",
    "\\toprule",
    "Component & Parameter & Estimate & Std. Error & Wald $z$ & $p$-value\\\\",
    "\\midrule",
    rows,
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}"
  )
  writeLines(out, file)
}

format_wugee_for_tex <- function(summary_df, spec) {
  labels_pm <- c("Intercept", "Time", "Time$^2$", "Baseline age", "Personal mean weight, lb", "Within-subject weight deviation, lb", "CKD", "Intercept", "Absolute difference in baseline age", "Absolute difference in personal mean weight", "CKD discordance", "Both CKD", "Intercept", "Time", "Time$^2$", "Weight deviation effect")
  labels_bc <- c("Intercept", "Time", "Time$^2$", "Baseline age", "Baseline weight, lb", "Change from baseline weight, lb", "CKD", "Intercept", "Absolute difference in baseline age", "Absolute difference in baseline weight", "CKD discordance", "Both CKD", "Intercept", "Time", "Time$^2$", "Weight change effect")
  labels <- if (spec == "personal_mean") labels_pm else labels_bc
  idx <- c(0:6, 0:4, 0:3)
  prefixes <- c(rep("\\beta", 7), rep("\\alpha", 5), rep("\\tau", 4))
  summary_df %>%
    mutate(
      Parameter = sprintf("$%s_{%d}$ (%s)", prefixes, idx, labels),
      Estimate = sprintf("%.3f", Estimate),
      `Std. Error` = sprintf("%.3f", Std.Error),
      `Wald z` = sprintf("%.3f", Wald_z),
      `p-value` = fmt_p(P_value)
    ) %>%
    select(Component, Parameter, Estimate, `Std. Error`, `Wald z`, `p-value`)
}

format_lsm_for_tex <- function(summary_df, spec, fit) {
  labels_pm <- c("Intercept", "Time", "Time$^2$", "Baseline age", "Personal mean weight, lb", "Within-subject weight deviation, lb", "CKD", "Intercept", "Time", "Time$^2$", "Baseline age", "Personal mean weight, lb", "Within-subject weight deviation, lb", "CKD")
  labels_bc <- c("Intercept", "Time", "Time$^2$", "Baseline age", "Baseline weight, lb", "Change from baseline weight, lb", "CKD", "Intercept", "Time", "Time$^2$", "Baseline age", "Baseline weight, lb", "Change from baseline weight, lb", "CKD")
  labels <- if (spec == "personal_mean") labels_pm else labels_bc
  idx <- c(0:6, 0:6)
  prefixes <- c(rep("\\beta", 7), rep("\\alpha", 7))
  summary_df %>%
    mutate(
      Parameter = sprintf("$%s_{%d}$ (%s)", prefixes, idx, labels),
      Estimate = sprintf("%.3f", Estimate),
      `Std. Error` = sprintf("%.3f", Std.Error),
      `Wald z` = sprintf("%.3f", Wald_z),
      `p-value` = fmt_p(P_value)
    ) %>%
    select(Component, Parameter, Estimate, `Std. Error`, `Wald z`, `p-value`)
}

message("Compiling revised UGEE solvers")
Rcpp::sourceCpp("WUGEE_subj_v.cpp")
Rcpp::sourceCpp("WUGEE_v3_v.cpp")

panel0 <- prepare_panel()
message(sprintf("Monotone-eligible episodes: %d | rows: %d", n_distinct(panel0$episode), nrow(panel0)))

results <- list()
all_checks <- list()
for (spec in c("personal_mean", "baseline_change")) {
  message("\n=== Specification: ", spec, " ===")
  panel_spec <- add_weight_spec(panel0, spec)
  checks <- validate_spec(panel_spec, spec)
  all_checks[[spec]] <- checks
  lsm <- fit_lsm_spec(panel_spec, spec)
  wugee <- fit_wugee_spec(panel_spec, spec)
  results[[spec]] <- list(lsm = lsm, wugee = wugee)

  write_latex_table(
    format_wugee_for_tex(wugee$subject_summary, spec),
    file.path(OUT_DIR, paste0("table_wugee_subject_", spec, ".tex")),
    paste("Subject-level weighted UGEE estimates for systolic blood pressure under the", gsub("_", "/", spec), "weight parameterization."),
    paste0("tab:ugee_subject_", spec)
  )
  write_latex_table(
    format_wugee_for_tex(wugee$pair_summary, spec),
    file.path(OUT_DIR, paste0("table_wugee_pair_", spec, ".tex")),
    paste("Pairwise weighted UGEE estimates for systolic blood pressure under the", gsub("_", "/", spec), "weight parameterization."),
    paste0("tab:ugee_pair_", spec)
  )
  write_latex_table(
    format_lsm_for_tex(lsm$summary, spec, lsm$fit),
    file.path(OUT_DIR, paste0("table_lsm_", spec, ".tex")),
    paste("Location-scale mixed model estimates for systolic blood pressure under the", gsub("_", "/", spec), "weight parameterization."),
    paste0("tab:lsm_", spec)
  )
}

comparison <- bind_rows(
  results$personal_mean$wugee$subject_summary %>% mutate(spec = "personal_mean", estimator = "UGEE subject"),
  results$baseline_change$wugee$subject_summary %>% mutate(spec = "baseline_change", estimator = "UGEE subject"),
  results$personal_mean$lsm$summary %>% mutate(spec = "personal_mean", estimator = "LSM"),
  results$baseline_change$lsm$summary %>% mutate(spec = "baseline_change", estimator = "LSM")
)
write.csv(comparison, file.path(OUT_DIR, "parameterization_comparison_all_models.csv"), row.names = FALSE)
saveRDS(results, file.path(OUT_DIR, "revised_realdata_results.rds"))
write.csv(bind_rows(all_checks, .id = "spec"), file.path(OUT_DIR, "validation_all_specs.csv"), row.names = FALSE)

message("\nRevised real-data pipeline complete. Outputs saved in ", OUT_DIR)
