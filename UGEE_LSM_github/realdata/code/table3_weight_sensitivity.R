suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(Rcpp)
})

base_dir <- "."   # directory holding CHAP_visit_level_data.csv (not distributed) and final_fit/
work_dir <- base_dir
final_dir <- file.path(base_dir, "final_fit")
out_dir <- file.path(work_dir, "table3_weight_sensitivity")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

K <- 10L
eps <- 1e-5
max_iter <- 999L
clip_eps <- 1e-6

fmt_p <- function(p) {
  ifelse(is.na(p), NA_character_, ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))
}

classify_pattern <- function(obs_vec) {
  if (all(obs_vec == 1L)) return("complete")
  if (all(obs_vec == 0L)) return("all_missing")
  first_miss <- min(which(obs_vec == 0L))
  if (any(obs_vec[seq(first_miss, length(obs_vec))] == 1L)) return("intermittent")
  "monotone_dropout"
}

to_ki_list <- function(panel, col) {
  wide <- panel %>%
    select(episode, visit_num, val = all_of(col)) %>%
    pivot_wider(names_from = visit_num, values_from = val) %>%
    arrange(episode)
  mat <- as.matrix(wide[, -1])
  lapply(seq_len(ncol(mat)), function(j) as.numeric(mat[, j]))
}

get_subj_vec <- function(panel, col) {
  panel %>% filter(visit_num == 1L) %>% arrange(episode) %>% pull(all_of(col))
}

safe_fit <- function(formula, data) {
  fit <- glm(formula, data = data, family = binomial())
  if (!fit$converged) warning("Dropout glm did not converge: ", deparse(formula))
  fit
}

clip_prob <- function(x) pmin(pmax(as.numeric(x), clip_eps), 1 - clip_eps)

prepare_panel <- function() {
  dat_raw <- read.csv(file.path(work_dir, "CHAP_visit_level_data.csv"), stringsAsFactors = FALSE) %>%
    mutate(
      visitdt = as.Date(visitdt),
      ckd = as.integer(ckd),
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

  panel <- dat_raw %>%
    filter(episode %in% mono_ids) %>%
    mutate(episode = factor(episode, levels = sort(unique(mono_ids)))) %>%
    arrange(episode, visit_num)

  age_grand_mean <- mean(panel$age_bl[panel$visit_num == 1L], na.rm = TRUE)
  baseline_weight <- panel %>%
    filter(visit_num == 1L) %>%
    transmute(episode, baseline_weight_raw = weight_lb)
  baseline_wt_grand_mean <- mean(baseline_weight$baseline_weight_raw, na.rm = TRUE)

  panel %>%
    left_join(baseline_weight, by = "episode") %>%
    mutate(
      age_c = age_bl - age_grand_mean,
      baseline_weight_c = baseline_weight_raw - baseline_wt_grand_mean,
      weight_change_from_baseline_lb = weight_lb - baseline_weight_raw,
      weight_between = baseline_weight_c,
      weight_within = weight_change_from_baseline_lb
    )
}

make_transition_data <- function(panel) {
  panel %>%
    arrange(episode, visit_num) %>%
    group_by(episode) %>%
    mutate(
      r_prev = lag(obs_sbp),
      r_cur = obs_sbp,
      last_sbp = lag(sbp),
      last_weight = lag(weight_lb)
    ) %>%
    ungroup() %>%
    filter(visit_num >= 2L, r_prev == 1L) %>%
    mutate(k = visit_num)
}

fit_visit_models <- function(transitions, denom_formula, num_formula = NULL) {
  by_visit <- split(transitions, transitions$k)
  lapply(by_visit, function(dat) {
    den <- safe_fit(denom_formula, dat)
    num <- if (is.null(num_formula)) NULL else safe_fit(num_formula, dat)
    list(k = unique(dat$k), den = den, num = num)
  })
}

predict_visit_prob <- function(model, dat) clip_prob(predict(model, newdata = dat, type = "response"))

build_weights <- function(panel, transitions, models, stabilized = FALSE, truncate_q = NULL) {
  weights <- panel %>%
    arrange(episode, visit_num) %>%
    select(episode, visit_num, obs_sbp, ckd) %>%
    mutate(weight = 1, denominator_pi = 1, numerator_pi = 1)

  work <- transitions %>%
    arrange(episode, visit_num) %>%
    mutate(p_den = NA_real_, p_num = NA_real_)

  for (nm in names(models)) {
    model_info <- models[[nm]]
    idx <- which(work$k == model_info$k)
    work$p_den[idx] <- predict_visit_prob(model_info$den, work[idx, ])
    if (stabilized) {
      work$p_num[idx] <- predict_visit_prob(model_info$num, work[idx, ])
    } else {
      work$p_num[idx] <- 1
    }
  }

  work <- work %>%
    group_by(episode) %>%
    arrange(visit_num, .by_group = TRUE) %>%
    mutate(
      denominator_pi = cumprod(p_den),
      numerator_pi = cumprod(p_num),
      weight = numerator_pi / denominator_pi
    ) %>%
    ungroup()

  weights <- weights %>%
    left_join(work %>% select(episode, visit_num, weight, denominator_pi, numerator_pi),
              by = c("episode", "visit_num"), suffix = c("", ".new")) %>%
    mutate(
      weight = coalesce(weight.new, weight),
      denominator_pi = coalesce(denominator_pi.new, denominator_pi),
      numerator_pi = coalesce(numerator_pi.new, numerator_pi)
    ) %>%
    select(episode, visit_num, obs_sbp, ckd, weight, denominator_pi, numerator_pi)

  truncation_value <- NA_real_
  if (!is.null(truncate_q)) {
    obs_w <- weights$weight[weights$obs_sbp == 1L]
    truncation_value <- as.numeric(quantile(obs_w, truncate_q, na.rm = TRUE))
    weights <- weights %>% mutate(weight = pmin(weight, truncation_value))
  }

  attr(weights, "transition_predictions") <- work
  attr(weights, "truncation_value") <- truncation_value
  weights
}

weight_diagnostics <- function(weights, strategy) {
  obs_w <- weights %>% filter(obs_sbp == 1L)
  qs <- quantile(obs_w$weight, probs = c(.01, .05, .25, .50, .75, .95, .99), na.rm = TRUE)
  data.frame(
    strategy = strategy,
    n_subjects = n_distinct(obs_w$episode),
    n_visits_observed = nrow(obs_w),
    mean = mean(obs_w$weight),
    sd = sd(obs_w$weight),
    min = min(obs_w$weight),
    p01 = unname(qs[1]),
    p05 = unname(qs[2]),
    p25 = unname(qs[3]),
    p50 = unname(qs[4]),
    p75 = unname(qs[5]),
    p95 = unname(qs[6]),
    p99 = unname(qs[7]),
    max = max(obs_w$weight),
    n_gt_5 = sum(obs_w$weight > 5),
    prop_gt_5 = mean(obs_w$weight > 5),
    n_gt_10 = sum(obs_w$weight > 10),
    prop_gt_10 = mean(obs_w$weight > 10),
    stringsAsFactors = FALSE
  )
}

summarize_fit <- function(fit, strategy) {
  param <- c(paste0("beta", 0:6), paste0("alpha", 0:4), paste0("tau", 0:3))
  label <- c(
    "beta0 (Intercept)", "beta1 (Time)", "beta2 (Time^2)", "beta3 (Baseline age)",
    "beta4 (Baseline weight, lb)", "beta5 (Change from baseline weight, lb)", "beta6 (CKD)",
    "alpha0 (Intercept)", "alpha1 (Absolute difference of baseline age)",
    "alpha2 (Absolute difference of baseline weight)", "alpha3 (CKD discordance)", "alpha4 (Both CKD)",
    "tau0 (Intercept)", "tau1 (Time)", "tau2 (Time^2)", "tau3 (Weight change effect)"
  )
  component <- c(rep("Marginal Mean", 7), rep("Between-subject Heterogeneity", 5), rep("Within-subject Variability", 4))
  est <- c(fit$Beta, fit$Eta)
  se <- sqrt(diag(fit$Sigma.theta))
  z <- est / se
  p <- 2 * pnorm(abs(z), lower.tail = FALSE)
  data.frame(strategy, component, param, label, estimate = est, std_error = se,
             wald_z = z, p_value = p, converged = fit$converged, iter = fit$iter,
             stringsAsFactors = FALSE)
}

fit_weighted_wugee <- function(panel, weights, strategy) {
  n <- n_distinct(panel$episode)
  y_ki_all <- to_ki_list(panel, "sbp")
  r_indiv_all <- to_ki_list(panel, "obs_sbp")
  lam_i_all <- weights %>%
    select(episode, visit_num, weight) %>%
    pivot_wider(names_from = visit_num, values_from = weight) %>%
    arrange(episode) %>%
    select(-episode) %>%
    as.matrix()
  lam_i_all <- lapply(seq_len(ncol(lam_i_all)), function(j) as.numeric(lam_i_all[, j]))

  wt_within_ki_all <- to_ki_list(panel, "weight_within")
  wt_within_ki_all <- lapply(wt_within_ki_all, function(x) { x[is.na(x)] <- 0; x })

  panel_cc <- panel %>% filter(!is.na(sbp))
  lm_init <- lm(sbp ~ visit_num + I(visit_num^2) + age_c + weight_between + weight_within + ckd, data = panel_cc)
  Beta_init <- as.numeric(coef(lm_init))
  var_sbp <- var(panel_cc$sbp, na.rm = TRUE)
  Eta_init <- c(log(var_sbp / 2), rep(0, 4), log(var_sbp / 4), rep(0, 3))
  Xi_init <- rep(0, 2L * (K - 1L))

  tm <- system.time({
    fit <- wugee_subj_keyed_weight_v(
      y_ki_all = y_ki_all,
      r_indiv_all = r_indiv_all,
      lam_i_all = lam_i_all,
      age_c_R = get_subj_vec(panel, "age_c"),
      wt_mean_R = get_subj_vec(panel, "weight_between"),
      wt_dev_ki_all = wt_within_ki_all,
      ckd_R = get_subj_vec(panel, "ckd"),
      Beta_init = Beta_init,
      Eta_init = Eta_init,
      Xi_init = Xi_init,
      eps = eps,
      max_iter = max_iter
    )
  })
  fit$elapsed_seconds <- unname(tm["elapsed"])
  saveRDS(fit, file.path(out_dir, paste0("fit_", strategy, ".rds")))
  summarize_fit(fit, strategy)
}

write_latex <- function(df, file, caption, label) {
  lines <- apply(df, 1, function(r) {
    paste(r[["Strategy"]], r[["Parameter"]], r[["Estimate"]], r[["Std. Error"]],
          r[["Wald z"]], r[["p-value"]], sep = " & ") |>
      paste0("\\\\")
  })
  writeLines(c(
    "\\begin{table}[ht]",
    "\\centering",
    paste0("\\caption{", caption, "}"),
    paste0("\\label{", label, "}"),
    "\\begin{tabular}{llrrrr}",
    "\\toprule",
    "Strategy & Parameter & Estimate & Std. Error & Wald $z$ & $p$-value\\\\",
    "\\midrule",
    lines,
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}"
  ), file)
}

write_diag_latex <- function(df, file) {
  fdf <- df %>%
    mutate(across(c(mean, sd, min, p01, p05, p25, p50, p75, p95, p99, max, prop_gt_5, prop_gt_10), ~sprintf("%.3f", .x)))
  lines <- apply(fdf, 1, function(r) {
    paste(r[["strategy"]], r[["n_subjects"]], r[["n_visits_observed"]], r[["mean"]],
          r[["sd"]], r[["min"]], r[["p50"]], r[["p95"]], r[["p99"]], r[["max"]],
          r[["n_gt_5"]], r[["n_gt_10"]], sep = " & ") |>
      paste0("\\\\")
  })
  writeLines(c(
    "\\begin{table}[ht]",
    "\\centering",
    "\\caption{Weight diagnostics for Table 3 dropout-weight sensitivity analyses.}",
    "\\label{tab:table3_weight_diagnostics}",
    "\\begin{tabular}{lrrrrrrrrrrr}",
    "\\toprule",
    "Strategy & Subjects & Visits & Mean & SD & Min & Median & P95 & P99 & Max & $N>5$ & $N>10$\\\\",
    "\\midrule",
    lines,
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}"
  ), file)
}

message("Compiling external-weight WUGEE solver")
Rcpp::sourceCpp(file.path(final_dir, "WUGEE_subj_final_keyed_weights.cpp"))

panel <- prepare_panel()
transitions <- make_transition_data(panel)

current_models <- fit_visit_models(transitions, r_cur ~ last_sbp)
expanded_models <- fit_visit_models(transitions, r_cur ~ last_sbp + age_c + baseline_weight_c + last_weight)
stab_time_models <- fit_visit_models(transitions,
                                     r_cur ~ last_sbp + age_c + baseline_weight_c + last_weight,
                                     r_cur ~ 1)
stab_baseline_models <- fit_visit_models(transitions,
                                         r_cur ~ last_sbp + age_c + baseline_weight_c + last_weight,
                                         r_cur ~ age_c + baseline_weight_c)

weights <- list(
  current_unstabilized = build_weights(panel, transitions, current_models, stabilized = FALSE),
  expanded_unstabilized = build_weights(panel, transitions, expanded_models, stabilized = FALSE),
  expanded_stabilized_time_only = build_weights(panel, transitions, stab_time_models, stabilized = TRUE),
  expanded_stabilized_baseline = build_weights(panel, transitions, stab_baseline_models, stabilized = TRUE)
)

stab_max <- max(weights$expanded_stabilized_time_only$weight[weights$expanded_stabilized_time_only$obs_sbp == 1L], na.rm = TRUE)
if (stab_max > 5) {
  weights$expanded_stabilized_time_only_truncated_99 <- build_weights(
    panel, transitions, stab_time_models, stabilized = TRUE, truncate_q = 0.99
  )
}

for (nm in names(weights)) {
  write.csv(weights[[nm]], file.path(out_dir, paste0("weights_", nm, ".csv")), row.names = FALSE)
  pred <- attr(weights[[nm]], "transition_predictions")
  write.csv(pred, file.path(out_dir, paste0("transition_predictions_", nm, ".csv")), row.names = FALSE)
}

diagnostics <- bind_rows(lapply(names(weights), function(nm) weight_diagnostics(weights[[nm]], nm)))
write.csv(diagnostics, file.path(out_dir, "weight_diagnostics.csv"), row.names = FALSE)
write_diag_latex(diagnostics, file.path(out_dir, "weight_diagnostics.tex"))

dropout_coefs <- bind_rows(lapply(names(expanded_models), function(nm) {
  k <- expanded_models[[nm]]$k
  den <- summary(expanded_models[[nm]]$den)$coefficients
  data.frame(model = "expanded_denominator_visit_specific", visit_num = k,
             term = rownames(den), estimate = den[, 1], std_error = den[, 2],
             p_value = den[, 4], row.names = NULL)
}))
write.csv(dropout_coefs, file.path(out_dir, "expanded_dropout_model_coefficients.csv"), row.names = FALSE)

fits <- bind_rows(lapply(setdiff(names(weights), "current_unstabilized"), function(nm) {
  message("Fitting ", nm)
  fit_weighted_wugee(panel, weights[[nm]], nm)
}))

current <- read.csv(file.path(final_dir, "table3_final.csv")) %>%
  transmute(
    strategy = "current_unstabilized_primary",
    component = c(rep("Marginal Mean", 7), rep("Between-subject Heterogeneity", 5), rep("Within-subject Variability", 4)),
    param = param,
    label = c(
      "beta0 (Intercept)", "beta1 (Time)", "beta2 (Time^2)", "beta3 (Baseline age)",
      "beta4 (Baseline weight, lb)", "beta5 (Change from baseline weight, lb)", "beta6 (CKD)",
      "alpha0 (Intercept)", "alpha1 (Absolute difference of baseline age)",
      "alpha2 (Absolute difference of baseline weight)", "alpha3 (CKD discordance)", "alpha4 (Both CKD)",
      "tau0 (Intercept)", "tau1 (Time)", "tau2 (Time^2)", "tau3 (Weight change effect)"
    ),
    estimate = estimate,
    std_error = std_error,
    wald_z = wald_z,
    p_value = p_value,
    converged = TRUE,
    iter = NA_integer_
  )

comparison <- bind_rows(current, fits)
write.csv(comparison, file.path(out_dir, "table3_weight_sensitivity_comparison.csv"), row.names = FALSE)

targets <- comparison %>%
  filter(param %in% c("tau1", "tau2", "tau3", "alpha3", "alpha4")) %>%
  arrange(factor(param, levels = c("tau1", "tau2", "tau3", "alpha3", "alpha4")), strategy)
write.csv(targets, file.path(out_dir, "table3_weight_sensitivity_targets.csv"), row.names = FALSE)

tex_df <- targets %>%
  mutate(
    Strategy = strategy,
    Parameter = label,
    Estimate = sprintf("%.3f", estimate),
    `Std. Error` = sprintf("%.3f", std_error),
    `Wald z` = sprintf("%.3f", wald_z),
    `p-value` = fmt_p(p_value)
  ) %>%
  select(Strategy, Parameter, Estimate, `Std. Error`, `Wald z`, `p-value`)
write_latex(tex_df, file.path(out_dir, "table3_weight_sensitivity_targets.tex"),
            "Target-parameter comparison for Table 3 dropout-weight sensitivity analyses.",
            "tab:table3_weight_sensitivity_targets")

all_tex_df <- comparison %>%
  mutate(
    Strategy = strategy,
    Parameter = label,
    Estimate = sprintf("%.3f", estimate),
    `Std. Error` = sprintf("%.3f", std_error),
    `Wald z` = sprintf("%.3f", wald_z),
    `p-value` = fmt_p(p_value)
  ) %>%
  select(Strategy, Parameter, Estimate, `Std. Error`, `Wald z`, `p-value`)
write_latex(all_tex_df, file.path(out_dir, "table3_weight_sensitivity_comparison.tex"),
            "Full Table 3 WUGEE comparison under alternative dropout-weight models.",
            "tab:table3_weight_sensitivity_full")

diag_brief <- diagnostics %>%
  transmute(strategy, mean = round(mean, 3), sd = round(sd, 3), max = round(max, 3),
            p99 = round(p99, 3), n_gt_5, n_gt_10)

primary_reproduced <- read.csv(file.path(final_dir, "table3_final.csv"))

memo <- c(
  "# Table 3 Weighting Sensitivity Analysis Memo",
  "",
  paste0("Run date: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  "",
  "## Code and Data",
  "",
  paste0("- Data: `", file.path(work_dir, "CHAP_visit_level_data.csv"), "`."),
  paste0("- Primary Table 3 source: `", file.path(final_dir, "table3_final.csv"), "`."),
  paste0("- External-weight solver: `", file.path(final_dir, "WUGEE_subj_final_keyed_weights.cpp"), "`."),
  "- Outcome/model variables confirmed in code: `sbp`; visits `visit_num` 1-10; baseline age `age_bl`; baseline/current weight `weight_lb`; CKD `ckd`.",
  "- Monotone-eligible analysis set: complete plus monotone-dropout episodes; intermittent and all-missing episodes are excluded before fitting.",
  "",
  "## Weight Construction",
  "",
  "- Current primary weights are the saved current Table 3 WUGEE results based on visit-specific retention models using most recently observed SBP only.",
  "- Expanded unstabilized weights use visit-specific denominator models `r_cur ~ last_sbp + age_c + baseline_weight_c + last_weight`.",
  "- Expanded stabilized time-only weights use the same denominator and a visit-specific intercept-only numerator.",
  "- Expanded stabilized baseline weights use the same denominator and visit-specific numerator `r_cur ~ age_c + baseline_weight_c`.",
  "- `last_weight` is the observed patient weight at the immediately preceding retained visit for each visit-to-visit transition.",
  "- Weights are cumulative through each visit and are applied only to observed SBP rows in the UGEE equations.",
  "",
  "## Diagnostics",
  "",
  capture.output(print(diag_brief, row.names = FALSE)),
  "",
  "## Main Target Parameters",
  "",
  capture.output(print(targets %>% select(strategy, param, estimate, std_error, wald_z, p_value), row.names = FALSE)),
  "",
  "## Recommendation",
  "",
  "The current Table 3 should not be kept as the only primary analysis without qualification. The expanded visit-specific unstabilized and stabilized weights preserve the broad time-variability pattern for tau1/tau2, but tau3 becomes much more negative and highly significant, and alpha3/alpha4 inference changes across stabilized variants. Because the dropout-weight specification materially affects several scientific targets, the stabilized expanded model should be considered for the primary analysis or, at minimum, the manuscript should report this sensitivity prominently rather than as a routine robustness check.",
  "",
  "## Deliverables",
  "",
  "- `weight_diagnostics.csv` and `weight_diagnostics.tex`",
  "- `table3_weight_sensitivity_comparison.csv` and `table3_weight_sensitivity_comparison.tex`",
  "- `table3_weight_sensitivity_targets.csv` and `table3_weight_sensitivity_targets.tex`",
  "- `weights_*.csv`, `transition_predictions_*.csv`, and `fit_*.rds` for auditability"
)
writeLines(memo, file.path(out_dir, "analysis_memo.md"))

message("Complete. Outputs saved to: ", out_dir)
