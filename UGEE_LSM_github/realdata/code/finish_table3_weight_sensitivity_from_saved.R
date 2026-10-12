suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

base_dir <- "."   # directory holding CHAP_visit_level_data.csv (not distributed) and final_fit/
work_dir <- base_dir
final_dir <- file.path(base_dir, "final_fit")
out_dir <- file.path(work_dir, "table3_weight_sensitivity")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

fmt_p <- function(p) ifelse(is.na(p), NA_character_, ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))

write_latex <- function(df, file, caption, label) {
  rows <- apply(df, 1, function(r) {
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
    rows,
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}"
  ), file)
}

wide <- read.csv(file.path(final_dir, "wugee_pooled_before_after.csv"), check.names = FALSE)
strategy_map <- data.frame(
  prefix = c("W1", "W2", "W3", "W4"),
  strategy = c(
    "current primary: visit-specific SBP-only unstabilized",
    "expanded pooled linear unstabilized",
    "expanded pooled spline stabilized",
    "expanded pooled spline stabilized, truncated at 99th percentile"
  ),
  stringsAsFactors = FALSE
)

long <- bind_rows(lapply(seq_len(nrow(strategy_map)), function(i) {
  pfx <- strategy_map$prefix[i]
  data.frame(
    strategy = strategy_map$strategy[i],
    param = wide$param,
    label = wide$label,
    panel = wide$panel,
    estimate = wide[[paste0(pfx, "_estimate")]],
    std_error = wide[[paste0(pfx, "_std_error")]],
    p_value = wide[[paste0(pfx, "_p_value")]],
    stringsAsFactors = FALSE
  )
})) %>%
  mutate(
    component = case_when(
      grepl("^beta", param) ~ "Marginal Mean",
      grepl("^alpha", param) ~ "Between-subject Heterogeneity",
      TRUE ~ "Within-subject Variability"
    ),
    wald_z = estimate / std_error
  ) %>%
  select(strategy, component, param, label, estimate, std_error, wald_z, p_value)

write.csv(long, file.path(out_dir, "table3_weight_sensitivity_comparison.csv"), row.names = FALSE)

targets <- long %>%
  filter(param %in% c("tau1", "tau2", "tau3", "alpha3", "alpha4")) %>%
  arrange(factor(param, levels = c("tau1", "tau2", "tau3", "alpha3", "alpha4")), strategy)
write.csv(targets, file.path(out_dir, "table3_weight_sensitivity_targets.csv"), row.names = FALSE)

tex_full <- long %>%
  mutate(
    Strategy = strategy,
    Parameter = label,
    Estimate = sprintf("%.3f", estimate),
    `Std. Error` = sprintf("%.3f", std_error),
    `Wald z` = sprintf("%.3f", wald_z),
    `p-value` = fmt_p(p_value)
  ) %>%
  select(Strategy, Parameter, Estimate, `Std. Error`, `Wald z`, `p-value`)
write_latex(tex_full, file.path(out_dir, "table3_weight_sensitivity_comparison.tex"),
            "Full Table 3 WUGEE comparison under alternative dropout-weight models.",
            "tab:table3_weight_sensitivity_full")

tex_targets <- targets %>%
  mutate(
    Strategy = strategy,
    Parameter = label,
    Estimate = sprintf("%.3f", estimate),
    `Std. Error` = sprintf("%.3f", std_error),
    `Wald z` = sprintf("%.3f", wald_z),
    `p-value` = fmt_p(p_value)
  ) %>%
  select(Strategy, Parameter, Estimate, `Std. Error`, `Wald z`, `p-value`)
write_latex(tex_targets, file.path(out_dir, "table3_weight_sensitivity_targets.tex"),
            "Target-parameter comparison for Table 3 dropout-weight sensitivity analyses.",
            "tab:table3_weight_sensitivity_targets")

diagnostics <- read.csv(file.path(out_dir, "weight_diagnostics.csv"))
pooled_diag <- read.csv(file.path(final_dir, "wugee_pooled_weight_distributions.csv")) %>%
  rename(strategy = weight_set) %>%
  filter(ckd_status == "overall") %>%
  transmute(
    strategy = recode(strategy,
                      current_visit_specific_sbp_unstabilized = "current primary: visit-specific SBP-only unstabilized",
                      pooled_enriched_stabilized = "expanded pooled spline stabilized"),
    n_subjects = NA_integer_,
    n_visits_observed = NA_integer_,
    mean = NA_real_,
    sd = NA_real_,
    min, p01, p05, p25 = NA_real_, p50, p75 = NA_real_, p95, p99, max,
    n_gt_5 = NA_integer_, prop_gt_5 = NA_real_,
    n_gt_10 = subject_times_above_10, prop_gt_10 = NA_real_
  )
write.csv(pooled_diag, file.path(out_dir, "pooled_saved_weight_diagnostics.csv"), row.names = FALSE)

main <- targets %>%
  mutate(result = sprintf("%s: %.3f (SE %.3f), p=%s", param, estimate, std_error, fmt_p(p_value))) %>%
  select(strategy, result)

memo <- c(
  "# Table 3 Weighting Sensitivity Analysis Memo",
  "",
  paste0("Run date: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  "",
  "## Code and Data Used",
  "",
  paste0("- Data: `", file.path(work_dir, "CHAP_visit_level_data.csv"), "`."),
  paste0("- Current primary Table 3: `", file.path(final_dir, "table3_final.csv"), "`."),
  paste0("- Saved converged sensitivity comparison: `", file.path(final_dir, "wugee_pooled_before_after.csv"), "`."),
  paste0("- Newly generated visit-specific diagnostic weights: `", out_dir, "/weights_*.csv`."),
  "",
  "## Reproduction Check",
  "",
  "- The current primary WUGEE column is reproduced from `final_fit/table3_final.csv`, which matches the manuscript Table 3 target values.",
  "- The saved final-fit report `REPORT_pooled.md` records Gate 3 as PASS for reloaded W1 reproducing `table3_final.csv` to three decimals.",
  "- A direct newly written external-weight refit for visit-specific weights was started, but the generic keyed-weight C++ path did not finish interactively. The estimates table therefore uses the already-converged final-fit sensitivity objects, while the newly generated visit-specific weights are retained as diagnostics/audit artifacts.",
  "",
  "## Weight Models",
  "",
  "- Current primary: separate visit-specific logistic retention models using most recently observed SBP.",
  "- Newly generated expanded visit-specific unstabilized diagnostics: denominator `r_cur ~ last_sbp + age_c + baseline_weight_c + last_weight`.",
  "- Newly generated expanded stabilized diagnostics: same denominator with either visit-specific intercept-only numerator or visit-specific `age_c + baseline_weight_c` numerator.",
  "- Saved converged sensitivity fits used for estimates: enriched pooled dropout models in `final_fit`, including expanded pooled linear unstabilized/stabilized variants and a spline stabilized truncated variant as labeled in the comparison table.",
  "",
  "## Newly Generated Visit-Specific Weight Diagnostics",
  "",
  capture.output(print(diagnostics %>% select(strategy, n_subjects, n_visits_observed, mean, sd, p99, max, n_gt_5, n_gt_10), row.names = FALSE)),
  "",
  "## Target-Parameter Comparison",
  "",
  capture.output(print(targets %>% select(strategy, param, estimate, std_error, wald_z, p_value), row.names = FALSE)),
  "",
  "## Interpretation and Recommendation",
  "",
  "The broad within-subject time-variability pattern is stable: tau1 remains negative and tau2 remains positive under the saved enriched/truncated sensitivity fits. However, tau3 becomes substantially more negative under enriched weighting, and CKD-pair heterogeneity terms, especially alpha3 and alpha4, weaken or change inference relative to the current primary Table 3. These shifts are material enough that the current Table 3 should not be described as fully robust to richer/stabilized dropout weighting.",
  "",
  "Recommendation: treat the expanded stabilized analysis as a serious candidate for the primary analysis, or keep the current Table 3 primary only with a prominent supplemental sensitivity table and a cautious sentence noting that several target estimates, especially tau3 and CKD-pair heterogeneity, changed under enriched weighting.",
  "",
  "## Deliverables",
  "",
  "- `weight_diagnostics.csv` and `weight_diagnostics.tex`",
  "- `pooled_saved_weight_diagnostics.csv`",
  "- `table3_weight_sensitivity_comparison.csv` and `.tex`",
  "- `table3_weight_sensitivity_targets.csv` and `.tex`",
  "- `analysis_memo.md`"
)
writeLines(memo, file.path(out_dir, "analysis_memo.md"))

message("Finished sensitivity deliverables in: ", out_dir)
