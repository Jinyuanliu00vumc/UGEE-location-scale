## ============================================================
## CHAP Study — Final Analysis
## ============================================================
## Pipeline:
##   1. Read and prepare CHAP_visit_level_data.csv
##      (long-format panel, one row per episode × visit)
##   2. Missing-data pipeline adapted from CHAP_missing_pattern.R
##      • Time axis  : visit number (1–10) instead of 4-month bins
##      • Subject ID : episode (one pregnancy) instead of mrn
##      • Outcome    : sbp (raw mmHg) instead of log-SBP bins
##      • Groups     : race_binary (Black / Non-Black) — the primary
##                     equity comparison within CHTN, analogous to
##                     CHTN vs NoCHTN in the original script
##   3. Figures 1–5 for both race groups (parallel to original script)
##   4. Subject-level missing-pattern heatmap — whole sample (Fig 6)
##   5. SBP mean & variability trajectory plots — whole sample (Figs T1–T4)
##   6. glmmTMB heteroskedastic model with centered predictors
##      and within-episode weight decomposition (mean + deviation)
##   7. Model diagnostics (Fig D: 5-panel page)
##   8. Outputs saved to CHAP_final_analysis.pdf
## ============================================================


## ============================================================
## 0)  Libraries
## ============================================================
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
  library(patchwork)
  library(glmmTMB)
})

theme_set(theme_bw(base_size = 12))


## ============================================================
## 1)  Load data
## ============================================================
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dat_raw <- read.csv("CHAP_visit_level_data.csv", stringsAsFactors = FALSE)

## Basic type coercions
dat_raw <- dat_raw %>%
  mutate(
    visitdt       = as.Date(visitdt),
    race_binary   = factor(race_binary, levels = c("Non-Black", "Black")),
    ckd           = as.integer(ckd),
    post_chap     = as.integer(post_chap),
    sbp_mono_miss = as.integer(sbp_mono_miss)
  )

cat(sprintf("Loaded: %d rows | %d episodes | %d visit slots each\n",
            nrow(dat_raw),
            n_distinct(dat_raw$episode),
            n_distinct(dat_raw$visit_num)))
cat(sprintf("Race breakdown (at visit 1): %s\n",
            paste(
              names(table(dat_raw$race_binary[dat_raw$visit_num == 1])),
              table(dat_raw$race_binary[dat_raw$visit_num == 1]),
              sep = " = ", collapse = "  |  "
            )))


## ============================================================
## 2)  Prepare long-format panel
##     One row per episode × visit_num (already in this format).
##     Add obs_sbp indicator (1 = SBP observed, 0 = monotone missing).
## ============================================================
panel <- dat_raw %>%
  mutate(
    obs_sbp = as.integer(!is.na(sbp)),    # 1 = observed; 0 = missing
    grp     = race_binary                 # grouping variable for pipeline
  ) %>%
  arrange(episode, visit_num)

K        <- max(panel$visit_num)          # = 10
N_ep     <- n_distinct(panel$episode)

cat(sprintf("Panel: %d episodes × %d visits = %d rows\n",
            N_ep, K, nrow(panel)))
cat(sprintf("SBP observed: %d  |  Monotone missing: %d  (%.1f%%)\n",
            sum(panel$obs_sbp),
            sum(panel$sbp_mono_miss),
            100 * mean(panel$sbp_mono_miss)))


## ============================================================
## 3)  Helper: classify_pattern()
##     Identical logic to CHAP_missing_pattern.R §3.
##     obs_vec : integer vector of 0/1 (0 = missing, 1 = observed)
## ============================================================
classify_pattern <- function(obs_vec) {
  if (all(obs_vec == 1L)) return("complete")
  if (all(obs_vec == 0L)) return("all_missing")
  first_miss <- min(which(obs_vec == 0L))
  if (any(obs_vec[seq(first_miss, length(obs_vec))] == 1L)) {
    return("intermittent")
  }
  return("monotone_dropout")
}


## ============================================================
## 4)  Episode-level pattern classification
## ============================================================
miss_class <- panel %>%
  arrange(episode, visit_num) %>%
  group_by(episode, grp) %>%
  summarize(
    miss_pattern = classify_pattern(obs_sbp),
    n_obs        = sum(obs_sbp),
    n_miss       = K - n_obs,
    first_miss   = ifelse(any(obs_sbp == 0L),
                          min(visit_num[obs_sbp == 0L]),
                          K + 1L),          # complete → sentinel
    .groups      = "drop"
  )

cat("\n=== Pattern classification summary ===\n")
miss_class %>%
  count(grp, miss_pattern) %>%
  group_by(grp) %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  print(n = Inf)


## ============================================================
## 5)  Core pipeline function (adapted from run_missing_pipeline)
##     Processes a single race group; returns a named list.
## ============================================================
run_visit_pipeline <- function(panel_grp, miss_class_grp, grp_label) {

  N_subj <- n_distinct(panel_grp$episode)

  ## Per-visit observation counts
  obs_by_visit <- panel_grp %>%
    group_by(visit_num) %>%
    summarize(
      n_total  = n(),
      n_obs    = sum(obs_sbp),
      pct_obs  = 100 * n_obs  / n_total,
      pct_miss = 100 - pct_obs,
      .groups  = "drop"
    )

  list(
    panel_grp    = panel_grp,
    miss_class   = miss_class_grp,
    obs_by_visit = obs_by_visit,
    K            = K,
    N_subj       = N_subj,
    grp_label    = grp_label
  )
}


## ============================================================
## 6)  Run pipeline for both race groups
## ============================================================
grp_labels <- levels(panel$grp)   # c("Non-Black", "Black")

pp_list <- lapply(grp_labels, function(g) {
  run_visit_pipeline(
    panel_grp      = panel      %>% filter(grp == g),
    miss_class_grp = miss_class %>% filter(grp == g),
    grp_label      = g
  )
})
names(pp_list) <- grp_labels

cat("\n=== Pipeline summary ===\n")
for (g in grp_labels) {
  pp <- pp_list[[g]]
  cat(sprintf("  %-12s : %d episodes | %d obs / %d miss (%.1f%% miss)\n",
              pp$grp_label,
              pp$N_subj,
              sum(pp$panel_grp$obs_sbp),
              sum(pp$panel_grp$sbp_mono_miss),
              100 * mean(pp$panel_grp$sbp_mono_miss)))
}


## ============================================================
## 7)  Plotting helper: Figures 1–3 per group
##     Mirrors make_group_figures() in CHAP_missing_pattern.R §7.
##     Colour scheme and layout unchanged; x-axis = visit number.
## ============================================================
make_visit_figures <- function(pp) {

  panel_grp    <- pp$panel_grp
  miss_class_g <- pp$miss_class
  obs_by_visit <- pp$obs_by_visit
  N_subj       <- pp$N_subj
  grp_label    <- pp$grp_label

  ## -- Fig 1: % missing / % observed per visit ----------------
  fig1 <- ggplot(obs_by_visit, aes(x = visit_num)) +
    geom_col(aes(y = pct_miss),
             fill = "#d62728", alpha = 0.70, width = 0.70) +
    geom_line(aes(y = pct_obs),
              colour = "#1f77b4", linewidth = 1.1) +
    geom_point(aes(y = pct_obs),
               colour = "#1f77b4", size = 2.8, shape = 16) +
    scale_x_continuous(name = "Visit number", breaks = 1:K) +
    scale_y_continuous(
      name     = "% missing  (bars)",
      limits   = c(0, 100),
      sec.axis = sec_axis(~ ., name = "% observed  (line + points)")
    ) +
    labs(
      title    = sprintf("[%s]  SBP Missingness per Visit", grp_label),
      subtitle = sprintf("N = %d episodes | K = %d visits", N_subj, K),
      caption  = "Red bars = % missing SBP; Blue line = % observed"
    ) +
    theme(
      axis.title.y.right = element_text(colour = "#1f77b4"),
      plot.title         = element_text(face = "bold"),
      plot.caption       = element_text(colour = "grey40", size = 9)
    )

  ## -- Fig 2: episode × visit heatmap -------------------------
  pattern_rank_map <- c(complete = 1L, monotone_dropout = 2L,
                        intermittent = 3L, all_missing = 4L)

  subj_order <- miss_class_g %>%
    mutate(pattern_rank = pattern_rank_map[miss_pattern]) %>%
    arrange(pattern_rank, first_miss, as.character(episode)) %>%
    mutate(subj_row = row_number())

  ## separator rows between patterns
  sep_rows <- subj_order %>%
    group_by(miss_pattern) %>%
    summarize(max_row = max(subj_row), .groups = "drop") %>%
    filter(miss_pattern != tail(levels(factor(miss_pattern, levels =
                        names(sort(pattern_rank_map)))), 1)) %>%
    pull(max_row)

  panel_hm <- panel_grp %>%
    left_join(subj_order %>% select(episode, subj_row, miss_pattern),
              by = "episode") %>%
    mutate(obs_label = factor(obs_sbp, levels = c(0L, 1L),
                              labels = c("Missing", "Observed")))

  fig2 <- ggplot(
    panel_hm,
    aes(x = visit_num,
        y = reorder(as.factor(episode), subj_row),
        fill = obs_label)
  ) +
    geom_tile(colour = NA) +
    geom_hline(yintercept = sep_rows + 0.5,
               colour = "white", linewidth = 1.0) +
    scale_fill_manual(
      values = c("Observed" = "#4daf4a", "Missing" = "#e41a1c"),
      name   = NULL
    ) +
    scale_x_continuous(name = "Visit number",
                       breaks = 1:K, expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0)) +
    labs(
      title    = sprintf("[%s]  Episode-Level Missing Pattern", grp_label),
      subtitle = sprintf("All %d episodes — sorted: pattern → first-miss visit",
                         N_subj),
      y        = "Episode (sorted)"
    ) +
    theme(
      axis.text.y     = element_blank(),
      axis.ticks.y    = element_blank(),
      panel.grid      = element_blank(),
      legend.position = "top",
      plot.title      = element_text(face = "bold"),
      plot.subtitle   = element_text(size = 9, colour = "grey30")
    )

  ## -- Fig 3: stacked bar — pattern composition per visit -----
  fig3_dat <- panel_grp %>%
    left_join(miss_class_g %>% select(episode, miss_pattern),
              by = "episode") %>%
    mutate(
      status = case_when(
        obs_sbp == 1L                                        ~ "Observed",
        obs_sbp == 0L & miss_pattern == "monotone_dropout"  ~ "Dropout (monotone)",
        obs_sbp == 0L & miss_pattern == "intermittent"      ~ "Intermittent",
        obs_sbp == 0L & miss_pattern == "all_missing"       ~ "All missing",
        TRUE                                                 ~ "Other"
      ),
      status = factor(status, levels = c(
        "Observed", "Dropout (monotone)", "Intermittent", "All missing"))
    ) %>%
    count(visit_num, status) %>%
    group_by(visit_num) %>%
    mutate(pct = 100 * n / sum(n)) %>%
    ungroup()

  fig3 <- ggplot(fig3_dat,
                 aes(x = visit_num, y = pct, fill = status)) +
    geom_col(width = 0.80) +
    scale_fill_manual(
      values = c("Observed"           = "#4daf4a",
                 "Dropout (monotone)" = "#ff7f00",
                 "Intermittent"       = "#e41a1c",
                 "All missing"        = "#984ea3"),
      name = NULL
    ) +
    scale_x_continuous(name = "Visit number", breaks = 1:K) +
    scale_y_continuous(name   = "% of episodes",
                       labels = label_percent(scale = 1)) +
    labs(
      title    = sprintf("[%s]  Observation Status per Visit", grp_label),
      subtitle = "Orange = monotone dropout | Red = intermittent"
    ) +
    theme(
      legend.position = "top",
      plot.title      = element_text(face = "bold"),
      plot.subtitle   = element_text(size = 9, colour = "grey30")
    )

  list(fig1 = fig1, fig2 = fig2, fig3 = fig3)
}

## Generate per-group figures
figs_list <- lapply(pp_list, make_visit_figures)


## ============================================================
## 8)  Figure 4 — Complete vs monotone_dropout by race group
##     (Analogue of Fig 4 in CHAP_missing_pattern.R §10b)
## ============================================================

## Build summary including excluded (intermittent / all-missing) rows
fig4_excl <- miss_class %>%
  mutate(
    miss_cat = case_when(
      miss_pattern == "complete"         ~ "Complete",
      miss_pattern == "monotone_dropout" ~ "Monotone dropout",
      TRUE                               ~ "Excluded\n(intermittent / all-missing)"
    ),
    miss_cat = factor(miss_cat, levels = c(
      "Complete",
      "Monotone dropout",
      "Excluded\n(intermittent / all-missing)"
    ))
  ) %>%
  count(grp, miss_cat) %>%
  group_by(grp) %>%
  mutate(pct = 100 * n / sum(n)) %>%
  ungroup()

pal4 <- c(
  "Complete"                               = "#2ca02c",
  "Monotone dropout"                       = "#ff7f0e",
  "Excluded\n(intermittent / all-missing)" = "#d62728"
)

fig4a <- ggplot(fig4_excl,
                aes(x = grp, y = n, fill = miss_cat)) +
  geom_col(position = position_dodge(width = 0.75),
           width = 0.65, colour = "white") +
  geom_text(aes(label = n),
            position = position_dodge(width = 0.75),
            vjust = -0.45, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = pal4, name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(x = NULL, y = "Number of episodes",
       title = "Episode Counts by Missing Pattern and Race Group") +
  theme(legend.position = "top",
        plot.title = element_text(face = "bold"),
        panel.grid.major.x = element_blank())

fig4b <- ggplot(fig4_excl,
                aes(x = grp, y = pct, fill = miss_cat)) +
  geom_col(position = position_dodge(width = 0.75),
           width = 0.65, colour = "white") +
  geom_text(aes(label = sprintf("%.1f%%", pct)),
            position = position_dodge(width = 0.75),
            vjust = -0.45, size = 3.3, fontface = "bold") +
  scale_fill_manual(values = pal4, name = NULL) +
  scale_y_continuous(labels = label_percent(scale = 1),
                     expand = expansion(mult = c(0, 0.12))) +
  labs(x = NULL, y = "% within race group",
       title = "Within-Group % by Missing Pattern") +
  theme(legend.position = "top",
        plot.title = element_text(face = "bold"),
        panel.grid.major.x = element_blank())

fig4 <- (fig4a | fig4b) +
  plot_annotation(
    title    = "Figure 4 — Monotone-Eligible Episodes: Black vs Non-Black (CHTN)",
    subtitle = sprintf(
      "Green = complete (all %d visits observed)  |  Orange = monotone dropout  |  Red = excluded",
      K
    ),
    theme = theme(
      plot.title    = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 9, colour = "grey30")
    )
  )


## ============================================================
## 9)  Figure 5 — Dropout timing by race group
##     Panel A: Retention curve (% still observed at each visit)
##     Panel B: New dropout events per visit
##     (Analogue of Fig 5 in CHAP_missing_pattern.R §11)
## ============================================================

## Panel A data: % observed at each visit × group
retention_dat <- panel %>%
  group_by(grp, visit_num) %>%
  summarize(
    n_subj     = n_distinct(episode),
    n_observed = sum(obs_sbp),
    pct_obs    = 100 * n_observed / n_subj,
    .groups    = "drop"
  )

## Panel B data: new dropout events per visit × group
dropout_events <- miss_class %>%
  filter(miss_pattern == "monotone_dropout") %>%
  count(grp, first_miss, name = "n_new_dropout") %>%
  rename(visit_num = first_miss) %>%
  complete(grp, visit_num = 1:K, fill = list(n_new_dropout = 0L))

## Group totals for legend labels
grp_totals <- miss_class %>% count(grp, name = "N_grp")

grp_colours <- c("Non-Black" = "#1f77b4", "Black" = "#d62728")

fig5a <- retention_dat %>%
  left_join(grp_totals, by = "grp") %>%
  ggplot(aes(x = visit_num, y = pct_obs, colour = grp, group = grp)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2.5, shape = 16) +
  scale_colour_manual(
    values = grp_colours, name = NULL,
    labels = sprintf("%s (N=%d)", grp_totals$grp, grp_totals$N_grp)
  ) +
  scale_x_continuous(name = NULL, breaks = 1:K) +
  scale_y_continuous(
    name   = "% episodes observed",
    limits = c(0, 100),
    labels = label_percent(scale = 1)
  ) +
  labs(
    title    = "Figure 5 — Dropout Timing by Race Group (CHTN)",
    subtitle = "Panel A: Retention — % still observed at each visit"
  ) +
  theme(
    plot.title      = element_text(face = "bold"),
    plot.subtitle   = element_text(size = 9, colour = "grey30"),
    legend.position = "top",
    axis.text.x     = element_blank(),
    axis.ticks.x    = element_blank()
  )

fig5b <- dropout_events %>%
  ggplot(aes(x = visit_num, y = n_new_dropout, fill = grp)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65,
           colour = "white") +
  scale_fill_manual(values = grp_colours, name = NULL) +
  scale_x_continuous(name = "Visit number", breaks = 1:K) +
  scale_y_continuous(name   = "New dropouts",
                     expand = expansion(mult = c(0, 0.12))) +
  labs(subtitle = "Panel B: New dropout events — first missing visit per episode") +
  theme(
    plot.subtitle      = element_text(size = 9, colour = "grey30"),
    legend.position    = "none",
    panel.grid.major.x = element_blank()
  )

## At-risk table (printed to console)
cat("\n=== At-risk table (n observed per visit) ===\n")
retention_dat %>%
  select(grp, visit_num, n_observed) %>%
  pivot_wider(names_from = grp, values_from = n_observed) %>%
  rename(Visit = visit_num) %>%
  print(n = Inf)

fig5 <- fig5a / fig5b + plot_layout(heights = c(2, 1.4))


## ============================================================
## 10) Figure 6 — Subject-Level Missing Pattern  (WHOLE SAMPLE)
##     No group stratification.  All monotone-eligible episodes
##     sorted by: pattern (complete first) → desc(first_miss) → episode ID.
##     Section labels on right margin: pattern × n.
## ============================================================

## Monotone-eligible only (complete + monotone_dropout)
mono_eligible <- miss_class %>%
  filter(miss_pattern %in% c("complete", "monotone_dropout"))

cat(sprintf(
  "\nMonotone-eligible (whole sample): %d / %d episodes (%.1f%%)\n  Complete: %d | Monotone dropout: %d\n",
  nrow(mono_eligible), N_ep, 100 * nrow(mono_eligible) / N_ep,
  sum(mono_eligible$miss_pattern == "complete"),
  sum(mono_eligible$miss_pattern == "monotone_dropout")
))

## Sorting: pattern rank → later first-miss first → episode ID
ep_order <- mono_eligible %>%
  mutate(pattern_rank = ifelse(miss_pattern == "complete", 1L, 2L)) %>%
  arrange(pattern_rank, desc(first_miss), as.character(episode)) %>%
  mutate(subj_row = row_number())

## Tile data for the heatmap
panel_fig6 <- panel %>%
  semi_join(mono_eligible %>% select(episode), by = "episode") %>%
  left_join(ep_order %>% select(episode, subj_row, miss_pattern),
            by = "episode") %>%
  mutate(obs_label = factor(obs_sbp, levels = c(0L, 1L),
                            labels = c("Missing", "Observed")))

## Separator row between complete and monotone_dropout
pattern_sep_row <- ep_order %>%
  filter(miss_pattern == "complete") %>%
  summarize(max_row = max(subj_row)) %>%
  pull(max_row)

## Section labels for right margin
section_labels_6 <- ep_order %>%
  group_by(miss_pattern) %>%
  summarize(
    mid_row = (min(subj_row) + max(subj_row)) / 2,
    n_sec   = n(),
    .groups = "drop"
  ) %>%
  mutate(
    label = sprintf("%s\n(n = %d)",
                    ifelse(miss_pattern == "complete",
                           "Complete", "Monotone\ndropout"),
                    n_sec)
  )

fig6 <- ggplot(
  panel_fig6,
  aes(x    = visit_num,
      y    = reorder(as.factor(episode), subj_row),
      fill = obs_label)
) +
  geom_tile(colour = NA, linewidth = 0) +
  ## Pattern boundary (white line between complete / dropout)
  geom_hline(yintercept = pattern_sep_row + 0.5,
             colour = "white", linewidth = 1.2) +
  ## Section labels on right margin
  geom_text(
    data = section_labels_6,
    aes(x = K + 0.6, y = mid_row, label = label),
    inherit.aes = FALSE,
    hjust = 0, size = 3.0, lineheight = 0.88, colour = "grey20"
  ) +
  scale_fill_manual(
    values = c("Observed" = "#4daf4a", "Missing" = "#e41a1c"),
    name   = NULL
  ) +
  scale_x_continuous(
    name   = "Visit",
    breaks = 1:K,
    limits = c(0.5, K + 3.2),    # right margin space for labels
    expand = c(0, 0)
  ) +
  scale_y_discrete(expand = c(0, 0)) +
  labs(
    title    = "Subject-Level Missing Pattern: Visits 1–10",
    subtitle = sprintf(
      "N = %d  (complete = %d, monotone dropout = %d)",
      nrow(mono_eligible),
      sum(mono_eligible$miss_pattern == "complete"),
      sum(mono_eligible$miss_pattern == "monotone_dropout")
    ),
    y = "Subject Cluster"
  ) +
  theme(
    axis.text.y     = element_blank(),
    axis.ticks.y    = element_blank(),
    panel.grid      = element_blank(),
    legend.position = "top",
    plot.title      = element_text(face = "bold"),
    plot.subtitle   = element_text(size = 9, colour = "grey30")
  )


## ============================================================
## 11) Per-visit missingness table (console)
## ============================================================
cat("\n=== Per-visit missingness by race group ===\n")
panel %>%
  group_by(grp, visit_num) %>%
  summarize(
    n_obs    = sum(obs_sbp),
    n_miss   = sum(sbp_mono_miss),
    pct_miss = round(100 * n_miss / (n_obs + n_miss), 1),
    .groups  = "drop"
  ) %>%
  pivot_wider(
    names_from  = grp,
    values_from = c(n_obs, n_miss, pct_miss),
    names_glue  = "{grp}_{.value}"
  ) %>%
  rename(Visit = visit_num) %>%
  print(n = Inf)


## ============================================================
## 11b) SBP Trajectory Visualisations — whole sample, no groups
##      Fig T1: Spaghetti (individual + group mean overlay)
##      Fig T2: Mean ± SD/SE ribbon
##      Fig T3: Violin + boxplot per visit
##      Fig T4: Variability trajectories (SD and CV per visit)
## ============================================================
dat_traj <- panel %>% filter(!is.na(sbp))

## Per-visit summary statistics
visit_summary <- dat_traj %>%
  group_by(visit_num) %>%
  summarize(
    n          = n(),
    mean_sbp   = mean(sbp),
    sd_sbp     = sd(sbp),
    se_sbp     = sd_sbp / sqrt(n),
    median_sbp = median(sbp),
    q25        = quantile(sbp, 0.25),
    q75        = quantile(sbp, 0.75),
    cv_sbp     = 100 * sd_sbp / mean_sbp,
    .groups    = "drop"
  )

cat("\n=== SBP per-visit summary ===\n")
print(visit_summary, n = Inf)

## -- Fig T1: Spaghetti plot ----------------------------------
fig_t1 <- ggplot(dat_traj,
                 aes(x = visit_num, y = sbp, group = episode)) +
  geom_line(alpha = 0.07, colour = "steelblue", linewidth = 0.3) +
  geom_line(data = visit_summary,
            aes(x = visit_num, y = mean_sbp, group = 1),
            colour = "firebrick", linewidth = 1.5, inherit.aes = FALSE) +
  geom_point(data = visit_summary,
             aes(x = visit_num, y = mean_sbp),
             colour = "firebrick", size = 3.0, inherit.aes = FALSE) +
  scale_x_continuous(name = "Visit", breaks = 1:K) +
  scale_y_continuous(name = "Systolic BP (mmHg)") +
  labs(
    title    = "Spaghetti Plot: Individual SBP Trajectories",
    subtitle = sprintf(
      "N = %d (%d observed rows)  |  Blue = individual; Red = group mean",
      n_distinct(dat_traj$episode), nrow(dat_traj)
    )
  ) +
  theme(plot.title    = element_text(face = "bold"),
        plot.subtitle = element_text(size = 9, colour = "grey30"))

## -- Fig T2: Mean ± SD/SE ribbon -----------------------------
fig_t2 <- ggplot(visit_summary, aes(x = visit_num)) +
  geom_ribbon(aes(ymin = mean_sbp - sd_sbp, ymax = mean_sbp + sd_sbp),
              fill = "steelblue", alpha = 0.18) +
  geom_ribbon(aes(ymin = mean_sbp - se_sbp, ymax = mean_sbp + se_sbp),
              fill = "steelblue", alpha = 0.45) +
  geom_line(aes(y = mean_sbp), colour = "steelblue", linewidth = 1.3) +
  geom_point(aes(y = mean_sbp), colour = "steelblue", size = 3.0) +
  geom_text(aes(y = mean_sbp + sd_sbp + 2.0,
                label = sprintf("n=%d", n)),
            size = 2.9, colour = "grey40") +
  scale_x_continuous(name = "Visit number", breaks = 1:K) +
  scale_y_continuous(name = "Systolic BP (mmHg)") +
  labs(
    title    = "Fig T2 — Mean SBP Trajectory ± SD  (Whole Sample)",
    subtitle = "Dark ribbon = ±1 SE  |  Light ribbon = ±1 SD  |  n = observations per visit"
  ) +
  theme(plot.title    = element_text(face = "bold"),
        plot.subtitle = element_text(size = 9, colour = "grey30"))

## -- Fig T3: Violin + boxplot per visit ----------------------
fig_t3 <- ggplot(dat_traj, aes(x = factor(visit_num), y = sbp)) +
  geom_violin(fill = "#9ecae1", colour = "grey60",
              alpha = 0.60, trim = TRUE, scale = "width") +
  geom_boxplot(width = 0.18, outlier.size = 0.8, outlier.alpha = 0.35,
               fill = "white", colour = "grey25", linewidth = 0.5) +
  stat_summary(fun = mean, geom = "point",
               colour = "firebrick", size = 2.5, shape = 18) +
  scale_x_discrete(name = "Visit number") +
  scale_y_continuous(name = "Systolic BP (mmHg)") +
  labs(
    title    = "Fig T3 — SBP Distribution per Visit  (Violin + Boxplot, Whole Sample)",
    subtitle = "Violin = kernel density  |  Box = IQR  |  Diamond = mean"
  ) +
  theme(plot.title    = element_text(face = "bold"),
        plot.subtitle = element_text(size = 9, colour = "grey30"))

## -- Fig T4: Variability trajectories (SD + CV) --------------
fig_t4a <- ggplot(visit_summary, aes(x = visit_num)) +
  geom_line(aes(y = sd_sbp), colour = "#e34a33", linewidth = 1.2) +
  geom_point(aes(y = sd_sbp), colour = "#e34a33", size = 2.8) +
  geom_text(aes(y = sd_sbp + 0.4, label = sprintf("%.1f", sd_sbp)),
            size = 2.9, colour = "grey40", vjust = 0) +
  scale_x_continuous(name = NULL, breaks = 1:K) +
  scale_y_continuous(name = "SD of SBP (mmHg)") +
  labs(subtitle = "Panel A: Standard deviation") +
  theme(plot.subtitle = element_text(size = 9, colour = "grey30"),
        axis.text.x   = element_blank(),
        axis.ticks.x  = element_blank())

fig_t4b <- ggplot(visit_summary, aes(x = visit_num)) +
  geom_line(aes(y = cv_sbp), colour = "#756bb1", linewidth = 1.2) +
  geom_point(aes(y = cv_sbp), colour = "#756bb1", size = 2.8) +
  geom_text(aes(y = cv_sbp + 0.15, label = sprintf("%.1f%%", cv_sbp)),
            size = 2.9, colour = "grey40", vjust = 0) +
  scale_x_continuous(name = "Visit", breaks = 1:K) +
  scale_y_continuous(name = "CV of SBP (%)") +
  labs(subtitle = "Panel B: Coefficient of variation  (SD/mean × 100)") +
  theme(plot.subtitle = element_text(size = 9, colour = "grey30"))

fig_t4 <- (fig_t4a / fig_t4b) +
  plot_annotation(
    title    = "SBP Variability Trajectories",
    subtitle = "Orange = SD trajectory  |  Purple = CV trajectory",
    theme    = theme(
      plot.title    = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 9, colour = "grey30")
    )
  )

## ============================================================
## 13) Save all figures to PDF  (14 pages)
## ============================================================
pdf("CHAP_Mean_traj.pdf", width = 11, height = 7.5)
print(fig_t1)                          # p 10
dev.off()

pdf("CHAP_Var_traj.pdf", width = 11, height = 7.5)
print(fig_t4)                          # p 10
dev.off()

pdf("CHAP_Monotone.pdf", width = 11, height = 7.5)
## Fig 6: full subject-level heatmap (whole sample, no groups)
print(fig6)                            # p 9
dev.off()
## ============================================================
## 12) Centering, weight decomposition, and glmmTMB model
##
##     Predictors:
##       age_c          = age_bl − overall mean(age_bl)
##       mean_weight_lb = episode-level mean of weight_lb,
##                        centred by the grand mean of episode means
##       dev_weight_lb  = weight_lb − (uncentred episode mean)
##       ckd            = 0/1 (unchanged)
##
##     Model:
##       Mean      : log(sbp) ~ visit_num + I(visit_num^2) +
##                              age_c + mean_weight_lb + dev_weight_lb +
##                              ckd + (1 | mrn)
##       Dispersion: ~ visit_num + I(visit_num^2) +
##                     age_c + mean_weight_lb + dev_weight_lb + ckd
## ============================================================
message("\n--- Centering predictors and decomposing weight ---")

## Work only on observed rows
dat_obs <- panel %>% filter(!is.na(sbp))

## Age: centre by overall mean (computed from observed rows)
age_grand_mean <- mean(dat_obs$age_bl, na.rm = TRUE)
cat(sprintf("  Age grand mean (observed rows): %.3f years\n", age_grand_mean))

## Weight: episode-level mean (from ALL panel rows, not just observed SBP,
##         to use the most complete weight picture)
ep_weight_mean <- panel %>%
  filter(!is.na(weight_lb)) %>%
  group_by(episode) %>%
  summarize(ep_mean_wt = mean(weight_lb, na.rm = TRUE), .groups = "drop")

## Grand mean of episode-level means → used to centre mean_weight_lb
wt_grand_mean <- mean(ep_weight_mean$ep_mean_wt, na.rm = TRUE)
cat(sprintf("  Weight grand mean (episode means): %.3f lbs\n", wt_grand_mean))

## Build modelling dataset: join episode means, compute deviations
dat_final <- dat_obs %>%
  left_join(ep_weight_mean, by = "episode") %>%
  mutate(
    ## Centred predictors
    age_c          = age_bl - age_grand_mean,
    ## Between-person: episode mean centred by grand mean
    mean_weight_lb = ep_mean_wt - wt_grand_mean,
    ## Within-person: deviation from (uncentred) episode mean
    dev_weight_lb  = weight_lb - ep_mean_wt
  )

## Verify centering
cat(sprintf("  Centering checks (should be ~0):\n"))
cat(sprintf("    mean(age_c)          = % .6f\n", mean(dat_final$age_c, na.rm = TRUE)))
cat(sprintf("    mean(mean_weight_lb) = % .6f\n", mean(dat_final$mean_weight_lb, na.rm = TRUE)))
cat(sprintf("    mean(dev_weight_lb)  = % .6f\n", mean(dat_final$dev_weight_lb, na.rm = TRUE)))
cat(sprintf("  dat_final: %d observed rows  |  %d episodes\n",
            nrow(dat_final), n_distinct(dat_final$episode)))

## ---- Fit heteroskedastic Gaussian model --------------------
form_mu_v   <- sbp ~ visit_num + I(visit_num^2) +
                           age_c + mean_weight_lb + dev_weight_lb + ckd +
                           (1 | mrn)
form_disp_v <- ~           visit_num + I(visit_num^2) +
                           age_c + mean_weight_lb + dev_weight_lb + ckd

library(glmmTMB)
message("Fitting glmmTMB (heteroskedastic Gaussian, centred predictors)...")
t_tmb_v <- system.time({
  fit_tmb_v <- glmmTMB(
    formula     = form_mu_v,
    dispformula = form_disp_v,
    data        = dat_final,
    family      = t_family(link = "identity")
  )
})


cat(sprintf("\n========== glmmTMB: Heteroskedastic Gaussian (visit scale, centred) ==========\n"))
print(summary(fit_tmb_v))
cat(sprintf(
  "\nFit diagnostics:\n  Hessian PD   : %s\n  Convergence  : %d  (0 = OK)\n  AIC          : %.4f\n  Log-Lik      : %.4f\n  Wall time    : %.1f s\n",
  fit_tmb_v$sdr$pdHess,
  fit_tmb_v$fit$convergence,
  AIC(fit_tmb_v),
  as.numeric(logLik(fit_tmb_v)),
  t_tmb_v["elapsed"]
))


## ============================================================
## 12b) Model Diagnostics
##      Fig D: 3-row × 2-column page
##        D1 — Residuals vs Fitted (location sub-model)
##        D2 — Q-Q plot of Pearson residuals
##        D3 — Scale-Location (heteroskedasticity check)
##        D4 — Residuals by visit number (boxplot overlay)
##        D5 — Residual histogram with reference density
##        D6 — Observed vs Fitted SBP
## ============================================================
message("Building model diagnostics...")

library(ggplot2)
library(dplyr)
library(patchwork)

library(dplyr)
library(ggplot2)
library(patchwork)

## fitted mean
fitted_sbp <- as.numeric(predict(fit_tmb_v, type = "response"))

## raw response residuals
res_raw <- dat_final$sbp - fitted_sbp

## fitted dispersion predictor
## for glmmTMB, type = "disp" is on the dispersion scale
disp_hat <- as.numeric(predict(fit_tmb_v, type = "disp"))

## for Gaussian / Student-t families in glmmTMB, the dispersion model is on log(sigma)
sd_hat <- exp(disp_hat)

## standardized residuals
res_std <- res_raw / sd_hat

sqrt_abs_res <- sqrt(abs(res_std))

diag_df <- dat_final %>%
  mutate(
    fitted_val   = fitted_sbp,
    res_raw      = res_raw,
    res_std      = res_std,
    sqrt_abs_res = sqrt_abs_res,
    obs_sbp      = sbp
  )

stopifnot(nrow(diag_df) == length(fitted_sbp),
          nrow(diag_df) == length(res_raw),
          nrow(diag_df) == length(res_std))

## ============================================================
## 12b) Model Diagnostics
##      Fig D: 3-row × 2-column page
## ============================================================

## D1: Standardized residuals vs fitted
fig_d1 <- ggplot(diag_df, aes(x = fitted_val, y = res_std)) +
  geom_point(alpha = 0.25, size = 0.9, colour = "steelblue") +
  geom_hline(yintercept = 0, colour = "red", linewidth = 0.8) +
  geom_smooth(method = "loess", se = TRUE,
              colour = "darkorange", linewidth = 0.9, fill = "orange") +
  labs(
    title = "D1 — Standardized Residuals vs Fitted",
    x = "Fitted SBP", y = "Standardized residuals"
  ) +
  theme(plot.title = element_text(face = "bold", size = 10))

## D2: Q-Q plot
fig_d2 <- ggplot(diag_df, aes(sample = res_std)) +
  stat_qq(alpha = 0.35, size = 0.9, colour = "steelblue") +
  stat_qq_line(colour = "red", linewidth = 0.8) +
  labs(
    title = "D2 — Q-Q Plot of Standardized Residuals",
    x = "Theoretical quantiles", y = "Sample quantiles"
  ) +
  theme(plot.title = element_text(face = "bold", size = 10))

## D3: Scale-location
fig_d3 <- ggplot(diag_df, aes(x = fitted_val, y = sqrt_abs_res)) +
  geom_point(alpha = 0.25, size = 0.9, colour = "steelblue") +
  geom_smooth(method = "loess", se = TRUE,
              colour = "darkorange", linewidth = 0.9, fill = "orange") +
  labs(
    title = "D3 — Heteroskedasticity",
    x = "Fitted SBP",
    y = expression(sqrt("|Standardized residuals|"))
  ) +
  theme(plot.title = element_text(face = "bold", size = 10))

## D4: Residuals by visit number
fig_d4 <- ggplot(diag_df, aes(x = factor(visit_num), y = res_std)) +
  geom_jitter(alpha = 0.20, size = 0.7, colour = "steelblue", width = 0.18) +
  geom_boxplot(aes(group = visit_num), alpha = 0, outlier.shape = NA,
               colour = "grey35", linewidth = 0.5) +
  geom_hline(yintercept = 0, colour = "red", linewidth = 0.8) +
  labs(
    title = "D4 — Standardized Residuals by Visit Number",
    x = "Visit number", y = "Standardized residuals"
  ) +
  theme(plot.title = element_text(face = "bold", size = 10))

## D5: Histogram
fig_d5 <- ggplot(diag_df, aes(x = res_std)) +
  geom_histogram(aes(y = after_stat(density)), bins = 45,
                 fill = "steelblue", colour = "white", alpha = 0.75) +
  stat_function(
    fun = dnorm,
    args = list(mean = mean(diag_df$res_std, na.rm = TRUE),
                sd   = sd(diag_df$res_std, na.rm = TRUE)),
    colour = "red", linewidth = 1.0
  ) +
  labs(
    title = "D5 — Residual Distribution",
    x = "Standardized residuals", y = "Density"
  ) +
  theme(plot.title = element_text(face = "bold", size = 10))

## D6: Observed vs fitted
r2_val <- cor(diag_df$obs_sbp, diag_df$fitted_val, use = "complete.obs")^2

fig_d6 <- ggplot(diag_df, aes(x = fitted_val, y = obs_sbp)) +
  geom_point(alpha = 0.20, size = 0.8, colour = "steelblue") +
  geom_abline(slope = 1, intercept = 0, colour = "red", linewidth = 0.9) +
  annotate("text", x = -Inf, y = Inf,
           label = sprintf("r² = %.3f", r2_val),
           hjust = -0.2, vjust = 1.4, size = 4, fontface = "bold") +
  labs(
    title = "D6 — Observed vs Fitted SBP",
    x = "Fitted SBP", y = "Observed SBP"
  ) +
  theme(plot.title = element_text(face = "bold", size = 10))

fig_diag <- (fig_d1 | fig_d2) / (fig_d3 | fig_d4) / (fig_d5 | fig_d6) +
  plot_annotation(
    title = "glmmTMB Model Diagnostics: Heteroskedastic Student-t Model",
    subtitle = paste0(
      "Location: SBP ~ visit + visit² + age_c + mean_weight_lb + dev_weight_lb + CKD + (1|mrn)\n",
      "Dispersion: ~ visit + visit² + age_c + mean_weight_lb + dev_weight_lb + CKD"
    ),
    theme = theme(
      plot.title    = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 8, colour = "grey30")
    )
  )

fig_diag
pdf("glmmTMBDiag.pdf", width = 7, height = 11)
## Fig D: model diagnostics
print(fig_diag)                        # p 14

dev.off()
## ============================================================
## 13) Save all figures to PDF  (14 pages)
## ============================================================
pdf("CHAP_final_analysis.pdf", width = 11, height = 7.5)

  ## Figs 1–3: Non-Black
  print(figs_list[["Non-Black"]]$fig1)   # p 1
  print(figs_list[["Non-Black"]]$fig2)   # p 2
  print(figs_list[["Non-Black"]]$fig3)   # p 3

  ## Figs 1–3: Black
  print(figs_list[["Black"]]$fig1)       # p 4
  print(figs_list[["Black"]]$fig2)       # p 5
  print(figs_list[["Black"]]$fig3)       # p 6

  ## Fig 4: complete vs dropout by race
  print(fig4)                            # p 7

  ## Fig 5: dropout timing
  print(fig5)                            # p 8

  ## Fig 6: full subject-level heatmap (whole sample, no groups)
  print(fig6)                            # p 9

  ## Figs T1–T4: SBP trajectory visualisations (whole sample)
  print(fig_t1)                          # p 10
  print(fig_t2)                          # p 11
  print(fig_t3)                          # p 12
  print(fig_t4)                          # p 13

  ## Fig D: model diagnostics
  print(fig_diag)                        # p 14

dev.off()

message("Saved → CHAP_final_analysis.pdf  (14 pages)")
cat("\n=== Analysis complete ===\n")
