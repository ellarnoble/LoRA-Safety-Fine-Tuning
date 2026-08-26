# ---------------------------------------------------------------------
# (1) Preliminaries
library(tidyverse)
library(lme4)
library(nnet)
library(car)
library(broom)
library(knitr)
library(emmeans)
library(scales)
library(effectsize)
library(performance)
library(ggeffects)
library(jsonlite)
library(blme)

# Convert response labels into a factor variable
finetune_safety_results_long$lg_response_label <- as.factor(finetune_safety_results_long$lg_response_label)

# Convert hct labels into consistent labels
finetune_safety_results_long <- finetune_safety_results_long %>%
  mutate(hct = as.integer(round(as.numeric(hct))))

# Rename base models for consistency with main text
finetune_safety_results_long <- finetune_safety_results_long %>%
  mutate(basemodel = dplyr::recode(basemodel,
                                   "Falcon7B" = "Falcon-7B",
                                   "Mistral7B" = "Mistral-7B",
                                   "Qwen7B" = "Qwen-7B"
  ))
basemodel_order <- c("Mistral-7B", "Qwen-7B", "Falcon-7B")

# ----------------------------------------------------------------------
# (2) HCT/SRR failure modes
# ----------------------------------------------------------------------
is_unsafe <- function(x) as.integer(x == "unsafe")

prepared <- finetune_safety_results_long %>%
  mutate(
    H = as.integer(hct),
    G = is_unsafe(lg_prompt_label),
    R_safe = 1L - is_unsafe(lg_response_label),
    hct_correct = as.integer(H == G)
  )

failure_counts <- prepared %>%
  group_by(basemodel, condition) %>%
  summarise(
    working = sum(G == 1 & H == 1 & R_safe == 1, na.rm = TRUE),
    detection_failure = sum(G == 1 & H == 0 & R_safe == 0, na.rm = TRUE),
    compliance_failure = sum(G == 1 & H == 1 & R_safe == 0, na.rm = TRUE),
    recovery = sum(G == 1 & H == 0 & R_safe == 1, na.rm = TRUE),
    .groups = "drop"
  )

failure_props <- failure_counts %>%
  mutate(
    unsafe_n = working + detection_failure + compliance_failure + recovery,
    working = working / unsafe_n,
    detection = detection_failure / unsafe_n,
    compliance = compliance_failure / unsafe_n,
    recovery = recovery / unsafe_n
  ) %>%
  select(basemodel, condition, working, detection, compliance, recovery) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

placement_summary <- failure_props %>%
  mutate(
    placement = case_when(
      condition == "baseline" ~ "baseline",
      condition == "full" ~ "full",
      TRUE ~ str_extract(condition, "early|middle|late")
    ),
    placement = factor(placement, levels = c("baseline", "early", "middle", "late", "full"))
  ) %>%
  group_by(basemodel, placement) %>%
  summarise(across(c(working, detection, compliance, recovery), mean), .groups = "drop") %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

plot_data <- placement_summary %>%
  select(basemodel, placement, working, recovery, compliance, detection) %>%
  pivot_longer(
    cols = c(working, recovery, compliance, detection),
    names_to = "outcome",
    values_to = "proportion"
  ) %>%
  mutate(
    outcome = factor(outcome,
                     levels = c("detection", "compliance", "recovery", "working"),
                     labels = c("Detection failure", "Compliance failure", "Recovery failure", "Working")
    )
  )

outcome_colors <- c(
  "Working"             = "darkseagreen4",
  "Recovery failure"    = "#D2CDF6",
  "Compliance failure"  = "slategray",
  "Detection failure"   = "red4"
)

failure_modes <- ggplot(plot_data, aes(x = placement, y = proportion, color = outcome, group = outcome)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.2) +
  facet_wrap(~ basemodel, nrow = 1) +
  scale_color_manual(values = outcome_colors, name = NULL,
                     breaks = c("Working", "Recovery failure", "Compliance failure", "Detection failure")) +
  scale_x_discrete(labels = c(
    "baseline" = "Baseline", "early" = "Early", "middle" = "Middle",
    "late" = "Late", "full" = "Full FT"
  )) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  labs(x = "Fine-tuning condition", y = "Percentage of unsafe prompts") +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "top",
    strip.text = element_text(face = "bold", color = "#52514e"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text  = element_text(color = "#52514e"),
    axis.title = element_text(color = "#52514e"),
    legend.text = element_text(color = "#52514e")
  )
failure_modes
ggsave("failure-modes-lineplot.png", plot = failure_modes,
       path = "C:/Users/Ella/OneDrive/Documents/Ella/UniMsc/DISSERTATION/Final/Write-up",
       width = 8, height = 6, dpi = 300, bg = "white")

# ------------------------------------------------------------------------
# (3) Modelling the effect of condition on SRR and HCT accuracy
# ------------------------------------------------------------------------
sig_stars <- function(p) case_when(p < .001 ~ "***", p < .01 ~ "**", p < .05 ~ "*", TRUE ~ "")

fit_unified <- function(outcome_var, G_value) {
  form <- as.formula(paste(outcome_var, "~ condition * source * basemodel + (1 | index)"))
  data_subset <- prepared %>% filter(G == G_value)
  glmer(form, data = data_subset, family = binomial(), nAGQ = 0)
}
m_srr_unsafe <- fit_unified("R_safe", G_value = 1)
m_srr_safe <- fit_unified("R_safe", G_value = 0)
m_hct_unsafe <- fit_unified("hct_correct", G_value = 1)
m_hct_safe <- fit_unified("hct_correct", G_value = 0)

extract_all_rows <- function(model, metric_name, prompt_type) {
  a <- car::Anova(model, type = "II")
  r2 <- performance::r2_nakagawa(model)
  icc_val <- performance::icc(model)
  tibble(
    metric = metric_name,
    prompt_type = prompt_type,
    term = rownames(a),
    chisq = a$Chisq,
    p_raw = a$`Pr(>Chisq)`,
    marginal_R2 = r2$R2_marginal,
    conditional_R2 = r2$R2_conditional,
    icc_adjusted = icc_val$ICC_adjusted,
    icc_unadjusted = icc_val$ICC_unadjusted,
    is_singular = isSingular(model)
  )
}

results <- bind_rows(
  extract_all_rows(m_srr_unsafe, "SRR", "Unsafe"),
  extract_all_rows(m_srr_safe, "SRR", "Safe"),
  extract_all_rows(m_hct_unsafe, "HCT", "Unsafe"),
  extract_all_rows(m_hct_safe, "HCT", "Safe")
) %>%
  group_by(metric, prompt_type) %>%
  mutate(p_holm = p.adjust(p_raw, method = "holm")) %>%
  ungroup() %>%
  mutate(stars = sig_stars(p_holm))

print(results, n = Inf, width = Inf)

# Refit SRR Safe prompts model due to 
m_srr_safe_blme <- bglmer(R_safe ~ condition * source * basemodel + (1 | index),
                          data = safe_data, family = binomial(),
                          fixef.prior = normal(cov = diag(9, ncol = 1)))

# Portion of variance explained by each variable

compute_term_r2 <- function(outcome_var, G_value) {
  data_subset <- prepared %>% filter(G == G_value)
  
  full_formula <- as.formula(paste(outcome_var, "~ condition * source * basemodel + (1 | index)"))
  m_full <- glmer(full_formula, data = data_subset, family = binomial(), nAGQ = 0)
  r2_full <- performance::r2_nakagawa(m_full)$R2_marginal
  
  reduced_formulas <- list(
    "Condition" = paste(outcome_var, "~ source * basemodel + (1 | index)"),
    "Source" = paste(outcome_var, "~ condition * basemodel + (1 | index)"),
    "Base model" = paste(outcome_var, "~ condition * source + (1 | index)"),
    "Condition x Source" = paste(outcome_var, "~ condition + source + basemodel + condition:basemodel + source:basemodel + (1 | index)"),
    "Condition x Base model" = paste(outcome_var, "~ condition + source + basemodel + condition:source + source:basemodel + (1 | index)"),
    "Source x Base model" = paste(outcome_var, "~ condition + source + basemodel + condition:source + condition:basemodel + (1 | index)"),
    "Condition x Source x Base model" = paste(outcome_var, "~ condition + source + basemodel + condition:source + condition:basemodel + source:basemodel + (1 | index)")
  )
  
  purrr::map_dfr(names(reduced_formulas), function(term_name) {
    m_reduced <- glmer(as.formula(reduced_formulas[[term_name]]), data = data_subset, family = binomial(), nAGQ = 0)
    r2_reduced <- performance::r2_nakagawa(m_reduced)$R2_marginal
    tibble(
      term = term_name,
      r2_full = r2_full,
      r2_reduced = r2_reduced,
      delta_r2 = r2_full - r2_reduced
    )
  })
}

term_r2_results <- bind_rows(
  compute_term_r2("R_safe", G_value = 1) %>% mutate(metric = "SRR", prompt_type = "Unsafe"),
  compute_term_r2("R_safe", G_value = 0) %>% mutate(metric = "SRR", prompt_type = "Safe"),
  compute_term_r2("hct_correct", G_value = 1) %>% mutate(metric = "HCT", prompt_type = "Unsafe"),
  compute_term_r2("hct_correct", G_value = 0) %>% mutate(metric = "HCT", prompt_type = "Safe")
) %>%
  mutate(delta_r2_pct = round(delta_r2 * 100, 2)) %>%
  select(metric, prompt_type, term, delta_r2_pct, r2_full, r2_reduced)

print(term_r2_results, n = Inf, width = Inf)

# ------------------------------------------------------------------------------
# Separate models per sub-group
# -------------------------------------------------------------------------------
subgroups <- list(
  `Unsafe: WildGuard` = prepared %>% filter(G == 1, source == "wildguard") %>% mutate(index = as.factor(index)),
  `Unsafe: BeaverTails` = prepared %>% filter(G == 1, source == "beavertails") %>% mutate(index = as.factor(index)),
  `Safe: WildGuard` = prepared %>% filter(G == 0, source == "wildguard") %>% mutate(index = as.factor(index)),
  `Safe: BeaverTails` = prepared %>% filter(G == 0, source == "beavertails") %>% mutate(index = as.factor(index))
)

fit_family <- function(outcome) {
  map(subgroups, function(d) {
    if (outcome == "hct_correct") d <- d %>% filter(!is.na(hct))
    pooled <- glmer(as.formula(paste(outcome, "~ condition + (1|index)")),
                    data = d, family = binomial(), nAGQ = 0)
    interaction <- glmer(as.formula(paste(outcome, "~ condition * basemodel + (1|index)")),
                         data = d, family = binomial(), nAGQ = 0)
    list(pooled = pooled, interaction = interaction)
  })
}

summarise_family <- function(fits, metric_label) {
  map_dfr(names(fits), function(subgroup) {
    pooled <- fits[[subgroup]]$pooled
    lrt    <- anova(pooled, fits[[subgroup]]$interaction)
    tibble(
      metric = metric_label,
      subgroup = subgroup,
      chisq = Anova(pooled, type = "II")$Chisq[1],
      p = Anova(pooled, type = "II")$`Pr(>Chisq)`[1],
      marginal_r2 = r2_nakagawa(pooled)$R2_marginal,
      interaction_chisq = lrt$Chisq[2],
      interaction_p = lrt$`Pr(>Chisq)`[2]
    )
  })
}

interaction_summary_tbl <- bind_rows(
  summarise_family(fit_family("R_safe"),      "SRR"),
  summarise_family(fit_family("hct_correct"), "HCT")
) %>%
  mutate(
    p_adj = p.adjust(p, method = "holm"),
    interaction_p_adj = p.adjust(interaction_p, method = "holm"),
    `Condition chisq` = paste0(round(chisq, 1), sig_stars(p_adj)),
    `Marginal R2` = sprintf("%.3f", marginal_r2),
    `Interaction chisq` = paste0(round(interaction_chisq, 1), sig_stars(interaction_p_adj))
  ) %>%
  select(metric, subgroup, `Condition chisq`, `Marginal R2`, `Interaction chisq`)
print(interaction_summary_tbl, n = Inf)

# ------------------------------------------------------------------------
# Which conditions are significantly better/worse than baseline
# ------------------------------------------------------------------------
extract_condition_or <- function(model, subgroup, basemodel) {
  s <- summary(model)$coefficients
  s <- s[rownames(s) != "(Intercept)", , drop = FALSE]
  tibble(
    subgroup = subgroup,
    basemodel = basemodel,
    condition = str_remove(rownames(s), "^condition"),
    OR = exp(s[, "Estimate"]),
    se = s[, "Std. Error"],
    p = s[, "Pr(>|z|)"]
  )
}

fit_per_basemodel_or <- function(outcome) {
  map_dfr(names(subgroups), function(subgroup) {
    d <- subgroups[[subgroup]]
    if (outcome == "hct_correct") d <- d %>% filter(!is.na(hct))
    map_dfr(basemodel_order, function(bm) {
      m <- glmer(as.formula(paste(outcome, "~ condition + (1|index)")),
                 data = d %>% filter(basemodel == bm), family = binomial(), nAGQ = 0)
      extract_condition_or(m, subgroup, bm)
    })
  })
}

condition_or_tbl_full <- fit_per_basemodel_or("R_safe") %>%
  tidyr::separate(subgroup, into = c("prompt_type", "source"), sep = ": ", remove = FALSE) %>%
  group_by(prompt_type, source, basemodel) %>%
  mutate(p_adj = p.adjust(p, method = "holm")) %>%
  ungroup() %>%
  mutate(result = case_when(
    se > 5 ~ "not estimable",
    p_adj < .05 & OR > 1 ~ "better than baseline",
    p_adj < .05 & OR < 1 ~ "worse than baseline",
    TRUE ~ "no significant difference"
  ))
condition_or_tbl_full$result <- factor(
  condition_or_tbl_full$result,
  levels = c("worse than baseline", "no significant difference",
             "not estimable", "better than baseline")
)

status_pal <- c(
  "worse than baseline"       = "red4",
  "no significant difference" = "snow",
  "not estimable"             = "grey40",
  "better than baseline"      = "darkseagreen4"
)
bubble_tbl <- condition_or_tbl_full %>%
  mutate(
    magnitude = abs(log(OR)),
    magnitude = ifelse(result == "not estimable", NA, magnitude),
    magnitude = pmin(magnitude, 3)
  )
ggplot(bubble_tbl, aes(x = basemodel, y = condition)) +
  geom_tile(fill = NA, color = "grey90", linewidth = 0.3) +
  geom_point(aes(fill = result, size = magnitude), shape = 21, color = "white",
             stroke = 0.4, alpha = 0.75, na.rm = TRUE) +
  geom_point(data = filter(bubble_tbl, result == "not estimable"),
             aes(fill = result), shape = 21, size = 3, color = "white",
             stroke = 0.4, alpha = 0.75, na.rm = TRUE, show.legend = FALSE) +
  facet_grid(prompt_type ~ source) +
  scale_fill_manual(values = status_pal, drop = FALSE, name = "SRR vs baseline") +
  scale_size_continuous(range = c(2, 9), name = "Effect size\n(|log OR|)") +
  labs(x = "Base model", y = "Condition") +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank(),
    strip.text = element_text(face = "bold")
  )

condition_or_tbl_full %>%
  select(prompt_type, source, basemodel, condition, OR, p_adj, result) %>%
  mutate(OR = round(OR, 2)) %>%
  arrange(prompt_type, source, basemodel, condition) %>%
  print(n = Inf, width = Inf)

# ---------------------------------------------------------------------
# (4) Per-condition summary table, feeding the placement/rank line plots
# ---------------------------------------------------------------------
raw_summary <- prepared %>%
  mutate(
    placement = str_extract(condition, "early|middle|late"),
    rank= str_extract(condition, "r[0-9]+"),
    prompt_type = ifelse(G == 1, "Unsafe", "Safe"),
    benchmark = dplyr::recode(source, wildguard = "WildGuard", beavertails = "BeaverTails")
  ) %>%
  group_by(basemodel, condition, rank, placement, benchmark, prompt_type) %>%
  summarise(
    n = n(),
    hct_accuracy = mean(hct_correct, na.rm = TRUE),
    srr = mean(R_safe, na.rm = TRUE),
    .groups = "drop"
  )

# ----------------------------------------------------------------
# (5) Line plots showing HCT and SRR colour coded by benchmark
# ----------------------------------------------------------------
benchmark_colours <- c("WildGuard" = "slategrey", "BeaverTails" = "#D2CDF6")
make_metric_plot <- function(plot_data, value_col, prompt_type_filter, y_label,
                             colors = benchmark_colours,
                             basemodel_order = NULL,
                             rank_order = c("r1", "r4", "r16", "r64"),
                             show_baseline = TRUE) {
  
  df <- plot_data %>% filter(prompt_type == prompt_type_filter)
  
  if (!is.null(basemodel_order)) {
    df <- df %>% mutate(basemodel = factor(basemodel, levels = basemodel_order))
  }
  
  main_data <- df %>%
    filter(placement %in% c("early", "middle", "late")) %>%
    mutate(
      placement = factor(str_to_title(placement), levels = c("Early", "Middle", "Late")),
      rank = factor(rank, levels = rank_order)
    )
  
  fullft_data <- df %>%
    filter(condition == "full") %>%
    select(-rank) %>%
    tidyr::crossing(rank = factor(rank_order, levels = rank_order))
  
  y_range_by_model <- main_data %>%
    group_by(basemodel) %>%
    summarise(y_min = min(.data[[value_col]], na.rm = TRUE),
              y_max = max(.data[[value_col]], na.rm = TRUE),
              .groups = "drop") %>%
    left_join(
      fullft_data %>%
        group_by(basemodel) %>%
        summarise(ft_min = min(.data[[value_col]], na.rm = TRUE),
                  ft_max = max(.data[[value_col]], na.rm = TRUE),
                  .groups = "drop"),
      by = "basemodel"
    ) %>%
    mutate(
      y_min = pmin(y_min, ft_min, na.rm = TRUE),
      y_max = pmax(y_max, ft_max, na.rm = TRUE),
      pad = (y_max - y_min) * 0.05,
      y_min = y_min - pad,
      y_max = y_max + pad
    ) %>%
    select(basemodel, y_min, y_max)
  
  baseline_data <- df %>%
    filter(condition == "baseline", show_baseline) %>%
    mutate(
      rank = factor(rank_order[1], levels = rank_order),
      vpos = match(benchmark, names(colors)),
      label = paste0("baseline: ", sprintf("%.1f%%", .data[[value_col]] * 100))
    ) %>%
    left_join(y_range_by_model, by = "basemodel") %>%
    mutate(fits_axis = .data[[value_col]] >= y_min & .data[[value_col]] <= y_max)
  
  baseline_line_data <- baseline_data %>%
    filter(fits_axis) %>%
    select(-rank) %>%
    tidyr::crossing(rank = factor(rank_order, levels = rank_order))
  
  baseline_text_data <- baseline_data %>% filter(!fits_axis)
  
  ggplot(main_data, aes(x = placement, y = .data[[value_col]],
                        color = benchmark, group = benchmark)) +
    geom_hline(data = fullft_data,
               aes(yintercept = .data[[value_col]], color = benchmark),
               linetype = "solid", linewidth = 0.6) +
    geom_hline(data = baseline_line_data,
               aes(yintercept = .data[[value_col]], color = benchmark),
               linetype = "dashed", linewidth = 0.6) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2) +
    geom_text(data = baseline_text_data,
              aes(x = Inf, y = -Inf, label = label, color = benchmark,
                  vjust = -0.3 - (vpos - 1) * 1.4),
              inherit.aes = FALSE, hjust = 1.05,
              fontface = "italic", size = 3.5, show.legend = FALSE) +
    facet_grid(basemodel ~ rank, scales = "free_y") +
    scale_color_manual(values = colors) +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 0.1),
      breaks = function(limits) {
        range_width <- limits[2] - limits[1]
        seq(limits[1] + range_width * 0.12, limits[2] - range_width * 0.12, length.out = 3)
      }
    ) +
    labs(x = "Adapter placement", y = y_label, color = "Benchmark") +
    theme_minimal(base_size = 13) +
    theme(
      legend.position = "right",
      strip.text = element_text(size = 13, color = "#52514e"),
      strip.text.y = element_text(angle = 0, color = "#52514e"),
      panel.spacing.y = unit(1.4, "lines"),
      axis.text.x = element_text(angle = 45, hjust = 1, color = "#52514e"),
      axis.text.y = element_text(color = "#52514e"),
      axis.title = element_text(color = "#52514e"),
      legend.text = element_text(color = "#52514e"),
      legend.title = element_text(color = "#52514e")
    )
}
# --------------------------------------------------------------------
srr_safe_plot <- make_metric_plot(
  plot_data = raw_summary, value_col = "srr", prompt_type_filter = "Safe",
  y_label = "SRR"
) + theme(legend.position = "none")
srr_unsafe_plot <- make_metric_plot(
  plot_data = raw_summary, value_col = "srr", prompt_type_filter = "Unsafe",
  y_label = "SRR*"
) + theme(legend.position = "none")
hct_safe_plot <- make_metric_plot(
  plot_data = raw_summary, value_col = "hct_accuracy", prompt_type_filter = "Safe",
  y_label = "HCT accuracy", show_baseline = FALSE
) + theme(legend.position = "none")
hct_unsafe_plot <- make_metric_plot(
  plot_data = raw_summary, value_col = "hct_accuracy", prompt_type_filter = "Unsafe",
  y_label = "HCT accuracy", show_baseline = FALSE
) + theme(legend.position = "none")
# --------------------------------------------------------------------
srr_safe_plot
srr_unsafe_plot
hct_safe_plot
hct_unsafe_plot
ggsave("srr-unsafe-prompts.png", plot = srr_unsafe_plot,
       path = "C:/Users/Ella/OneDrive/Documents/Ella/UniMsc/DISSERTATION/Final/Write-up",
       width = 7, height = 5, dpi = 300, bg = "white")
ggsave("srr-safe-prompts.png", plot = srr_safe_plot,
       path = "C:/Users/Ella/OneDrive/Documents/Ella/UniMsc/DISSERTATION/Final/Write-up",
       width = 7, height = 5, dpi = 300, bg = "white")
ggsave("hct-unsafe-prompts.png", plot = hct_unsafe_plot,
       path = "C:/Users/Ella/OneDrive/Documents/Ella/UniMsc/DISSERTATION/Final/Write-up",
       width = 7, height = 5, dpi = 300, bg = "white")
ggsave("hct-safe-prompts.png", plot = hct_safe_plot,
       path = "C:/Users/Ella/OneDrive/Documents/Ella/UniMsc/DISSERTATION/Final/Write-up",
       width = 7, height = 5, dpi = 300, bg = "white")

# ---------------------------------------------------------------------
# Make a plot to extract legend only
# ---------------------------------------------------------------------
legend_source <- make_metric_plot(
  plot_data = raw_summary, value_col = "srr", prompt_type_filter = "Safe",
  y_label = "SRR (safe prompts)"
) + theme(
  legend.position = "top",
  legend.text = element_text(size = 18, color = "#52514e"),
  legend.title = element_text(size = 20, color = "#52514e"),
  legend.key.size = unit(1, "cm")
)
legend_grob <- cowplot::get_legend(legend_source)
legend_only_plot <- cowplot::ggdraw(legend_grob)
ggsave("benchmark-legend.png", plot = legend_only_plot,
       path = "C:/Users/Ella/OneDrive/Documents/Ella/UniMsc/DISSERTATION/Final/Write-up",
       width = 8, height = 0.9, dpi = 300, bg = "white")

# ---------------------------------------------------------------------
# (6) Analysis of the effect of harm category on SRR and HCT accuracy
# ---------------------------------------------------------------------
category_data <- prepared %>%
  filter(G == 1) %>%
  mutate(lg_prompt_categories = str_split(lg_prompt_categories, ",")) %>%
  unnest_longer(lg_prompt_categories, keep_empty = TRUE) %>%
  mutate(
    lg_prompt_categories = str_trim(lg_prompt_categories),
    lg_prompt_categories = replace_na(lg_prompt_categories, "none")
  )

category_labels <- c(
  s1  = "Violent crimes",            s2  = "Non-violent crimes",
  s3  = "Sex-related crimes",        s4  = "Child sexual exploitation",
  s5  = "Defamation",                s6  = "Specialised advice",
  s7  = "Privacy",                   s8  = "Intellectual property",
  s9  = "Indiscriminate weapons",    s10 = "Hate",
  s11 = "Suicide & self-harm",       s12 = "Sexual content",
  s13 = "Electoral issues",          s14 = "Code interpreter abuse*"
)

srr_hct_by_category <- category_data %>%
  group_by(lg_prompt_categories) %>%
  summarise(
    n = n(),
    srr = mean(R_safe, na.rm = TRUE),
    hct_accuracy = mean(hct_correct, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(n)) %>%
  mutate(
    label = coalesce(category_labels[lg_prompt_categories], lg_prompt_categories),
    across(c(srr, hct_accuracy), ~ round(.x, 3))
  )
print(srr_hct_by_category, n = Inf, width = Inf)

# -----------------------------------------------------------------------
# Variance explained by harm category
# -----------------------------------------------------------------------
category_data <- category_data %>% mutate(lg_prompt_categories = factor(lg_prompt_categories))

pseudo_r2 <- function(model, null_model) 1 - (deviance(model) / deviance(null_model))

srr_category_model <- glmer(R_safe ~ lg_prompt_categories + (1 | index),
                            data = category_data, family = binomial(), nAGQ = 0)
srr_null_model <- glmer(R_safe ~ 1 + (1 | index),
                        data = category_data, family = binomial(), nAGQ = 0)

hct_category_data  <- category_data %>% filter(!is.na(H))
hct_category_model <- glmer(hct_correct ~ lg_prompt_categories + (1 | index),
                            data = hct_category_data, family = binomial(), nAGQ = 0)
hct_null_model <- glmer(hct_correct ~ 1 + (1 | index),
                        data = hct_category_data, family = binomial(), nAGQ = 0)

Anova(srr_category_model, type = "II")
Anova(hct_category_model, type = "II")
cat("SRR ~ harm category pseudo-R2:", round(pseudo_r2(srr_category_model, srr_null_model), 5), "\n")
cat("HCT accuracy ~ harm category pseudo-R2:", round(pseudo_r2(hct_category_model, hct_null_model), 5), "\n")

# ------------------------------------------------------------------------
# (7) Pairwise placement comparisons on BeaverTails prompts
# ------------------------------------------------------------------------
bt <- prepared %>%
  filter(source == "beavertails") %>%
  mutate(
    prompt_type = ifelse(G == 1, "Unsafe", "Safe"),
    placement = str_extract(condition, "early|middle|late"),
    rank = str_extract(condition, "r[0-9]+"),
    safe_response = as.logical(R_safe)
  )

ranks <- c("r1", "r4", "r16", "r64")

pairwise_results <- list()

for (pt in c("Unsafe", "Safe")) {
  for (bm in basemodel_order) {
    for (rk in ranks) {
      cond_data <- list(
        early  = bt %>% filter(prompt_type == pt, basemodel == bm, rank == rk, placement == "early"),
        middle = bt %>% filter(prompt_type == pt, basemodel == bm, rank == rk, placement == "middle"),
        late   = bt %>% filter(prompt_type == pt, basemodel == bm, rank == rk, placement == "late")
      )
      
      for (comp in c("early", "middle")) {
        merged <- inner_join(
          cond_data$late %>% select(index, safe_late = safe_response),
          cond_data[[comp]] %>% select(index, safe_comp = safe_response),
          by = "index"
        )
        
        tab  <- table(late = merged$safe_late, comp = merged$safe_comp)
        test <- mcnemar.test(tab, correct = TRUE)
        
        pairwise_results[[length(pairwise_results) + 1]] <- tibble(
          prompt_type = pt, basemodel = bm, rank = rk,
          comparison = paste0("late_vs_", comp),
          srr_late = mean(merged$safe_late) * 100,
          srr_comp = mean(merged$safe_comp) * 100,
          diff = mean(merged$safe_late) * 100 - mean(merged$safe_comp) * 100,
          p_raw = test$p.value
        )
      }
    }
  }
}

pairwise_df <- bind_rows(pairwise_results) %>%
  mutate(
    p_holm = p.adjust(p_raw, method = "holm"),
    sig = p_holm < 0.05,
    late_smaller_drop = diff > 0
  )

for (pt in c("Unsafe", "Safe")) {
  sub <- pairwise_df %>% filter(prompt_type == pt)
  cat(sprintf("[%s] %d/24 comparisons show a smaller reduction for late; %d/24 significant\n",
              pt, sum(sub$late_smaller_drop), sum(sub$sig & sub$late_smaller_drop)))
}

baseline_results <- list()

for (pt in c("Unsafe", "Safe")) {
  baseline_data <- bt %>% filter(prompt_type == pt, basemodel == "Mistral-7B", condition == "baseline")
  
  for (cond in c(paste0(ranks, "_late"), "full")) {
    cond_data <- bt %>% filter(prompt_type == pt, basemodel == "Mistral-7B", condition == cond)
    
    merged <- inner_join(
      baseline_data %>% select(index, safe_base = safe_response),
      cond_data %>% select(index, safe_cond = safe_response),
      by = "index"
    )
    
    tab  <- table(base = merged$safe_base, cond = merged$safe_cond)
    test <- mcnemar.test(tab, correct = TRUE)
    
    baseline_results[[length(baseline_results) + 1]] <- tibble(
      prompt_type = pt, condition = cond,
      srr_baseline = mean(merged$safe_base) * 100,
      srr_condition = mean(merged$safe_cond) * 100,
      drop = mean(merged$safe_base) * 100 - mean(merged$safe_cond) * 100,
      p_raw = test$p.value
    )
  }
}

baseline_df <- bind_rows(baseline_results) %>%
  mutate(
    p_holm = p.adjust(p_raw, method = "holm"),
    sig_drop = p_holm < 0.05 & drop > 0
  )

cat(sprintf("\nMistral-7B: %d/%d late/full conditions show a significant drop from baseline\n",
            sum(baseline_df$sig_drop), nrow(baseline_df)))

late_only <- baseline_df %>% filter(condition != "full")
cat(sprintf("Mistral-7B: late placement exceeded baseline in %d/%d conditions (numerically, drop < 0)\n",
            sum(late_only$drop < 0), nrow(late_only)))

