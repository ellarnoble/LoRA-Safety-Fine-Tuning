library(readxl)
library(dplyr)
library(stringr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(ggh4x)
# -----------------------------------------------------------------------
# (1) Load and clean
# -----------------------------------------------------------------------
raw <- finetune_safety_results_long
# Parse condition -> rank, placement
raw <- raw %>%
  dplyr::mutate(
    rank = dplyr::case_when(
      condition == "baseline" ~ "Baseline",
      condition == "full" ~ "full FT",
      TRUE ~ str_extract(condition, "(?<=r)\\d+")
    ),
    placement = dplyr::case_when(
      condition %in% c("baseline", "full") ~ "-",
      TRUE ~ str_extract(condition, "(?<=_)[a-z]+$")
    )
  )
# -----------------------------------------------------------------------
# (2) HCT accuracy (exclude classification failures, i.e. hct_label is NA)
# -----------------------------------------------------------------------
hct_acc <- raw %>%
  dplyr::filter(!is.na(hct_label)) %>%
  dplyr::group_by(basemodel, condition, rank, placement) %>%
  dplyr::summarise(
    hct_accuracy = mean(hct_label == lg_prompt_label) * 100,
    n_hct = n(),
    .groups = "drop"
  )
# -----------------------------------------------------------------------
# (3) SRR (harmful prompts only)
# -----------------------------------------------------------------------
srr <- raw %>%
  dplyr::filter(lg_prompt_label == "unsafe") %>%
  dplyr::group_by(basemodel, condition, rank, placement) %>%
  dplyr::summarise(
    srr = mean(lg_response_label == "safe") * 100,
    n_harmful = n(),
    .groups = "drop"
  )
safety <- hct_acc %>%
  dplyr::select(basemodel, condition, rank, placement, hct_accuracy, n_hct) %>%
  dplyr::full_join(srr %>% dplyr::select(basemodel, condition, srr, n_harmful),
                   by = c("basemodel", "condition"))
# -----------------------------------------------------------------------
# (4) Utility index from Table 10 (IFEval prompt/inst + MMLU, normalised as a
# % of baseline performance
# -----------------------------------------------------------------------
capability <- tribble(
  ~basemodel, ~condition,   ~prompt, ~inst,  ~mmlu,
  "Falcon7B", "baseline",   100.0,   100.0,  100.0,
  "Falcon7B", "r1_early",   99.7,    98.8,   98.7,
  "Falcon7B", "r1_middle",  95.3,    95.6,   98.8,
  "Falcon7B", "r1_late",    86.7,    90.0,   99.2,
  "Falcon7B", "r4_early",   100.3,   99.6,   99.3,
  "Falcon7B", "r4_middle",  93.7,    94.9,   98.8,
  "Falcon7B", "r4_late",    89.4,    90.7,   99.2,
  "Falcon7B", "r16_early",  101.7,   101.2,  99.5,
  "Falcon7B", "r16_middle", 94.4,    95.4,   98.8,
  "Falcon7B", "r16_late",   87.7,    90.7,   99.3,
  "Falcon7B", "r64_early",  99.7,    99.6,   99.4,
  "Falcon7B", "r64_middle", 93.0,    93.7,   98.9,
  "Falcon7B", "r64_late",   89.7,    91.7,   99.1,
  "Falcon7B", "full",       96.7,    97.7,   99.4,
  
  "Mistral7B", "baseline",   100.0,   100.0,  100.0,
  "Mistral7B", "r1_early",   89.1,    94.2,   96.7,
  "Mistral7B", "r1_middle",  94.8,    96.5,   99.3,
  "Mistral7B", "r1_late",    97.4,    99.8,   100.1,
  "Mistral7B", "r4_early",   95.7,    96.9,   97.5,
  "Mistral7B", "r4_middle",  100.4,   100.7,  100.6,
  "Mistral7B", "r4_late",    102.6,   100.0,  99.7,
  "Mistral7B", "r16_early",  104.8,   104.4,  97.7,
  "Mistral7B", "r16_middle", 95.7,    96.5,   99.5,
  "Mistral7B", "r16_late",   103.5,   102.2,  100.0,
  "Mistral7B", "r64_early",  103.0,   102.7,  98.0,
  "Mistral7B", "r64_middle", 98.7,    98.2,   100.3,
  "Mistral7B", "r64_late",   102.6,   101.3,  100.0,
  "Mistral7B", "full",       17.8,    22.8,   43.6,
  
  "Qwen7B", "baseline",   100.0,   100.0,  100.0,
  "Qwen7B", "r1_early",   88.8,    94.0,   100.1,
  "Qwen7B", "r1_middle",  79.8,    86.4,   100.0,
  "Qwen7B", "r1_late",    68.9,    78.7,   99.8,
  "Qwen7B", "r4_early",   87.0,    91.4,   99.7,
  "Qwen7B", "r4_middle",  86.3,    89.5,   99.8,
  "Qwen7B", "r4_late",    73.3,    79.1,   99.8,
  "Qwen7B", "r16_early",  88.5,    92.3,   99.7,
  "Qwen7B", "r16_middle", 80.7,    86.1,   99.8,
  "Qwen7B", "r16_late",   65.8,    75.8,   99.7,
  "Qwen7B", "r64_early",  87.9,    93.0,   99.9,
  "Qwen7B", "r64_middle", 83.9,    88.3,   99.7,
  "Qwen7B", "r64_late",   73.6,    79.2,   99.9,
  "Qwen7B", "full",       50.6,    62.1,   98.2
) %>%
  dplyr::mutate(
    ifeval_avg = (prompt + inst) / 2,
    utility = (ifeval_avg + mmlu) / 2
  )
# -----------------------------------------------------------------------
# 5. Combine (baseline AND full FT excluded)
# -----------------------------------------------------------------------
plot_df <- safety %>%
  dplyr::filter(!condition %in% c("baseline", "full")) %>%
  dplyr::inner_join(capability %>% dplyr::select(basemodel, condition, utility),
                    by = c("basemodel", "condition")) %>%
  tidyr::pivot_longer(cols = c(hct_accuracy, srr), names_to = "metric", values_to = "value") %>%
  dplyr::mutate(
    metric = dplyr::recode(metric, hct_accuracy = "HCT accuracy (%)", srr = "SRR (%)"),
    rank = factor(rank, levels = c("1", "4", "16", "64")),
    placement = factor(placement, levels = c("early", "middle", "late")),
    basemodel = dplyr::recode(basemodel, Falcon7B = "Falcon", Mistral7B = "Mistral", Qwen7B = "Qwen")
  )
# -----------------------------------------------------------------------
# 6. Pareto style plot
# -----------------------------------------------------------------------
is_pareto_optimal <- function(utility, value) {
  n <- length(utility)
  optimal <- rep(TRUE, n)
  for (i in seq_len(n)) {
    dominated_by_any <- any(
      utility >= utility[i] & value >= value[i] & (utility > utility[i] | value > value[i])
    )
    optimal[i] <- !dominated_by_any
  }
  optimal
}
plot_df <- plot_df %>%
  dplyr::group_by(basemodel, metric) %>%
  dplyr::mutate(pareto = is_pareto_optimal(utility, value)) %>%
  ungroup()
frontier_df <- plot_df %>%
  dplyr::filter(pareto) %>%
  dplyr::arrange(basemodel, metric, utility)
# -----------------------------------------------------------------------
# 7. Plots (split into HCT and SRR)
# -----------------------------------------------------------------------
rank_colors <- c(
  "1" = "#c7e9b4", "4" = "#D2CDF6", "16" = "darkseagreen4",
  "64" = "#253494"
)
placement_shapes <- c("early" = 16, "middle" = 15, "late" = 17)
# Hand-write the base model labels
basemodel_labels <- c(
  "Falcon" = "Falcon-7B",
  "Mistral" = "Mistral-7B",
  "Qwen" = "Qwen-7B"
)
# x-axis (utility index) is shared across both plots
UTILITY_XLIM <- c(85, 105)
make_utility_plot <- function(data, x_breaks, x_labels) {
  ggplot(data, aes(x = utility, y = value)) +
    geom_vline(xintercept = 100, linetype = "dotted", colour = "grey60") +
    geom_point(aes(colour = rank, shape = placement), size = 5, alpha = 0.9) +
    geom_point(data = dplyr::filter(data, pareto),
               shape = 1, size = 7, stroke = 1.3, colour = "black") +
    facet_wrap(~ basemodel, scales = "free_y", labeller = as_labeller(basemodel_labels)) +
    scale_colour_manual(values = rank_colors, name = "Rank") +
    scale_shape_manual(values = placement_shapes, name = "Placement") +
    scale_x_continuous(breaks = x_breaks, labels = x_labels) +
    coord_cartesian(xlim = UTILITY_XLIM) +
    theme_minimal(base_size = 13) +
    theme(
      legend.position = "right",
      strip.text = element_text(face = "bold", color = "#52514e"),
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.text  = element_text(color = "#52514e"),
      axis.title = element_text(color = "#52514e"),
      legend.text  = element_text(color = "#52514e"),
      legend.title = element_text(color = "#52514e"),
      legend.key.size = unit(1.1, "lines"),
      panel.grid.minor = element_blank(),
      aspect.ratio = 1
    ) +
    guides(
      colour = guide_legend(override.aes = list(size = 5, alpha = 1)),
      shape = guide_legend(override.aes = list(size = 5, alpha = 1, colour = "#52514e"))
    )
}
hct_df <- plot_df %>% filter(metric == "HCT accuracy (%)")
srr_df <- plot_df %>% filter(metric == "SRR (%)")
# Hand-write the x-axis breaks
UTILITY_X_BREAKS <- c(85, 90, 95, 100)
UTILITY_X_LABELS <- c("85", "90", "95", "100")
p_hct <- make_utility_plot(hct_df, UTILITY_X_BREAKS, UTILITY_X_LABELS)
p_srr <- make_utility_plot(srr_df, UTILITY_X_BREAKS, UTILITY_X_LABELS)
# Hand-write each plot's axis titles
p_hct <- p_hct + labs(x = "Utility index (%)", y = "HCT accuracy (%)*")
p_srr <- p_srr + labs(x = "Utility index (%)*", y = "SRR (%)")
# Per-base-model y-ranges
p_hct <- p_hct +
  ggh4x::facetted_pos_scales(
    y = list(
      basemodel == "Falcon" ~ scale_y_continuous(limits = c(83, 85.5), breaks = seq(83, 85, 1)),
      basemodel == "Mistral" ~ scale_y_continuous(limits = c(82, 85.5), breaks = seq(82, 85, 1)),
      basemodel == "Qwen" ~ scale_y_continuous(limits = c(80, 85.5), breaks = seq(80, 85, 1))
    )
  )
p_srr <- p_srr +
  ggh4x::facetted_pos_scales(
    y = list(
      basemodel == "Falcon" ~ scale_y_continuous(limits = c(80, 100), breaks = seq(80, 100, 5)),
      basemodel == "Mistral" ~ scale_y_continuous(limits = c(80, 100), breaks = seq(80, 100, 5)),
      basemodel == "Qwen" ~ scale_y_continuous(limits = c(80, 100), breaks = seq(80, 100, 5))
    )
  )
p_hct
p_srr
