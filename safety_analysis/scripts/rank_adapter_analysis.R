# ---------------------------------------------------------------------
# (1) Preliminaries
# ---------------------------------------------------------------------
library(dplyr)
library(readxl)
library(lme4)
library(car)
library(purrr)
library(tidyr)
library(tibble)
library(stringr)
library(performance)
library(nnet)

# Data preparation

basemodels <- c("Falcon7B", "Mistral7B", "Qwen7B")

rank_placement_data <- rank_placement_data %>%
  mutate(
    index              = as.factor(index),
    basemodel          = factor(basemodel, levels = basemodels),
    rank               = factor(rank, levels = c("r1", "r4", "r16", "r64")),
    placement          = factor(placement, levels = c("early", "middle", "late")),
    source             = factor(source),
    lg_prompt_label    = as.factor(lg_prompt_label),
    lg_response_label  = as.factor(lg_response_label),
    hct                = as.integer(round(as.numeric(hct))),
    prompt_type        = factor(str_to_title(as.character(lg_prompt_label)),
                                levels = c("Safe", "Unsafe")),
    R_safe             = as.integer(as.character(lg_response_label) == "safe"),
    hct_correct        = as.integer(hct_agrees_with_lg)
  )

unsafe_data <- rank_placement_data %>% filter(prompt_type == "Unsafe")



# Helper functions

add_stars <- function(p) {
  case_when(
    p < .001 ~ "***",
    p < .01 ~ "**",
    p < .05 ~ "*",
    TRUE ~ ""
  )
}

r2_marginal <- function(model) {
  r2_nakagawa(model)$R2_marginal
}

extract_type2_terms <- function(model, n_terms) {
  aov <- Anova(model, type = "II")
  tibble(
    term = rownames(aov),
    df = aov[["Df"]],
    chisq = aov[["Chisq"]],
    p_raw = aov[["Pr(>Chisq)"]]
  ) %>%
    mutate(
      p_holm = p.adjust(p_raw, method = "holm", n = n_terms),
      stars = add_stars(p_holm)
    )
}

# --------------------------------------------------------------
# (2a) Full model (15-term), unsafe prompts only
#   rank * placement * basemodel * source, for each outcome variable
# --------------------------------------------------------------
# Runtime Warning: requires substaintial RAM; uncomment to run

#full_srr_fit <- glmer(R_safe ~ rank * placement * basemodel * source
#                      + (1|index), 
#                      data = unsafe_data, 
#                      family = binomial,
#                      control = glmerControl(optimizer = "bobyqa", 
#                                             optCtrl = list(maxfun = 2e5)
#))


#full_hct_fit <- glmer(hct_correct ~ rank * placement * basemodel * source
#                      + (1|index), 
#                      data = unsafe_data, 
#                      family = binomial,
#                      control = glmerControl(optimizer = "bobyqa", 
#                                             optCtrl = list(maxfun = 2e5)
#                      ))

# --------------------------------------------------------------
# (2b) Full model (15-term), safe prompts only
#   rank * placement * basemodel * source, for each outcome variable
# --------------------------------------------------------------
# Runtime Warning: requires substaintial RAM; uncomment to run

#full_srr_fit_safe <- glmer(R_safe ~ rank * placement * basemodel * source
#                      + (1|index), 
#                      data = safe_data, 
#                      family = binomial,
#                      control = glmerControl(optimizer = "bobyqa", 
#                                             optCtrl = list(maxfun = 2e5)
#))
#full_hct_fit_safe <- glmer(hct_correct ~ rank * placement * basemodel * source
#                      + (1|index), 
#                      data = safe_data, 
#                      family = binomial,
#                      control = glmerControl(optimizer = "bobyqa", 
#                                             optCtrl = list(maxfun = 2e5)
#                      ))
# -------------------------------------------------------------
# (3a) Reduced model (11-term), rank:source family dropped as not sig in 
# full model 
# -------------------------------------------------------------

# SRR model 
srr_reduced_model <- glmer(
  R_safe ~ rank + placement + basemodel + source +
    rank:placement + rank:basemodel + placement:basemodel +
    placement:source + basemodel:source +
    rank:placement:basemodel + placement:basemodel:source +
    (1 | index),
  data = unsafe_data, family = binomial, nAGQ = 0
)

# HCT model
hct_data <- unsafe_data %>% filter(!is.na(hct_correct))

hct_reduced_model <- glmer(
  hct_correct ~ rank + placement + basemodel + source +
    rank:placement + rank:basemodel + placement:basemodel +
    placement:source + basemodel:source +
    rank:placement:basemodel + placement:basemodel:source +
    (1 | index),
  data = hct_data, family = binomial, nAGQ = 0
)

# Results table
srr_reduced_terms <- extract_type2_terms(srr_reduced_model, n_terms = 11) %>%
  mutate(metric = "SRR")
hct_reduced_terms <- extract_type2_terms(hct_reduced_model, n_terms = 11) %>%
  mutate(metric = "HCT")

rank_placement_reduced_results <- bind_rows(srr_reduced_terms, hct_reduced_terms)
reduced_model_r2 <- tibble(
  metric = c("SRR", "HCT"),
  r2     = c(r2_marginal(srr_reduced_model), r2_marginal(hct_reduced_model))
)

print(rank_placement_reduced_results, n= Inf, width = Inf)
print(reduced_model_r2)

# Individual contributions to r sqaured for each term

reduced_terms <- c(
  "rank", "placement", "basemodel", "source",
  "rank:placement", "rank:basemodel", "placement:basemodel",
  "placement:source", "basemodel:source",
  "rank:placement:basemodel", "placement:basemodel:source"
)

term_vars <- function(term) strsplit(term, ":")[[1]]

# terms to drop for a given target term = itself + any term whose variable
# set is a superset of the target's (i.e. every higher-order relative)
terms_to_drop <- function(target, all_terms) {
  target_vars <- term_vars(target)
  all_terms[sapply(all_terms, function(t) all(target_vars %in% term_vars(t)))]
}

fit_reduced_r2 <- function(outcome, data, drop_terms, all_terms) {
  keep_terms <- setdiff(all_terms, drop_terms)
  rhs <- paste(keep_terms, collapse = " + ")
  form <- as.formula(paste0(outcome, " ~ ", rhs, " + (1 | index)"))
  reduced <- glmer(form, data = data, family = binomial, nAGQ = 0)
  performance::r2_nakagawa(reduced)$R2_marginal
}

compute_term_r2_rankplacement <- function(full_model, outcome, data, all_terms) {
  r2_full <- performance::r2_nakagawa(full_model)$R2_marginal
  map_dfr(all_terms, function(term) {
    cat("  Dropping:", term, "\n")
    drop <- terms_to_drop(term, all_terms)
    r2_reduced <- fit_reduced_r2(outcome, data, drop, all_terms)
    tibble(term = term, r2_full = r2_full, r2_reduced = r2_reduced,
           delta_r2 = r2_full - r2_reduced,
           delta_r2_pct = round((r2_full - r2_reduced) * 100, 2))
  })
}

cat("=== SRR: computing term-level R^2 (unsafe prompts, nAGQ=0) ===\n")
srr_term_r2 <- compute_term_r2_rankplacement(
  srr_reduced_model, "R_safe", unsafe_data, reduced_terms
) %>% mutate(metric = "SRR")

cat("=== HCT: computing term-level R^2 (unsafe prompts, nAGQ=0) ===\n")
hct_term_r2 <- compute_term_r2_rankplacement(
  hct_reduced_model, "hct_correct", hct_data, reduced_terms
) %>% mutate(metric = "HCT")

rank_placement_term_r2 <- bind_rows(srr_term_r2, hct_term_r2)
print(rank_placement_term_r2, n = Inf, width = Inf)

# Check convergence on fitted models

check_convergence <- function(model) {
  msgs <- model@optinfo$conv$lme4$messages
  tibble(
    converged = length(msgs) == 0,
    messages  = if (length(msgs) == 0) NA_character_ else paste(msgs, collapse = "; "),
    singular  = isSingular(model)
  )
}

headline_convergence <- bind_rows(
  bind_cols(model = "srr_reduced_model", check_convergence(srr_reduced_model)),
  bind_cols(model = "hct_reduced_model", check_convergence(hct_reduced_model))
)
print(headline_convergence, n = Inf, width = Inf)

safe_data <- rank_placement_data %>% filter(prompt_type == "Safe")

# -------------------------------------------------------------
# (3b) Reduced model (11-term), safe prompts, rank:source family dropped
# as not sig in full model
# -------------------------------------------------------------
# SRR model 
srr_reduced_model_safe <- glmer(
  R_safe ~ rank + placement + basemodel + source +
    rank:placement + rank:basemodel + placement:basemodel +
    placement:source + basemodel:source +
    rank:placement:basemodel + placement:basemodel:source +
    (1 | index),
  data = safe_data, family = binomial, nAGQ = 0
)
# HCT model
hct_data_safe <- safe_data %>% filter(!is.na(hct_correct))
hct_reduced_model_safe <- glmer(
  hct_correct ~ rank + placement + basemodel + source +
    rank:placement + rank:basemodel + placement:basemodel +
    placement:source + basemodel:source +
    rank:placement:basemodel + placement:basemodel:source +
    (1 | index),
  data = hct_data_safe, family = binomial, nAGQ = 0
)
# Results table
srr_reduced_terms_safe <- extract_type2_terms(srr_reduced_model_safe, n_terms = 11) %>%
  mutate(metric = "SRR", prompt_type = "Safe")
hct_reduced_terms_safe <- extract_type2_terms(hct_reduced_model_safe, n_terms = 11) %>%
  mutate(metric = "HCT", prompt_type = "Safe")

# Add prompt_type to the existing unsafe-prompt results for consistency
rank_placement_reduced_results <- rank_placement_reduced_results %>%
  mutate(prompt_type = "Unsafe") %>%
  bind_rows(srr_reduced_terms_safe, hct_reduced_terms_safe)

reduced_model_r2 <- bind_rows(
  reduced_model_r2 %>% mutate(prompt_type = "Unsafe"),
  tibble(
    metric      = c("SRR", "HCT"),
    r2          = c(r2_marginal(srr_reduced_model_safe), r2_marginal(hct_reduced_model_safe)),
    prompt_type = "Safe"
  )
)
print(rank_placement_reduced_results, n = Inf, width = Inf)
print(reduced_model_r2)

# Individual contributions to r squared for each term (safe prompts)
cat("=== SRR: computing term-level R^2 (safe prompts, nAGQ=0) ===\n")
srr_term_r2_safe <- compute_term_r2_rankplacement(
  srr_reduced_model_safe, "R_safe", safe_data, reduced_terms
) %>% mutate(metric = "SRR", prompt_type = "Safe")
cat("=== HCT: computing term-level R^2 (safe prompts, nAGQ=0) ===\n")
hct_term_r2_safe <- compute_term_r2_rankplacement(
  hct_reduced_model_safe, "hct_correct", hct_data_safe, reduced_terms
) %>% mutate(metric = "HCT", prompt_type = "Safe")

rank_placement_term_r2 <- rank_placement_term_r2 %>%
  mutate(prompt_type = "Unsafe") %>%
  bind_rows(srr_term_r2_safe, hct_term_r2_safe)
print(rank_placement_term_r2, n = Inf, width = Inf)

# Check convergence on the newly fitted safe-prompt models
headline_convergence <- bind_rows(
  headline_convergence,
  bind_cols(model = "srr_reduced_model_safe", check_convergence(srr_reduced_model_safe)),
  bind_cols(model = "hct_reduced_model_safe", check_convergence(hct_reduced_model_safe))
)
print(headline_convergence, n = Inf, width = Inf)


# ----------------------------------------------------------------
# (4) Failure-mode data prep (unsafe prompts only)
# ----------------------------------------------------------------

is_unsafe <- function(label) as.character(label) == "unsafe"

prepared <- rank_placement_data %>%
  mutate(
    G      = as.integer(is_unsafe(lg_prompt_label)),  
    H      = hct,                                    
    R_safe = as.integer(as.character(lg_response_label) == "safe")
  )

# Per-condition confusion matrix + failure-mode proportions (12 rank/placement
# conditions only — no full/baseline rows in this dataset)

summary_tbl <- prepared %>%
  filter(G == 1) %>%
  count(basemodel, rank, placement, H, R_safe) %>%
  group_by(basemodel, rank, placement) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

failure_props <- summary_tbl %>%
  mutate(
    failure_mode = case_when(
      H == 1 & R_safe == 1 ~ "working",
      H == 0 & R_safe == 0 ~ "detection_failure",
      H == 1 & R_safe == 0 ~ "compliance_failure",
      H == 0 & R_safe == 1 ~ "recovery_failure"
    )
  )

failure_data <- prepared %>%
  filter(G == 1) %>%
  mutate(
    failure_mode = case_when(
      H == 1 & R_safe == 1 ~ "working",
      H == 0 & R_safe == 0 ~ "detection_failure",
      H == 1 & R_safe == 0 ~ "compliance_failure",
      H == 0 & R_safe == 1 ~ "recovery_failure"
    ),
    failure_mode = factor(failure_mode,
                          levels = c("working", "detection_failure", "compliance_failure", "recovery_failure"))
  ) %>%
  filter(!is.na(failure_mode))

failure_data$failure_mode <- relevel(failure_data$failure_mode, ref = "working")


# -------------------------------------------------------------------
# (5) Failure-mode model fully crossed
# -------------------------------------------------------------------

full_model <- multinom(
  failure_mode ~ rank * placement * basemodel * source,
  data = failure_data, maxit = 1000, trace = FALSE
)

wald_results <- Anova(full_model, type = "II")
print(wald_results)


# -------------------------------------------------------------------
# (6) Reduced failure-mode model; due to sparsley populated cells
# -------------------------------------------------------------------

reduced_failure_mode <- multinom(
  failure_mode ~ rank + placement + basemodel + source +
    rank:placement + rank:basemodel + placement:basemodel +
    placement:source + basemodel:source +
    rank:placement:basemodel + placement:basemodel:source,
  data = failure_data, maxit = 1000, trace = FALSE
)

wald_results <- Anova(reduced_failure_mode, type = "II")
print(wald_results)

# -----------------------------------------------------------------------
# (7) Per-feature pseudo-R2 contributions
# -----------------------------------------------------------------------
null_model <- multinom(failure_mode ~ 1, data = failure_data, trace = FALSE)
pseudo_r2 <- function(model) {
  1 - (model$deviance / null_model$deviance)
}
full_r2 <- pseudo_r2(reduced_failure_mode)

no_rank_model <- multinom(
  failure_mode ~ placement + basemodel + source +
    placement:basemodel + placement:source + basemodel:source +
    placement:basemodel:source,
  data = failure_data, maxit = 1000, trace = FALSE
)
no_placement_model <- multinom(
  failure_mode ~ rank + basemodel + source +
    rank:basemodel + basemodel:source,
  data = failure_data, maxit = 1000, trace = FALSE
)
no_basemodel_model <- multinom(
  failure_mode ~ rank + placement + source +
    rank:placement + placement:source,
  data = failure_data, maxit = 1000, trace = FALSE
)
no_source_model <- multinom(
  failure_mode ~ rank + placement + basemodel +
    rank:placement + rank:basemodel + placement:basemodel +
    rank:placement:basemodel,
  data = failure_data, maxit = 1000, trace = FALSE
)

feature_contributions_tbl <- tibble(
  feature = c("rank", "placement", "basemodel", "source"),
  pseudo_r2_incl_interactions = c(
    full_r2 - pseudo_r2(no_rank_model),
    full_r2 - pseudo_r2(no_placement_model),
    full_r2 - pseudo_r2(no_basemodel_model),
    full_r2 - pseudo_r2(no_source_model)
  )
) %>%
  mutate(pct_of_full_r2 = 100 * pseudo_r2_incl_interactions / full_r2)

print(feature_contributions_tbl)
print(full_r2)

# -----------------------------------------------------------------------
# (8) Convergence check across all fitted models
# -----------------------------------------------------------------------
convergence_check <- tibble(
  model = c("reduced_model", "null_model", "no_rank_model", "no_placement_model",
            "no_basemodel_model", "no_source_model"),
  convergence = c(reduced_failure_mode$convergence, null_model$convergence,
                  no_rank_model$convergence, no_placement_model$convergence,
                  no_basemodel_model$convergence, no_source_model$convergence)
)
print(convergence_check) 

# -----------------------------------------------------------------------
# (9) Failure mode on safe prompts
# -----------------------------------------------------------------------
hct_fpr <- prepared %>%
  filter(G == 0) %>%
  group_by(basemodel, condition) %>%
  summarise(
    n_safe = n(),
    false_positives = sum(H == 1, na.rm = TRUE),
    fpr = mean(H == 1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(fpr = round(fpr, 3))

print(hct_fpr, n = Inf)

# ---------------------------------------------------------------------------
# (10) Pairwise post-hoc comparison tests across adapter placement conditions 
# ---------------------------------------------------------------------------
benchmark_data <- prepared %>%
  mutate(
    benchmark    = dplyr::recode(source, wildguard = "WildGuard", beavertails = "BeaverTails"),
    prompt_type  = ifelse(G == 1, "Unsafe", "Safe"),
    placement    = str_extract(condition, "early|middle|late"),
    rank         = str_extract(condition, "r[0-9]+"),
    safe_response = as.logical(R_safe)
  )

ranks      <- c("r1", "r4", "r16", "r64")
placements <- c("early", "middle", "late")

# -----------------------------------------------------------------------
# (10a) Late vs. early/middle placement, SRR — both benchmarks
# -----------------------------------------------------------------------
placement_pairwise_results <- list()
for (bench in c("WildGuard", "BeaverTails")) {
  for (pt in c("Unsafe", "Safe")) {
    for (bm in basemodel_order) {
      for (rk in ranks) {
        cond_data <- list(
          early  = benchmark_data %>% filter(benchmark == bench, prompt_type == pt, basemodel == bm, rank == rk, placement == "early"),
          middle = benchmark_data %>% filter(benchmark == bench, prompt_type == pt, basemodel == bm, rank == rk, placement == "middle"),
          late   = benchmark_data %>% filter(benchmark == bench, prompt_type == pt, basemodel == bm, rank == rk, placement == "late")
        )
        for (comp in c("early", "middle")) {
          merged <- inner_join(
            cond_data$late %>% select(index, safe_late = safe_response),
            cond_data[[comp]] %>% select(index, safe_comp = safe_response),
            by = "index"
          )
          tab  <- table(late = merged$safe_late, comp = merged$safe_comp)
          test <- mcnemar.test(tab, correct = TRUE)
          placement_pairwise_results[[length(placement_pairwise_results) + 1]] <- tibble(
            benchmark = bench, prompt_type = pt, basemodel = bm, rank = rk,
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
}
placement_pairwise_df <- bind_rows(placement_pairwise_results) %>%
  mutate(
    p_holm = p.adjust(p_raw, method = "holm"),
    sig = p_holm < 0.05,
    late_larger = diff > 0
  )
for (bench in c("WildGuard", "BeaverTails")) {
  for (pt in c("Unsafe", "Safe")) {
    sub <- placement_pairwise_df %>% filter(benchmark == bench, prompt_type == pt)
    cat(sprintf("[%s, %s] %d/24 comparisons favour late placement; %d/24 significant\n",
                bench, pt, sum(sub$late_larger), sum(sub$sig & sub$late_larger)))
  }
}

# -----------------------------------------------------------------------
# (10b) Placement conditions vs. baseline, SRR — all base models & benchmarks
# -----------------------------------------------------------------------
baseline_pairwise_results <- list()
for (bench in c("WildGuard", "BeaverTails")) {
  for (pt in c("Unsafe", "Safe")) {
    for (bm in basemodel_order) {
      baseline_data <- benchmark_data %>%
        filter(benchmark == bench, prompt_type == pt, basemodel == bm, condition == "baseline")
      for (pl in placements) {
        for (rk in ranks) {
          cond <- paste0(rk, "_", pl)
          cond_data <- benchmark_data %>%
            filter(benchmark == bench, prompt_type == pt, basemodel == bm, condition == cond)
          merged <- inner_join(
            baseline_data %>% select(index, safe_base = safe_response),
            cond_data %>% select(index, safe_cond = safe_response),
            by = "index"
          )
          tab  <- table(base = merged$safe_base, cond = merged$safe_cond)
          test <- mcnemar.test(tab, correct = TRUE)
          baseline_pairwise_results[[length(baseline_pairwise_results) + 1]] <- tibble(
            benchmark = bench, prompt_type = pt, basemodel = bm,
            placement = pl, rank = rk, condition = cond,
            srr_baseline = mean(merged$safe_base) * 100,
            srr_condition = mean(merged$safe_cond) * 100,
            diff = mean(merged$safe_cond) * 100 - mean(merged$safe_base) * 100,
            p_raw = test$p.value
          )
        }
      }
    }
  }
}
baseline_pairwise_df <- bind_rows(baseline_pairwise_results) %>%
  mutate(
    p_holm = p.adjust(p_raw, method = "holm"),
    sig_decrease = p_holm < 0.05 & diff < 0
  )
# e.g. Safe/early+middle significance by base model (paragraph 3):
baseline_pairwise_df %>%
  filter(prompt_type == "Safe", placement %in% c("early", "middle")) %>%
  group_by(basemodel) %>%
  summarise(n_sig_decrease = sum(sig_decrease), n_tested = n(), .groups = "drop") %>%
  print()

# -----------------------------------------------------------------------
# (10c) Middle vs. early/late placement, balanced (macro-averaged) HCT accuracy
# -----------------------------------------------------------------------
hct_benchmark_data <- benchmark_data %>%
  filter(!is.na(hct_correct)) %>%
  mutate(hct_ok = as.logical(hct_correct))

fisher_combine <- function(pvals) {
  stat <- -2 * sum(log(pvals))
  1 - pchisq(stat, df = 2 * length(pvals))
}

hct_placement_results <- list()
for (bm in basemodel_order) {
  for (rk in ranks) {
    cond_data <- list(
      early  = hct_benchmark_data %>% filter(basemodel == bm, rank == rk, placement == "early"),
      middle = hct_benchmark_data %>% filter(basemodel == bm, rank == rk, placement == "middle"),
      late   = hct_benchmark_data %>% filter(basemodel == bm, rank == rk, placement == "late")
    )
    for (comp in c("early", "late")) {
      p_by_cell  <- c()
      acc_middle <- c()
      acc_comp   <- c()
      for (bench in c("WildGuard", "BeaverTails")) {
        for (pt in c("Unsafe", "Safe")) {
          mid_cell  <- cond_data$middle %>% filter(benchmark == bench, prompt_type == pt)
          comp_cell <- cond_data[[comp]] %>% filter(benchmark == bench, prompt_type == pt)
          merged <- inner_join(
            mid_cell  %>% select(index, hct_mid  = hct_ok),
            comp_cell %>% select(index, hct_comp = hct_ok),
            by = "index"
          )
          tab  <- table(mid = merged$hct_mid, comp = merged$hct_comp)
          test <- mcnemar.test(tab, correct = TRUE)
          key <- paste(bench, pt)
          p_by_cell[key]  <- test$p.value
          acc_middle[key] <- mean(merged$hct_mid) * 100
          acc_comp[key]   <- mean(merged$hct_comp) * 100
        }
      }
      hct_placement_results[[length(hct_placement_results) + 1]] <- tibble(
        basemodel = bm, rank = rk, comparison = paste0("middle_vs_", comp),
        balanced_hct_middle = mean(acc_middle),
        balanced_hct_comp   = mean(acc_comp),
        diff  = mean(acc_middle) - mean(acc_comp),
        p_raw = fisher_combine(p_by_cell)
      )
    }
  }
}
hct_placement_df <- bind_rows(hct_placement_results) %>%
  mutate(
    p_holm = p.adjust(p_raw, method = "holm"),
    sig = p_holm < 0.05,
    middle_larger = diff > 0
  )
cat(sprintf("[Balanced HCT] %d/24 comparisons favour middle placement; %d/24 significant\n",
            sum(hct_placement_df$middle_larger), sum(hct_placement_df$sig & hct_placement_df$middle_larger)))                   
