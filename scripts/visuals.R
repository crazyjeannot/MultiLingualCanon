############################## Multilingual Canon ##############################
#------------------------------------------------------------------------------#

# --- Last edited: 2025-07-02 -------------------------------------------------#

#------------------------------------------------------------------------------#
## Required packages ----------------------------------------------------------#
#------------------------------------------------------------------------------#

library(ggplot2)
library(ggthemes)
library(dplyr)
library(readr)
library(patchwork)
library(scales)
library(tidyverse)
library(cowplot)
library(stringr)
library(gridExtra)
library(changepoint)
library(zoo)

#------------------------------------------------------------------------------#
## Overall relationship between canonisation and centrality -------------------#
#------------------------------------------------------------------------------#

# Scatterplots ----------------------------------------------------------------#

setwd("C:\\Users\\Brottrager\\Documents\\Projects\\MultilingualCanon\\networkMetrics_time_sensitive_3nn")

files <- dir(pattern = "network_metrics.+.csv")

for (file in files) {
  df <- read.csv(file)
  
  centrality_vars <- c("indegree", "pagerank", "betweenness", "closeness")
  available_vars <- centrality_vars[centrality_vars %in% names(df)]
  
  df <- df %>% drop_na(canonisation_score)
  
  plots <- lapply(available_vars, function(var) {
    ggplot(df, aes_string(x = var, y = "canonisation_score")) +
      geom_point(alpha = 0.6, colour = "grey20") +
      geom_smooth(method = "lm", se = FALSE, linetype = "dashed", colour = "firebrick") +
      scale_y_continuous(limits = c(0, 1)) +
      theme_par(base_size = 12) +
      labs(
        title = paste("Canonisation vs.", var),
        x = var,
        y = "Canonisation Score"
      ) +
      theme(plot.margin = unit(c(1, 5, 1, 1), 'pt'),
            legend.position = c(0.9, 0.855))
  })
  
  plot_title <- cowplot::ggdraw()
  
  grid <- cowplot::plot_grid(plotlist = plots, labels = "AUTO", ncol = 2)
  final_plot <- cowplot::plot_grid(plot_title, grid, ncol = 1, rel_heights = c(0.1, 1))
  
  out_name <- paste0(tools::file_path_sans_ext(file), "_centrality_scatter.png")
  ggsave(out_name, plot = final_plot, width = 10, height = 8)
}


# Summarised scatterplots -----------------------------------------------------#

df_all <- map_dfr(files, function(file) {
  lang <- gsub("^network_metrics_(.*?)\\.csv$", "\\1", file)
  read_csv(file) %>%
    mutate(language = lang)
})

# Normalise centralities per language
normalise <- function(x) {
  rng <- range(x, na.rm = TRUE)
  if (diff(rng) == 0) return(rep(0, length(x)))
  (x - rng[1]) / diff(rng)
}

df_norm <- df_all %>%
  group_by(language) %>%
  mutate(
    indegree = normalise(indegree),
    pagerank = normalise(pagerank),
    betweenness = normalise(betweenness),
    closeness = normalise(closeness)
  ) %>%
  ungroup()


df_long <- df_norm %>%
  select(title, indegree, pagerank, betweenness, closeness, canonisation_score, language) %>%
  pivot_longer(cols = c(indegree, pagerank, betweenness, closeness),
               names_to = "centrality_type",
               values_to = "centrality_value") %>%
  filter(!is.na(canonisation_score)) %>%
  mutate(canonisation_score = pmin(pmax(canonisation_score, 0), 1))  

df_long$centrality_type <- factor(df_long$centrality_type, levels = c("indegree", "pagerank",
                                                                      "betweenness", "closeness"))
df_long$language <- factor(df_long$language, levels = c("DK", "EN", "FR", "DE"))

# Plot: smoothed lines, black-and-white with linetype
ggplot(df_long, aes(x = centrality_value, y = canonisation_score, linetype = centrality_type)) +
  geom_smooth(method = "lm", se = FALSE, color = "black") +
  scale_linetype_manual(values = c("solid", "1342", "dotted", "twodash")) +  
  facet_wrap(~ language) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "Normalized centrality", y = "Canonisation score", linetype = "Centrality type") +
  theme_par() +
  theme(plot.margin = unit(c(1, 5, 1, 1), 'pt'),
        legend.position = "bottom")

ggsave("overallCorr.png", width = 8, height = 5, dpi = 360)

## Temporal trends ------------------------------------------------------------#

bin_size <- 10

df_long <- df_long %>%
  mutate(pub_year = as.numeric(str_extract(title, "^\\d{4}")))

# Bin years into intervals
all_long <- df_long %>%
  mutate(year_bin = floor(pub_year / bin_size) * bin_size)

# Calculate Spearman correlations per language, year_bin, and centrality
correlations <- all_long %>%
  group_by(language, year_bin, centrality_type) %>%
  summarise(
    rho = cor(canonisation_score, centrality_value, method = "spearman", use = "complete.obs"),
    n = n(),
    .groups = "drop"
  ) %>%
  filter(n >= 5)

write.csv(correlations, "temporalTrend//correlationsTemporal.csv")

ggplot(correlations, aes(x = year_bin, y = rho, linetype = centrality_type)) +
  geom_line(size = 1) +
  geom_point() +
  facet_wrap(~ language, ncol = 2, scales = "free_x") +
  labs(
    x = paste0("Publication Year (", bin_size, "-year bins)"),
    y = expression(rho ~ "(Spearman correlation)"),
    linetype = "Centrality Measure"
  ) +
  theme_par() +
  theme(plot.margin = unit(c(1, 5, 1, 1), 'pt'),
        legend.position = "bottom")

ggsave("temporalTrend//overallTemporalTrend.png", width = 12, height = 10, dpi = 360)

## Temporal trends (sliding window) -------------------------------------------#

bin_size <- 10   # sliding window size in years
step_size <- 1   # step size for sliding windows

df_long <- df_long %>%
  mutate(pub_year = as.numeric(str_extract(title, "^\\d{4}")))

results <- data.frame()

groups <- df_long %>% 
  distinct(language, centrality_type)

for (i in seq_len(nrow(groups))) {
  lang_i <- groups$language[i]
  cent_i <- groups$centrality_type[i]
  
  df_sub <- df_long %>%
    filter(language == lang_i, centrality_type == cent_i)
  
  min_year <- min(df_sub$pub_year, na.rm = TRUE)
  max_year <- max(df_sub$pub_year, na.rm = TRUE)
  
  for (start_year in seq(min_year, max_year - bin_size, by = step_size)) {
    end_year <- start_year + bin_size
    
    df_window <- df_sub %>%
      filter(pub_year >= start_year & pub_year < end_year)
    
    if (nrow(df_window) >= 5) {
      rho <- cor(df_window$canonisation_score, df_window$centrality_value, method = "spearman", use = "complete.obs")
      results <- rbind(results, data.frame(
        language = lang_i,
        centrality_type = cent_i,
        start = start_year,
        end = end_year,
        midpoint = start_year + bin_size / 2,
        rho = rho,
        n = nrow(df_window)
      ))
    }
  }
}

write.csv(results, "temporalTrend/correlationsTemporal_sliding.csv", row.names = FALSE)

ggplot(results, aes(x = midpoint, y = rho, linetype = centrality_type, color = centrality_type)) +
  geom_line(size = 1) +
  geom_point(aes(size = n), alpha = 0.6) +
  facet_wrap(~ language, ncol = 2, scales = "free_x") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "Sliding Window Spearman Correlations of Canonisation and Centrality",
    x = paste0("Publication Year (", bin_size, "-year sliding windows, step ", step_size, " years)"),
    y = expression(rho ~ "(Spearman correlation)"),
    linetype = "Centrality Measure",
    color = "Centrality Measure",
    size = "Texts in window"
  ) +
  theme_minimal()

ggsave("temporalTrend/overallTemporalTrend_sliding.png", width = 12, height = 10, dpi = 360)

## Change point detection -----------------------------------------------------#

change_points <- data.frame()

groups <- results %>%
  distinct(language, centrality_type)

for (i in seq_len(nrow(groups))) {
  lang_i <- groups$language[i]
  cent_i <- groups$centrality_type[i]
  
  df_sub <- results %>%
    filter(language == lang_i, centrality_type == cent_i) %>%
    arrange(midpoint)
  
  rho_vec <- df_sub$rho
  rho_smooth <- zoo::rollmean(rho_vec, k = 5, fill = NA, align = "center")
  rho_smooth <- na.omit(rho_smooth)  
  
  if(length(rho_smooth) > 1) {  
    cpt <- cpt.meanvar(rho_smooth, method = "BinSeg")
    cps <- cpts(cpt)
    
    # Map indices to year midpoints
    # Offset cps by floor(k/2) due to rollmean alignment center
    offset <- floor(5 / 2)
    cps_corrected <- cps + offset
    cps_corrected <- cps_corrected[cps_corrected <= nrow(df_sub)]
    
    cp_years <- df_sub$midpoint[cps_corrected]
    
    if (length(cp_years) > 0) {
      cp_df <- data.frame(
        language = lang_i,
        centrality_type = cent_i,
        change_point_year = cp_years
      )
      change_points <- rbind(change_points, cp_df)
    }
  }
}

print(change_points)
write.csv(change_points, "temporalTrend/change_points.csv")

ggplot(results, aes(x = midpoint, y = rho, color = centrality_type, linetype = centrality_type)) +
  geom_line(size = 1) +
  geom_point(aes(size = n), alpha = 0.6) +
  facet_wrap(~ language, ncol = 2, scales = "free_x") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_vline(data = change_points, aes(xintercept = change_point_year), color = "red", linetype = "dotted") +
  labs(
    title = "Sliding Window Spearman Correlations with Change Points",
    x = paste0("Publication Year (", bin_size, "-year sliding windows, step ", step_size, " years)"),
    y = expression(rho ~ "(Spearman correlation)"),
    color = "Centrality Measure",
    linetype = "Centrality Measure",
    size = "Texts in window"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

change_points <- change_points %>%
  mutate(
    label_y = case_when(
      centrality_type == "indegree" ~ 0.4,
      centrality_type == "pagerank" ~ 0.5,
      centrality_type == "betweenness" ~ 0.6,
      centrality_type == "closeness" ~ 0.7,
      TRUE ~ 1.05
    )
  )

ggplot(results, aes(x = midpoint)) +
  geom_point(aes(y = rho, size = n, color = centrality_type), alpha = 0.7) +
  facet_wrap(~ language, ncol = 2, scales = "free_x") +
  geom_vline(data = change_points, aes(xintercept = change_point_year),
             color = "black", linetype = "dotted") +
  scale_color_manual(values = c(
    "indegree" = "gray8", 
    "pagerank" = "darkred", 
    "betweenness" = "#FF7F50",
    "closeness" = "steelblue"
  )) +
  #geom_text(data = change_points, aes(x = change_point_year, 
  #          y = label_y, label = change_point_year), 
  #          angle = 90, vjust = 0, hjust = 0, color = "black", size = 3) +  # higher position, aligned nicely
  labs(
    #title = "Change Points Highlighted on Correlation Timeline",
    x = paste0("Publication Year (", bin_size, "-year windows)"),
    y = expression(rho ~ "(Spearman correlation)"),
    color = "Centrality Measure",
    shape = "Centrality Measure",
    size = "Texts in window"
  ) +
  theme_par() +
  theme(legend.position = "bottom")

ggsave("temporalTrend/overallTemporalTrend_changePoints.png", width = 12, height = 10, dpi = 360)

## Outliers -------------------------------------------------------------------#

output_dir <- "outlier_texts"
dir.create(output_dir, showWarnings = FALSE)

df_outliers <- df_long %>%
  group_by(language, centrality_type) %>%
  mutate(
    residual = centrality_value - canonisation_score,
    z_residual = scale(residual)[,1],
    outlier = abs(z_residual) > 2,
    absolute_residual = abs(residual),
    outlier_class = case_when(
      !outlier ~ "Normal",
      residual > 0 ~ "Structurally Central - Culturally Peripheral",
      residual < 0 ~ "Culturally Central - Structurally Peripheral"
    ),
    
    iqr_resid = IQR(residual, na.rm = TRUE),
    close_alignment = abs(residual) < 0.25 * iqr_resid,
    #canonised_aligned = close_alignment & canonisation_score > 0.25
  ) %>%
  ungroup()

df_outliers <- subset(df_outliers, close_alignment == TRUE & centrality_type == "indegree")
write.csv(df_outliers, "outlier_texts//outliers_only.csv")

df_danish <- df_outliers %>% filter(language == "DK")
p1 <- ggplot(df_danish, aes(x = centrality_value, y = canonisation_score, color = outlier_class)) +
  geom_point(alpha = 0.8) +
  facet_wrap(~centrality_type, scales = "free_x") +
  scale_color_manual(values = c(
    "Normal" = "gray80",
    "Structurally Central - Culturally Peripheral" = "steelblue", 
    "Culturally Central - Structurally Peripheral" = "#FF7F50"
  )) +
  labs(
    title = "DK",
    x = "Centrality Value",
    y = "Canonisation Score",
    color = "Outlier Type"
  ) +
  theme_par() +
  theme(
    plot.margin = unit(c(1, 5, 1, 1), "pt"),
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16),
  )


df_english <- df_outliers %>% filter(language == "EN")
p2 <- ggplot(df_english, aes(x = centrality_value, y = canonisation_score, color = outlier_class)) +
  geom_point(alpha = 0.8) +
  facet_wrap(~centrality_type, scales = "free_x") +
  scale_color_manual(values = c(
    "Normal" = "gray80", 
    "Structurally Central - Culturally Peripheral" = "steelblue", 
    "Culturally Central - Structurally Peripheral" = "#FF7F50"
  )) +
  labs(
    title = "EN",
    x = "Centrality Value",
    y = "Canonisation Score"
  ) +
  theme_par() +
  theme(
    plot.margin = unit(c(1, 5, 1, 1), "pt"),
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16),
  )


df_french <- df_outliers %>% filter(language == "FR")
p3 <- ggplot(df_french, aes(x = centrality_value, y = canonisation_score, color = outlier_class)) +
  geom_point(alpha = 0.8) +
  facet_wrap(~centrality_type, scales = "free_x") +
  scale_color_manual(values = c(
    "Normal" = "gray80", 
    "Structurally Central - Culturally Peripheral" = "steelblue", 
    "Culturally Central - Structurally Peripheral" = "#FF7F50"
  )) +
  labs(
    title = "FR",
    x = "Centrality Value",
    y = "Canonisation Score"
  ) +
  theme_par() +
  theme(
    plot.margin = unit(c(1, 5, 1, 1), "pt"),
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16),
  )


df_german <- df_outliers %>% filter(language == "DE")
p4 <- ggplot(df_german, aes(x = centrality_value, y = canonisation_score, color = outlier_class)) +
  geom_point(alpha = 0.8) +
  facet_wrap(~centrality_type, scales = "free_x") +
  scale_color_manual(values = c(
    "Normal" = "gray80", 
    "Structurally Central - Culturally Peripheral" = "steelblue", 
    "Culturally Central - Structurally Peripheral" = "#FF7F50"
  )) +
  labs(
    title = "DE",
    x = "Centrality Value",
    y = "Canonisation Score"
  ) +
  theme_par() +
  theme(
    plot.margin = unit(c(1, 5, 1, 1), "pt"),
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16),
  )

prow <- plot_grid(
  p1 + theme(legend.position="none"),
  p2 + theme(legend.position="none"),
  p3 + theme(legend.position="none"),
  p4 + theme(legend.position="none"))

legend <- get_legend(
  p1 + theme(legend.position = "bottom")
)

plot_grid(prow, legend, ncol = 1, rel_heights = c(1, .1))

ggsave("outlier_texts//allOutliers.png", width = 12, height = 10, dpi = 360)

outliers_list <- split(df_outliers, df_outliers$language)

for (lang in names(outliers_list)) {
  df_lang <- outliers_list[[lang]]
  filename <- file.path(output_dir, paste0(lang, "_outliers.csv"))
  write.csv(
    df_lang %>% select(title, centrality_type, centrality_value, canonisation_score, outlier_class, residual, z_residual),
    file = filename,
    row.names = FALSE
  )
}

for (lang in names(outliers_list)) {
  df_lang <- outliers_list[[lang]]
  filename <- file.path(output_dir, paste0(lang, "_outliersTexts.csv"))
  write.csv(
    df_lang %>%
      filter(outlier) %>%
      select(title, centrality_type, centrality_value, canonisation_score, outlier_class, residual, z_residual),
    file = filename,
    row.names = FALSE
  )
}


outlier_data <- df_outliers %>%
  filter(outlier_class != "Normal") %>%
  distinct(title, .keep_all = TRUE)

write.csv(outlier_data, "outlier_texts//outlier_data_ALL.csv")

outlier_summary <- df_outliers %>%
  filter(outlier_class != "Normal") %>%
  mutate(pub_decade = factor(floor(pub_year / 10) * 10)) %>%
  group_by(language, pub_decade, outlier_class) %>%
  summarise(n = n(), .groups = "drop")

write.csv(outlier_summary, "outlier_texts//outlier_data_summary.csv")

ggplot(outlier_data, aes(x = pub_year, fill = outlier_class)) +
  geom_histogram(binwidth = 10, position = "dodge", color = "black") +  # 10-year bins
  facet_wrap(~ language, scales = "free_y") +
  labs(
    x = "Publication Year",
    y = "Count",
    fill = "Outlier Class"
  ) +
  theme_par() +
  scale_fill_manual(values = c(
    "Structurally Central - Culturally Peripheral" = "steelblue", 
    "Culturally Central - Structurally Peripheral" = "#FF7F50"
  )) +
  theme(
    plot.margin = unit(c(1, 5, 1, 1), "pt"),
    legend.position = "bottom"
  )

ggsave("outlier_texts//allOutliersDist.png", width = 10, height = 6, dpi = 360)

#------------------------------------------------------------------------------#
## Relationship between canonisation and centrality in clusters ---------------#
#------------------------------------------------------------------------------#

# rho per cluster -------------------------------------------------------------#

setwd("C:\\Users\\Brottrager\\Documents\\Projects\\MultilingualCanon\\clusterCorrelationResults_time_sensitive_3nn")

files <- dir(pattern = "louvain_cluster.+.csv")

all_data <- do.call(rbind, lapply(files, function(file) {
  df <- read.csv(file)
  
  df_long <- df %>%
    pivot_longer(
      cols = starts_with("spearman_"),
      names_to = "metric",
      names_prefix = "spearman_",
      values_to = "correlation"
    ) %>%
    left_join(
      df %>%
        pivot_longer(
          cols = starts_with("p_"),
          names_to = "metric",
          names_prefix = "p_",
          values_to = "p_value"
        ),
      by = c("language", "cluster_id", "metric")
    ) %>%
    mutate(file = file) 
}))

y_min <- floor(min(all_data$correlation, na.rm = TRUE) * 10) / 10
y_max <- ceiling(max(all_data$correlation, na.rm = TRUE) * 10) / 10

for (file in files) {
  df <- read.csv(file)
  
  df_long <- df %>%
    pivot_longer(
      cols = starts_with("spearman_"),
      names_to = "metric",
      names_prefix = "spearman_",
      values_to = "correlation"
    ) %>%
    left_join(
      df %>%
        pivot_longer(
          cols = starts_with("p_"),
          names_to = "metric",
          names_prefix = "p_",
          values_to = "p_value"
        ),
      by = c("language", "cluster_id", "metric")
    )
  
  df_long$cluster_id <- factor(df_long$cluster_id)
  
  plot_metric <- function(metric_name) {
    df_plot <- df_long %>% filter(metric == metric_name)
    
    ggplot(df_plot, aes(x = cluster_id, y = correlation, fill = p_value < 0.05)) +
      geom_col(show.legend = FALSE) +
      geom_hline(yintercept = 0, linetype = "dashed") +
      scale_y_continuous(limits = c(y_min, y_max), breaks = seq(y_min, y_max, 0.1)) +
      scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "grey70")) +
      labs(
        title = paste("Spearman Correlation (Canonisation ~", metric_name, ")"),
        x = "Louvain Cluster",
        y = "Spearman ρ"
      ) +
      theme_par() +
      theme(plot.margin = unit(c(1, 5, 1, 1), 'pt'),
            legend.position = c(0.9, 0.855))
  }
  
  for (m in unique(df_long$metric)) {
    p <- plot_metric(m)
    out_file <- gsub(".csv", paste0("_cluster_correlations_", m, ".png"), file)
    ggsave(out_file, plot = p, width = 8, height = 5, dpi = 360)
  }
}

# Centrality grid -------------------------------------------------------------#

for (file in files) {
  df <- read.csv(file)
  
  df_long <- df %>%
    pivot_longer(
      cols = starts_with("spearman_"),
      names_to = "metric",
      names_prefix = "spearman_",
      values_to = "correlation"
    ) %>%
    left_join(
      df %>%
        pivot_longer(
          cols = starts_with("p_"),
          names_to = "metric",
          names_prefix = "p_",
          values_to = "p_value"
        ),
      by = c("language", "cluster_id", "metric")
    )
  
  df_long$cluster_id <- factor(df_long$cluster_id)
  
  plot_metric <- function(metric_name) {
    df_plot <- df_long %>% filter(metric == metric_name)
    
    ggplot(df_plot, aes(x = cluster_id, y = correlation, fill = p_value < 0.05)) +
      geom_col(show.legend = FALSE) +
      geom_hline(yintercept = 0, linetype = "dashed") +
      scale_y_continuous(limits = c(y_min, y_max), breaks = seq(y_min, y_max, 0.1)) +
      scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "grey70")) +
      labs(
        title = paste("ρ ~", metric_name),
        x = "Cluster",
        y = "Spearman ρ"
      ) +
      theme_par() +
      theme(plot.margin = unit(c(1, 5, 1, 1), 'pt'))
  }
  
  p1 <- plot_metric("indegree")
  p2 <- plot_metric("pagerank")
  p3 <- plot_metric("betweenness")
  p4 <- plot_metric("closeness")
  
  grid_plot <- (p1 | p2) / (p3 | p4) +
    plot_annotation(title = paste0("Centrality–Canonisation Correlation\n", file))
  
  out_file <- gsub(".csv", "_centrality_grid.png", file)
  ggsave(out_file, plot = grid_plot, width = 12, height = 10, dpi = 360)
}

# Centrality grid: Summary ----------------------------------------------------#

grey_shades <- c(
  "indegree"   = "#4D4D4D",  
  "pagerank"   = "#999999",  
  "betweenness"= "#BEBEBE",  
  "closeness"  = "#E6E6E6"  
)

for (file in files) {
  df <- read.csv(file)
  
  df_long <- df %>%
    pivot_longer(
      cols = starts_with("spearman_"),
      names_to = "metric",
      names_prefix = "spearman_",
      values_to = "correlation"
    ) %>%
    left_join(
      df %>%
        pivot_longer(
          cols = starts_with("p_"),
          names_to = "metric",
          names_prefix = "p_",
          values_to = "p_value"
        ),
      by = c("language", "cluster_id", "metric")
    )
  
  
  lang <- unique(df_long$language)
  
  df_long$cluster_id <- factor(df_long$cluster_id, 
                               levels = sort(unique(df_long$cluster_id)))
  df_long$metric <- factor(df_long$metric, levels = c("indegree", "pagerank",
                                                      "betweenness", "closeness"))

  p <- ggplot(df_long, aes(x = cluster_id, y = correlation, fill = metric)) +
    geom_tile(
      data = df_long %>% filter(p_value < 0.05),
      aes(x = cluster_id, y = 0, height = 0.9),
      width = 0.8,
      fill = "steelblue",
      alpha = 0.3,
      inherit.aes = FALSE
    ) +
    geom_col(position = position_dodge(width = 0.8), color = "black") +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_x_discrete(drop = FALSE) +
    scale_fill_manual(values = grey_shades, name = "Centrality Measure") +
    labs(
      title = lang,
      x = "Cluster",
      y = expression(rho)
    ) +
    scale_y_continuous(
      limits = c(-0.5, 0.5),
      breaks = seq(-0.5, 0.5, 0.1)
    ) +
    theme_par() +
    theme(
      plot.margin = unit(c(1, 5, 1, 1), "pt"),
      legend.position = "bottom",
      legend.text = element_text(size = 20),
      legend.title = element_text(size = 20),
    )
  
  assign(paste0("plot_", lang), p)
}

prow <- plot_grid(
  plot_DK + theme(legend.position="none"),
  plot_EN + theme(legend.position="none"),
  plot_FR + theme(legend.position="none"),
  plot_DE + theme(legend.position="none"))

legend <- get_legend(
  plot_DK + theme(legend.position = "bottom")
)

plot_grid(prow, legend, ncol = 1, rel_heights = c(1, 0.1, 1, 0.1, 1, 0.1, 1))

ggsave("combined_centrality_correlation_greyscale.png",
       width = 14, height = 10, dpi = 360)

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
