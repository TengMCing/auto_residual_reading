library(tensorflow)
library(keras)
library(tidyverse)
library(yardstick)
library(visage)
library(glue)

mod_32 <- load_model_tf("keras_tuner/best_models/experiment_factor/residual_plots/32")
mod_64 <- load_model_tf("keras_tuner/best_models/experiment_factor/residual_plots/64")

test_32 <- flow_images_from_directory("data/experiment_factor/residual_plots/32/mixed/test",
                                      target_size = c(32L, 32L),
                                      shuffle = FALSE)
test_64 <- flow_images_from_directory("data/experiment_factor/residual_plots/64/mixed/test",
                                      target_size = c(64L, 64L),
                                      shuffle = FALSE)


# get predictions ---------------------------------------------------------

pred_32 <- mod_32$predict(test_32) %>%
  as.data.frame() %>%
  mutate(truth = test_32$classes + 1) %>%
  mutate(truth = names(test_32$class_indices)[truth]) %>%
  mutate(filenames = test_32$filenames) %>%
  mutate(plot_uid = gsub(".*/(.*).png", "\\1", filenames)) %>%
  mutate(plot_uid = as.integer(plot_uid))

pred_32 <- pred_32 %>%
  mutate(pred = V1 > 0.5) %>%
  mutate(pred = ifelse(pred, "not_null", "null")) %>%
  mutate(pred = factor(pred)) %>%
  mutate(truth = factor(truth))

pred_32 %>% bal_accuracy(estimate = pred, truth = truth)

pred_64 <- mod_64$predict(test_64) %>%
  as.data.frame() %>%
  mutate(truth = test_64$classes + 1) %>%
  mutate(truth = names(test_64$class_indices)[truth]) %>%
  mutate(filenames = test_64$filenames) %>%
  mutate(plot_uid = gsub(".*/(.*).png", "\\1", filenames)) %>%
  mutate(plot_uid = as.integer(plot_uid))

pred_64 <- pred_64 %>%
  mutate(pred = V1 > 0.5) %>%
  mutate(pred = ifelse(pred, "not_null", "null")) %>%
  mutate(pred = factor(pred)) %>%
  mutate(truth = factor(truth))

pred_64 %>% bal_accuracy(estimate = pred, truth = truth)


# get meta ----------------------------------------------------------------

meta <- readRDS("data/experiment_factor/residual_plots/meta.rds")

es_ref <- vi_survey %>%
  group_by(unique_lineup_id) %>%
  summarise(across(everything(), first)) %>%
  select(shape, e_sigma, a, b, n, x_dist, effect_size) %>%
  group_by(shape, e_sigma, a, b, n, x_dist) %>%
  summarise(effect_size = first(effect_size))

pred_32 <- pred_32 %>%
  left_join(meta) %>%
  left_join(es_ref) %>%
  mutate(type = ifelse(!is.na(shape), "polynomial", "heteroskedasticity"))

pred_64 <- pred_64 %>%
  left_join(meta) %>%
  left_join(es_ref) %>%
  mutate(type = ifelse(!is.na(shape), "polynomial", "heteroskedasticity"))

# roc ---------------------------------------------------------------------

roc_auc(pred_32, estimate = V1, truth = truth)
roc_auc(pred_64, estimate = V1, truth = truth)

bind_rows(mutate(pred_32, res = 32), 
          mutate(pred_64, res = 64)) %>%
  group_by(res) %>%
  roc_curve(estimate = V1, truth = truth) %>% 
  autoplot()



# check model -------------------------------------------------------------

check_model <- list()

# Distribution of predicted probability on the test set
check_model$p_hat$distribution <- function(dat, res = 32) {
  dat %>%
    ggplot() +
    geom_histogram(aes(V1)) +
    ggtitle(quote("Distribution of " ~ hat(P)(y == not_null) ~ "on the test set"), 
            subtitle = glue("Input size {res} * {res}")) +
    xlab(quote(hat(P)(y == not_null))) +
    facet_grid(type ~ truth, labeller = label_both) +
    theme_light()
}

# Boxplot of predicted probability conditional on factors
check_model$p_hat$null$x_dist$distribution <- function(dat, res = 32) {
  dat %>%
    filter(truth == "null") %>%
    ggplot() +
    geom_boxplot(aes(factor(x_dist), V1)) +
    ylab(quote(hat(P)(y == not_null))) +
    xlab("Distribution of fitted values") +
    scale_x_discrete(labels = c("discrete uniform", "lognormal", "normal", "uniform")) +
    ggtitle(quote("Distribution of " ~ hat(P)(y == not_null ~"|"~ truth == null) ~ "on the test set"), 
            subtitle = glue("Input size {res} * {res}")) +
    theme_light()
}

check_model$p_hat$null$n$distribution <- function(dat, res = 32) {
  dat %>%
    filter(truth == "null") %>%
    ggplot() +
    geom_boxplot(aes(factor(n), V1)) +
    xlab("Number of observations") +
    ggtitle(quote("Distribution of " ~ hat(P)(y == not_null ~"|"~ truth == null) ~ "on the test set"), 
            subtitle = glue("Input size {res} * {res}")) +
    theme_light()
}


# check 32 ----------------------------------------------------------------

check_model$p_hat$distribution(pred_32, 32)
check_model$p_hat$null$x_dist$distribution(pred_32, 32)
check_model$p_hat$null$n$distribution(pred_32, 32)

# Boxplot of predicted probability conditional on factors
# n | null
pred_32 %>%
  filter(truth == "null") %>%
  ggplot() +
  geom_boxplot(aes(factor(n), V1)) +
  xlab("Number of observations") +
  ggtitle(quote("Distribution of " ~ hat(P)(y == not_null ~"|"~ truth == null) ~ "on the test set"), 
          subtitle = "Input size 32 * 32") +
  theme_light()

# Boxplot of predicted probability conditional on factors
# shape | n | e_sigma | not_null
pred_32 %>%
  filter(truth == "not_null") %>%
  filter(type == "polynomial") %>%
  mutate(e_sigma = factor(e_sigma, levels = c(4, 2, 1, 0.5))) %>%
  ggplot() +
  geom_hline(yintercept = 0.5, col = "red") +
  geom_boxplot(aes(factor(shape), V1, fill = factor(n))) +
  ylab(quote(hat(P)(y == not_null))) +
  xlab("shape") +
  scale_x_discrete(labels = c("U", "S", "M", "Triple-U")) +
  ggtitle(quote("Distribution of " ~ hat(P)(y == not_null ~"|"~ truth == not_null) ~ "on the test set"), 
          subtitle = "Input size 32 * 32") +
  facet_grid(~e_sigma, labeller = label_both) +
  scale_fill_brewer(palette = "YlOrRd") +
  labs(fill = "n") +
  theme_light()

pred_32 %>%
  filter(truth == "not_null") %>%
  filter(type == "polynomial") %>%
  mutate(e_sigma = factor(e_sigma, levels = c(4, 2, 1, 0.5))) %>%
  ggplot() +
  geom_hline(yintercept = 0.5, col = "red") +
  geom_boxplot(aes(factor(x_dist), V1, fill = factor(n))) +
  ylab(quote(hat(P)(y == not_null))) +
  xlab("Fitted value distribution") +
  scale_x_discrete(labels = c("discrete uniform", "lognormal", "normal", "uniform")) +
  ggtitle(quote("Distribution of " ~ hat(P)(y == not_null ~"|"~ truth == not_null) ~ "on the test set"), 
          subtitle = "Input size 32 * 32") +
  facet_grid(~e_sigma, labeller = label_both) +
  scale_fill_brewer(palette = "YlOrRd") +
  labs(fill = "n") +
  theme_light()

# Boxplot of predicted probability conditional on factors
# e_sigma
pred_32 %>%
  filter(truth == "not_null") %>%
  filter(type == "polynomial") %>%
  ggplot() +
  geom_boxplot(aes(factor(e_sigma), V1)) +
  ylab(quote(hat(P)(y == not_null))) +
  xlab(quote(sigma ~ "of the error term")) +
  ggtitle(quote("Distribution of " ~ hat(P)(y == not_null ~"|"~ truth == null) ~ "on the test set"), 
          subtitle = "Input size 32 * 32") +
  theme_light()

# Boxplot of predicted probability conditional on factors
# a
pred_32 %>%
  filter(truth == "null") %>%
  filter(type == "heteroskedasticity") %>%
  ggplot() +
  geom_boxplot(aes(factor(a), V1)) +
  xlab("a") +
  ggtitle(quote("Distribution of " ~ hat(P)(y == not_null ~"|"~ truth == null) ~ "on the test set"), 
          subtitle = "Input size 32 * 32") +
  theme_light()

# Boxplot of predicted probability conditional on factors
# b
pred_32 %>%
  filter(truth == "null") %>%
  filter(type == "heteroskedasticity") %>%
  ggplot() +
  geom_boxplot(aes(factor(b), V1)) +
  xlab("b") +
  ggtitle(quote("Distribution of " ~ hat(P)(y == not_null ~"|"~ truth == null) ~ "on the test set"), 
          subtitle = "Input size 32 * 32") +
  theme_light()


# check 64 ----------------------------------------------------------------

pred_64 %>%
  ggplot() +
  geom_histogram(aes(V1)) +
  ggtitle(quote("Distribution of " ~ hat(P)(y == not_null) ~ "on the test set"), 
          subtitle = "Input size 64 * 64") +
  xlab(quote(hat(P)(y == not_null))) +
  facet_grid(type ~ truth, labeller = label_both) +
  theme_light()


# common ------------------------------------------------------------------



pred_32_wrong <- pred_32 %>%
  filter(pred != truth) %>%
  pull(plot_uid)

pred_64_wrong <- pred_64 %>%
  filter(pred != truth) %>%
  pull(plot_uid)

both_wrong <- intersect(pred_32_wrong, pred_64_wrong)

