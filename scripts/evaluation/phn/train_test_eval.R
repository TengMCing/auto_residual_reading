library(tidyverse)
library(keras)
library(tensorflow)
library(here)
library(glue)
library(yardstick)

ALL_RES <- c(32L, 64L)

keras_mod <- list()
for (res in ALL_RES) {
  keras_mod[[res]] <- here(glue("keras_tuner/best_models/phn/es/{res}")) %>%
    load_model_tf()
}

train_pred <- list()
test_pred <- list()

train_result <- list()
test_result <- list()

meta <- read_csv("data/phn/residual_plots/meta.csv")
for (res in ALL_RES) {
  train_set <- flow_images_from_directory(here(glue("data/phn/residual_plots/{res}/train")),
                                          target_size = c(res, res),
                                          batch_size = 80000L,
                                          shuffle = FALSE)
  
  test_set <- flow_images_from_directory(here(glue("data/phn/residual_plots/{res}/test")),
                                         target_size = c(res, res),
                                         batch_size = 8000L,
                                         shuffle = FALSE)
  
  train_auxiliary <- data.frame(plot_uid = train_set$filenames %>%
                                  {gsub(".*/(.*).png", "\\1", .)} %>% 
                                  as.integer()) %>%
    left_join(select(meta, plot_uid, measure_monotonic, measure_sparse, 
                     measure_splines, measure_striped, n, effect_size))
  
  test_auxiliary <- data.frame(plot_uid = test_set$filenames %>%
                                 {gsub(".*/(.*).png", "\\1", .)} %>% 
                                 as.integer()) %>%
    left_join(select(meta, plot_uid, measure_monotonic, measure_sparse, 
                     measure_splines, measure_striped, n, effect_size))
  
  train_images <- reticulate::iter_next(train_set)[[1]]
  train_pred <- list(train_images, select(train_auxiliary, -plot_uid, -effect_size)) %>%
    {keras_mod[[res]]$predict(.)}
  rm(train_images)
  
  test_images <- reticulate::iter_next(test_set)[[1]]
  test_pred <- list(test_images, select(test_auxiliary, -plot_uid, -effect_size)) %>%
    {keras_mod[[res]]$predict(.)}
  rm(test_images)
  
  train_result[[res]] <- data.frame(plot_uid = train_auxiliary$plot_uid,
                                    pred = train_pred, 
                                    truth = log(train_auxiliary$effect_size + 1))
  test_result[[res]] <- data.frame(plot_uid = test_auxiliary$plot_uid,
                                   pred = test_pred, 
                                   truth = log(test_auxiliary$effect_size + 1))
  
  rm(train_set)
  rm(test_set)
}

result <- bind_rows(map_df(ALL_RES, ~mutate(train_result[[.x]], res = .x)),
                    map_df(ALL_RES, ~mutate(test_result[[.x]], res = .x))) %>%
  left_join(meta)

result %>%
  group_by(res, data_type, response) %>%
  rmse(truth = truth, estimate = pred) %>%
  kableExtra::kable() %>%
  kableExtra::kable_styling()

result %>%
  group_by(res, data_type, response) %>%
  mae(truth = truth, estimate = pred) %>%
  kableExtra::kable() %>%
  kableExtra::kable_styling()

result %>%
  filter(truth != 0) %>%
  ggplot() +
  geom_abline() +
  geom_point(aes(pred, truth), alpha = 0.1) +
  facet_grid(res ~ data_type,
             labeller = label_both) +
  ggtitle("Truth vs. prediction for not null residual plots") +
  xlab("Prediction") +
  ylab("Truth")

result %>%
  filter(truth != 0) %>%
  ggplot() +
  geom_hline(yintercept = 0) +
  geom_point(aes(pred, truth - pred), alpha = 0.1) +
  geom_smooth(aes(pred, truth - pred)) + 
  facet_grid(res ~ data_type,
             labeller = label_both) +
  ggtitle("Residuals vs. prediction for not null residual plots") +
  xlab("Prediction") +
  ylab("Residuals")

result %>%
  filter(truth == 0) %>%
  ggplot() +
  geom_histogram(aes(pred)) + 
  facet_wrap(res ~ data_type,
             labeller = label_both,
             scales = "free_y") +
  ggtitle("Distribution of prediction for null residual plots")


