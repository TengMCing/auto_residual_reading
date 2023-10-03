library(tensorflow)
library(keras)
library(tidyverse)
library(yardstick)
library(visage)

mod <- load_model_tf("keras_tuner/best_models/experiment_factor/vector/64")
test_set <- read_csv("data/experiment_factor/vector/64/test.csv")

test_set$pred <- mod$predict(test_set[,1:64]) %>% as.numeric()
test_set$pred_label <- ifelse(test_set$pred > 0.5, "not_null", "null")

conf_mat(test_set %>%
           mutate(pred_label = factor(pred_label),
                  response = factor(response)), 
         estimate = pred_label, 
         truth = response)

bal_accuracy(test_set %>%
               mutate(pred_label = factor(pred_label),
                      response = factor(response)), 
             estimate = pred_label, 
             truth = response)

vi_lineup <- get_vi_lineup()

std_var <- function(x) (x - min(x))/max(x - min(x)) * 2 - 1

residual_to_vector <- function(resids, fitted, target_len = TARGET_LEN) {
  ordered_resids <- resids[order(fitted)]
  ordered_fitted <- sort(fitted)
  
  fitted_gap <- (max(ordered_fitted) - min(ordered_fitted))/target_len
  
  fitted_position <- floor((ordered_fitted - min(ordered_fitted))/fitted_gap) + 1L
  fitted_position <- ifelse(fitted_position > target_len, target_len, fitted_position)
  
  map_dbl(1:target_len, function(i) {
    mean(ordered_resids[i == fitted_position])
  }) %>%
    {ifelse(is.nan(.), 0, .)} %>%
    std_var()
}

lineup_dat <- map(vi_lineup, function(lineup) {
  this_dat <- lineup$data %>%
    filter(null == FALSE)
  residual_to_vector(this_dat$.resid, this_dat$.fitted, 64)
}) %>%
  reduce(rbind) %>%
  `rownames<-`(NULL) %>%
  as.data.frame() %>%
  mutate(unique_lineup_id = names(vi_lineup))

lineup_dat$pred <- mod$predict(lineup_dat[,1:64]) %>% as.numeric()
lineup_dat$pred_label <- ifelse(lineup_dat$pred > 0.5, "not_null", "null")

lineup_dat <- lineup_dat %>%
  left_join(vi_survey %>%
              group_by(unique_lineup_id) %>%
              summarise(across(everything(), first)) %>%
              select(unique_lineup_id, 
                     attention_check, 
                     null_lineup, 
                     answer, 
                     effect_size,
                     conventional_p_value,
                     p_value,
                     type,
                     shape,
                     a,
                     b,
                     x_dist,
                     e_dist,
                     e_sigma,
                     include_z,
                     n,
                     alpha))

lineup_dat <- mutate(lineup_dat, model_reject = lineup_dat$pred > 0.5)

library(ggmosaic)
lineup_dat %>%
  filter(!null_lineup) %>%
  filter(!attention_check) %>%
  filter(x_dist == "uniform") %>%
  mutate(model_reject = ifelse(model_reject, "Reject", "Not")) %>%
  mutate(conv_reject = ifelse(conventional_p_value <= 0.05, "Reject", "Not")) %>%
  mutate(across(c(model_reject, conv_reject), 
                ~factor(.x, levels = c("Reject", "Not")))) %>%
  ggplot() +
  geom_mosaic(aes(x = ggmosaic::product(model_reject, conv_reject), 
                  fill = model_reject)) +
  facet_grid(~type) +
  ylab("Computer visual model") +
  xlab("Conventional tests") +
  labs(fill = "Conventional tests") +
  scale_fill_brewer("", palette = "Dark2") +
  theme_bw() +
  theme(legend.position = "none") +
  coord_fixed()

# Compare to visual test
lineup_dat %>%
  filter(!null_lineup) %>%
  filter(!attention_check) %>%
  filter(x_dist == "uniform") %>%
  mutate(model_reject = ifelse(model_reject, "Reject", "Not")) %>%
  mutate(visual_reject = ifelse(p_value <= 0.05, "Reject", "Not")) %>%
  mutate(across(c(model_reject, visual_reject), 
                ~factor(.x, levels = c("Reject", "Not")))) %>%
  ggplot() +
  geom_mosaic(aes(x = ggmosaic::product(model_reject, visual_reject), 
                  fill = model_reject)) +
  facet_grid(~type) +
  ylab("Computer visual model") +
  xlab("Visual tests") +
  labs(fill = "Visual tests") +
  scale_fill_brewer("", palette = "Dark2") +
  theme_bw() +
  theme(legend.position = "none") +
  coord_fixed()

lineup_dat %>%
  filter(!null_lineup) %>%
  filter(!attention_check) %>%
  filter(x_dist == "uniform") %>%
  mutate(model_reject = ifelse(model_reject, "Reject", "Not")) %>%
  mutate(visual_reject = ifelse(p_value <= 0.05, "Reject", "Not")) %>%
  mutate(across(c(model_reject, visual_reject), 
                ~factor(.x, levels = c("Reject", "Not")))) %>%
  ggplot() +
  geom_mosaic(aes(x = ggmosaic::product(visual_reject, model_reject), 
                  fill = visual_reject)) +
  facet_grid(~type) +
  xlab("Computer visual model") +
  ylab("Visual tests") +
  labs(fill = "Visual tests") +
  scale_fill_brewer("", palette = "Dark2") +
  theme_bw() +
  theme(legend.position = "none") +
  coord_fixed()

lineup_dat %>%
  filter(!null_lineup) %>%
  filter(!attention_check) %>%
  mutate(diff_decision = ifelse((conventional_p_value <= 0.05) != model_reject, "Yes", "No")) %>%
  mutate(model_reject = ifelse(model_reject, "Reject", "Not")) %>%
  ggplot() +
  ggbeeswarm::geom_quasirandom(aes(log(effect_size), model_reject, col = diff_decision),
                               orientation = "y",
                               alpha = 0.6) +
  facet_grid(x_dist~type, scales = "free_x") +
  xlab("Log of effect size") +
  ylab("Computer vision model") +
  labs(col = "Different from conventional test") +
  theme_light() +
  theme(legend.position = "bottom") +
  scale_color_brewer(palette = "Dark2")

lineup_dat %>%
  filter(!null_lineup) %>%
  filter(!attention_check) %>%
  mutate(diff_decision = ifelse((p_value <= 0.05) != model_reject, "Yes", "No")) %>%
  mutate(model_reject = ifelse(model_reject, "Reject", "Not")) %>%
  ggplot() +
  ggbeeswarm::geom_quasirandom(aes(log(effect_size), model_reject, col = diff_decision),
                               orientation = "y",
                               alpha = 0.6) +
  facet_grid(x_dist~type, scales = "free_x") +
  xlab("Log of effect size") +
  ylab("Computer vision model") +
  labs(col = "Different from visual test") +
  theme_light() +
  theme(legend.position = "bottom") +
  scale_color_brewer(palette = "Set1")

# Consider a new graphical representation of this data.
# Bin the effect size
# 

# Estimate power curves
min_es <- lineup_dat %>%
  filter(!null_lineup) %>%
  filter(!attention_check) %>%
  filter(x_dist == "uniform") %>%
  pull(effect_size) %>%
  min()

max_es <- lineup_dat %>%
  filter(!null_lineup) %>%
  filter(!attention_check) %>%
  filter(x_dist == "uniform") %>%
  pull(effect_size) %>%
  max()

lineup_mod_pred <- lineup_dat %>%
  filter(!null_lineup) %>%
  filter(!attention_check) %>%
  filter(x_dist == "uniform") %>%
  mutate(computer_vision = ifelse(model_reject, 0.04, 0.06)) %>%
  mutate(visual_test = p_value) %>%
  mutate(conventional_test = conventional_p_value) %>%
  pivot_longer(c(computer_vision, visual_test, conventional_test)) %>%
  mutate(reject = value <= 0.05) %>%
  select(type, effect_size, name, reject) %>%
  mutate(offset0 = log(0.05/0.95)) %>%
  nest(dat = c(effect_size, offset0, reject)) %>%
  mutate(mod = map(dat, 
                   ~glm(reject ~ effect_size - 1, 
                        family = binomial(), 
                        data = .x,
                        offset = offset0))) %>%
  mutate(power = map(mod, function(mod) {
    result <- data.frame(effect_size = seq(min_es, max_es, 0.1),
                         offset0 = log(0.05/0.95))
    result$power <- predict(mod, type = "response", newdata = result)
    result
  })) %>%
  select(-dat, -mod) %>%
  unnest(power) %>%
  mutate(log_effect_size = log(effect_size))

lineup_mod_pred %>%
  ggplot() +
  geom_line(aes(log_effect_size, power, col = name), size = 1) +
  facet_wrap(~type, nrow = 2) +
  theme_light() +
  xlab("Log of effect size") +
  ylab("Power") +
  labs(col = "")
 
