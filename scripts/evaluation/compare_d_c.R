library(tensorflow)
library(keras)
library(tidyverse)
library(yardstick)
library(visage)

mod <- load_model_tf("temp_model/mixed_224")
d_test_set <- flow_images_from_directory("data/experiment_factor/residual_plots/native/mixed/test",
                                         target_size = c(224L, 224L), 
                                         shuffle = FALSE)
c_test_set <- flow_images_from_directory("data/phn_extended/residual_plots/native/mixed/test",
                                         target_size = c(224L, 224L), 
                                         shuffle = FALSE)
d_pred <- mod$predict(d_test_set)
c_pred <- mod$predict(c_test_set)

d_meta <- readRDS("data/experiment_factor/residual_plots/meta.rds")
c_meta <- readRDS("data/phn_extended/residual_plots/meta.rds")

d_pred <- d_pred %>%
  as.data.frame() %>%
  mutate(truth = d_test_set$labels) %>%
  mutate(filename = d_test_set$filenames) %>%
  mutate(plot_uid = as.integer(gsub(".*/(.*).png", "\\1", filename))) %>%
  left_join(d_meta)

c_pred <- c_pred %>%
  as.data.frame() %>%
  mutate(truth = c_test_set$labels) %>%
  mutate(filename = c_test_set$filenames) %>%
  mutate(plot_uid = as.integer(gsub(".*/(.*).png", "\\1", filename))) %>%
  left_join(c_meta)

simple_es <- function(this_mod, n) {
  map_dbl(1:10, ~this_mod$gen(n) %>% this_mod$sample_effect_size()) %>%
    mean()
}

# Ensure the support of the predictor is [-1, 1]
stand_dist <- function(x) (x - min(x))/max(x - min(x)) * 2 - 1

get_x_var <- function(dist_name) {
  
  # Define the x variable
  rand_uniform_x <- rand_uniform(-1, 1)
  rand_normal_raw_x <- rand_normal(sigma = 0.3)
  rand_normal_x <- closed_form(~stand_dist(rand_normal_raw_x))
  rand_lognormal_raw_x <- rand_lognormal(sigma = 0.6)
  rand_lognormal_x <- closed_form(~stand_dist(rand_lognormal_raw_x/3 - 1))
  rand_discrete_x <- rand_uniform_d(-1, 1, k = 5, even = TRUE)
  
  switch(as.character(dist_name),
         uniform = rand_uniform_x,
         normal = rand_normal_x,
         lognormal = rand_lognormal_x,
         even_discrete = rand_discrete_x)
} 

get_x_var_extended <- function(dist_name, sigma = 0.3, k = 5, even = TRUE) {
  
  # Define the x variable
  rand_uniform_x <- rand_uniform(-1, 1)
  rand_normal_raw_x <- rand_normal(sigma = sigma)
  rand_normal_x <- closed_form(~stand_dist(rand_normal_raw_x))
  rand_lognormal_raw_x <- rand_lognormal(sigma = sigma)
  rand_lognormal_x <- closed_form(~stand_dist(rand_lognormal_raw_x/3 - 1))
  rand_discrete_x <- rand_uniform_d(-1, 1, k = k, even = even)
  
  switch(as.character(dist_name),
         uniform = rand_uniform_x,
         normal = rand_normal_x,
         lognormal = rand_lognormal_x,
         even_discrete = rand_discrete_x)
} 


d_pred_poly <- d_pred %>%
  filter(!is.na(shape)) %>%
  rowwise() %>%
  mutate(es = simple_es(poly_model(shape = shape, 
                                   sigma = e_sigma, 
                                   x = get_x_var(x_dist)),
                        n = n))

d_pred_heter <- d_pred %>%
  filter(!is.na(a)) %>%
  rowwise() %>%
  mutate(es = simple_es(heter_model(a = a, 
                                    b = b, 
                                    x = get_x_var(x_dist)),
                        n = n))


d_pred <- bind_rows(d_pred_poly, d_pred_heter)


GENERALIZED_POLY_MODEL <- bandicoot::new_class(POLY_MODEL, class_name = "GENERALIZED_POLY_MODEL")
bandicoot::register_method(GENERALIZED_POLY_MODEL, 
                           hermite = function(j) {
                             suppressMessages(as.function(mpoly::hermite(j)))
                           })
GENERALIZED_POLY_MODEL$set_formula(raw_z_formula = raw_z ~ hermite(j)((x - min(x))/max(x - min(x)) * 4 - 2))
bandicoot::register_method(GENERALIZED_POLY_MODEL, 
                           ..init.. = function(j = 2, 
                                               sigma = 1, 
                                               include_z = TRUE, 
                                               x = visage::rand_uniform(-1, 1, env = new.env(parent = parent.env(self))),
                                               e = visage::rand_normal(0, sigma, env = new.env(parent = parent.env(self)))) {
                             hermite <- self$hermite
                             raw_z <- visage::closed_form(eval(self$raw_z_formula), env = new.env(parent = parent.env(self)))
                             z <- visage::closed_form(eval(self$z_formula), env = new.env(parent = parent.env(self)))
                             
                             # Use the init method from the VI_MODEL class
                             bandicoot::use_method(self, visage::VI_MODEL$..init..)(
                               prm = list(j = j, include_z = include_z, sigma = sigma, x = x, raw_z = raw_z, z = z, e = e),
                               prm_type = list(j = "o", include_z = "o", sigma = "o", x = "r", raw_z = "r", z = "r", e = "r"),
                               formula = self$formula,
                               null_formula = self$null_formula,
                               alt_formula = self$alt_formula
                             )
                             
                             return(invisible(self))
                           })
generalized_poly_model <- GENERALIZED_POLY_MODEL$instantiate

c_pred_poly <- c_pred %>%
  filter(!is.na(j)) %>%
  rowwise() %>%
  mutate(es = simple_es(generalized_poly_model(j = j,
                                               sigma = e_sigma, 
                                               x = get_x_var_extended(x_dist)),
                        n = n))

c_pred_heter <- c_pred %>%
  filter(!is.na(a)) %>%
  rowwise() %>%
  mutate(es = simple_es(heter_model(a = a, 
                                    b = b, 
                                    x = get_x_var(x_dist)),
                        n = n))

c_pred <- bind_rows(c_pred_poly, c_pred_heter)

d_pred <- d_pred %>%
  mutate(model_reject = V1 > 0.5) %>%
  mutate(truth = ifelse(0, "not_null", "null")) %>%
  rename(effect_size = es) %>%
  mutate(type = ifelse(!is.na(shape), "poly", "heter"))

c_pred <- c_pred %>%
  mutate(model_reject = V1 > 0.5) %>%
  mutate(truth = ifelse(0, "not_null", "null")) %>%
  rename(effect_size = es) %>%
  mutate(type = ifelse(!is.na(j), "poly", "heter"))


# Estimate power curves
min_es <- min(c(d_pred$effect_size, c_pred$effect_size))

max_es <- max(c(d_pred$effect_size, c_pred$effect_size))

d_mod_pred <- d_pred %>%
  filter(x_dist == "uniform") %>%
  mutate(discrete = ifelse(model_reject, 0.04, 0.06)) %>%
  pivot_longer(c(discrete)) %>%
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

d_mod_pred %>%
  ggplot() +
  geom_line(aes(log_effect_size, power, col = name), size = 1) +
  facet_wrap(~type, nrow = 2) +
  theme_light() +
  xlab("Log of effect size") +
  ylab("Power") +
  labs(col = "")

c_mod_pred <- c_pred %>%
  filter(x_dist == "uniform") %>%
  mutate(cont = ifelse(model_reject, 0.04, 0.06)) %>%
  pivot_longer(c(cont)) %>%
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

d_mod_pred %>%
  bind_rows(c_mod_pred) %>%
  ggplot() +
  geom_line(aes(log_effect_size, power, col = name), size = 1) +
  facet_wrap(~type, nrow = 2) +
  theme_light() +
  xlab("Log of effect size") +
  ylab("Power") +
  labs(col = "")

# Check those that being incorrectly predicted
