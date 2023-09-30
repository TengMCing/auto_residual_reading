library(visage)
library(here)
library(glue)
library(tidyverse)
# OSX doesn't like `foreach` for some reasons, so we use furrr instead. 
library(furrr)
plan(multisession)

set.seed(10086)

proj_dir <- here()

# Number of plots per model
NUM_PLOTS_PER_MODEL <- 20000

# Define the x variable
get_x_var <- function(dist_name) {
  
  # Ensure the support of the predictor is [-1, 1]
  stand_dist <- function(x) (x - min(x))/max(x - min(x)) * 2 - 1
  
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


# poly --------------------------------------------------------------------


poly_model_parameters <- expand.grid(shape = 1:4,
                                     e_sigma = c(0.5, 1, 2, 4),
                                     x_dist = c("uniform", 
                                                "normal", 
                                                "lognormal", 
                                                "even_discrete"),
                                     n = c(50, 100, 300))

all_dat <- map(1:NUM_PLOTS_PER_MODEL, function(i) {
  this_row <- sample(nrow(poly_model_parameters), 1)
  shape_ <- poly_model_parameters$shape[this_row]
  e_sigma_ <- poly_model_parameters$e_sigma[this_row]
  x_dist_ <- poly_model_parameters$x_dist[this_row]
  n_ <- poly_model_parameters$n[this_row]
  
  not_null_mod <- poly_model(shape = shape_,
                             x = get_x_var(x_dist_),
                             include_z = TRUE,
                             sigma = e_sigma_)
  
  null_mod <- poly_model(shape = shape_,
                         x = get_x_var(x_dist_),
                         include_z = FALSE,
                         sigma = e_sigma_)
  
  list(row = this_row, 
       not_null_dat = not_null_mod$gen_lineup(n_) %>%
         select(.fitted, .resid, k), 
       null_dat = null_mod$gen_lineup(n_) %>%
         select(.fitted, .resid, k))
})

poly_metadata <- map_df(1:NUM_PLOTS_PER_MODEL, function(i) {
  poly_model_parameters[all_dat[[i]]$row, ]
})

not_null_dat <- map(1:NUM_PLOTS_PER_MODEL, function(i) {
  all_dat[[i]]$not_null_dat
})

null_dat <- map(1:NUM_PLOTS_PER_MODEL, function(i) {
  all_dat[[i]]$null_dat
})

plot_dat <- append(not_null_dat, null_dat)
rm(all_dat)
rm(not_null_dat)
rm(null_dat)

future_map2(plot_dat, 1:length(plot_dat), function(this_dat, this_id, proj_dir) {
  this_plot <- this_dat %>%
    VI_MODEL$plot_lineup(theme = theme_light(), 
                         remove_axis = TRUE, 
                         remove_legend = TRUE, 
                         remove_grid_line = TRUE)
  
  ggsave(glue("{proj_dir}/data/source/experiment_factor/lineups/poly_{this_id}.png"),
         this_plot,
         width = 7,
         height = 7)
}, proj_dir, .progress = TRUE)


# heter -------------------------------------------------------------------

heter_model_parameters <- expand.grid(a = c(-1, 0, 1),
                                      b = c(0.25, 1, 4, 16, 64),
                                      x_dist = c("uniform", 
                                                 "normal", 
                                                 "lognormal", 
                                                 "even_discrete"),
                                      n = c(50, 100, 300))

all_dat <- map(1:NUM_PLOTS_PER_MODEL, function(i) {
  this_row <- sample(nrow(heter_model_parameters), 1)
  a_ <- heter_model_parameters$a[this_row]
  b_ <- heter_model_parameters$b[this_row]
  x_dist_ <- heter_model_parameters$x_dist[this_row]
  n_ <- heter_model_parameters$n[this_row]
  
  not_null_mod <- heter_model(a = a_,
                              b = b_,
                              x = get_x_var(x_dist_))
  
  null_mod <- heter_model(a = a_,
                          b = 0,
                          x = get_x_var(x_dist_))
  
  list(row = this_row, 
       not_null_dat = not_null_mod$gen_lineup(n_) %>%
         select(.fitted, .resid, k), 
       null_dat = null_mod$gen_lineup(n_) %>%
         select(.fitted, .resid, k))
})

heter_metadata <- map_df(1:NUM_PLOTS_PER_MODEL, function(i) {
  heter_model_parameters[all_dat[[i]]$row, ]
})

not_null_dat <- map(1:NUM_PLOTS_PER_MODEL, function(i) {
  all_dat[[i]]$not_null_dat
})

null_dat <- map(1:NUM_PLOTS_PER_MODEL, function(i) {
  all_dat[[i]]$null_dat
})

plot_dat <- append(not_null_dat, null_dat)
rm(all_dat)
rm(not_null_dat)
rm(null_dat)


future_map2(plot_dat, 1:length(plot_dat), function(this_dat, this_id, proj_dir) {
  this_plot <- this_dat %>%
    VI_MODEL$plot_lineup(theme = theme_light(), 
                         remove_axis = TRUE, 
                         remove_legend = TRUE, 
                         remove_grid_line = TRUE)
  
  ggsave(glue("{proj_dir}/data/source/experiment_factor/lineups/heter_{this_id}.png"),
         this_plot,
         width = 7,
         height = 7)
}, proj_dir, .progress = TRUE)


# metadata ----------------------------------------------------------------

create_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
}

create_dir(glue("{proj_dir}/data/metadata/experiment_factor"))

num_of_train <- floor(NUM_PLOTS_PER_MODEL * 0.9)
num_of_test <- NUM_PLOTS_PER_MODEL - num_of_train

bind_rows(poly_metadata %>%
            mutate(plot_id = glue("poly_{1:NUM_PLOTS_PER_MODEL}")) %>%
            mutate(data_type = c(rep("train", num_of_train), rep("test", num_of_test))),
          heter_metadata %>%
            mutate(plot_id = glue("heter_{1:NUM_PLOTS_PER_MODEL}")) %>%
            mutate(data_type = c(rep("train", num_of_train), rep("test", num_of_test)))) %>%
  write_csv(glue("{proj_dir}/data/metadata/experiment_factor/lineups.csv"))


# low_res -----------------------------------------------------------------

PIL <- reticulate::import("PIL")

for (filename in list.files(glue("{proj_dir}/data/source/experiment_factor/lineups"))) {
  im <- PIL$Image$open(glue("{proj_dir}/data/source/experiment_factor/lineups/{filename}"))
  for (res in c(32L, 64L, 128L, 256L)) {
    new_im <- im$resize(c(res, res))
    create_dir(glue("{proj_dir}/data/{res}/experiment_factor/lineups"))
    new_im$save(glue("{proj_dir}/data/{res}/experiment_factor/lineups/{filename}"))
  }
  im$close()
}

