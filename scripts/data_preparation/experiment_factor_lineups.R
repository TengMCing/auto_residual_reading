
# library -----------------------------------------------------------------

library(tidyverse)
library(visage)
library(here)
library(glue)


# multicore ---------------------------------------------------------------

library(doMC)
library(foreach)

registerDoMC()

cat(glue("{getDoParWorkers()} workers is used by `doMC`!"))

set.seed(10086)

# Global setting ----------------------------------------------------------

# Number of samples per parameter
SAMPLE_PER_PARAMETER <- list(train = 10, test = 1)

# Number of parameter sets per model
TOTAL_NUM_PARAMETER <- 2000

# Data folder for saving plots
DATA_FOLDER <- "data/experiment_factor/lineups"

# draw_plots --------------------------------------------------------------

# The global uid for plots
PLOT_UID <- 0

# The global meta data for plots
PLOT_META <- data.frame()

# Draw plots for a violation model
draw_plots <- function(violation, not_null, null, n, meta_vector) {
  mod <- list()
  mod$not_null <- not_null
  mod$null <- null
  
  for (response in c("not_null", "null")) {
    for (data_type in c("train", "test")) {
      plot_dat <- map(1:SAMPLE_PER_PARAMETER[[data_type]], 
                      ~mod[[response]]$gen_lineup(n))
      
      # Speed up the plot drawing
      num_plots <- length(plot_dat)
      foreach(this_dat = plot_dat, 
              this_plot_id = (PLOT_UID + 1):(PLOT_UID + num_plots)) %dopar% {
                this_plot <- this_dat %>%
                  VI_MODEL$plot(theme = theme_light(), 
                                remove_axis = TRUE, 
                                remove_legend = TRUE, 
                                remove_grid_line = TRUE)
                
                # The lineup layout contains 4 rows and 5 cols
                ggsave(glue(here("{DATA_FOLDER}/native/{violation}/{data_type}/{response}/{this_plot_id}.png")), 
                       this_plot, 
                       width = 7, 
                       height = 7)
              }
      
      for (.unused in 1:num_plots) {
        PLOT_UID <<- PLOT_UID + 1
        PLOT_META <<- PLOT_META %>%
          bind_rows(c(plot_uid = PLOT_UID, 
                      meta_vector, 
                      data_type = data_type, 
                      response = response))
      }
    }
  }
}

# get_x_var ---------------------------------------------------------------

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

# parameter_range ---------------------------------------------------------

parameter_choice <- function(choices, transformer = NULL) {
  function() {
    if (!is.null(transformer)) 
      transformer(sample(choices, 1)) 
    else 
      sample(choices, 1)
  }
}

parameter_discrete <- function(low, high, transformer = NULL) {
  function() {
    if (!is.null(transformer)) 
      transformer(sample(low:high, 1)) 
    else 
      sample(low:high, 1)
  }
}

parameter_continuous <- function(low, high, transformer = NULL) {
  function() {
    if (!is.null(transformer)) 
      transformer((high - low) * runif(1) + low) 
    else 
      (high - low) * runif(1) + low
  }
}

# poly_data ---------------------------------------------------------------

poly_parameter_range <- list(shape = parameter_discrete(1, 4), 
                             e_sigma = parameter_choice(c(0.5, 1, 2, 4)),
                             x_dist = parameter_choice(c("uniform", "normal", "lognormal", "even_discrete")),
                             n = parameter_choice(c(50, 100, 300)))

for (i in 1:TOTAL_NUM_PARAMETER) {
  
  this_parameter <- map(poly_parameter_range, ~.x())
  
  draw_plots(violation = "poly",
             not_null = poly_model(shape = this_parameter$shape,
                                   x = get_x_var(this_parameter$x_dist),
                                   sigma = this_parameter$e_sigma),
             null = poly_model(shape = this_parameter$shape,
                               x = get_x_var(this_parameter$x_dist),
                               include_z = FALSE,
                               sigma = this_parameter$e_sigma),
             n = this_parameter$n,
             meta_vector = this_parameter)
}

# heter_data --------------------------------------------------------------

heter_parameter_range <- list(a = parameter_choice(c(-1, 0, 1)), 
                              b = parameter_choice(c(0.25, 1, 4, 16, 64)),
                              x_dist = parameter_choice(c("uniform", "normal", "lognormal", "even_discrete")),
                              n = parameter_choice(c(50, 100, 300)))

for (i in 1:TOTAL_NUM_PARAMETER) {
  
  this_parameter <- map(heter_parameter_range, ~.x())
  
  draw_plots(violation = "heter",
             not_null = heter_model(a = this_parameter$a,
                                    b = this_parameter$b,
                                    x = get_x_var(this_parameter$x_dist)),
             null = heter_model(a = this_parameter$a,
                                b = 0,
                                x = get_x_var(this_parameter$x_dist)),
             n = this_parameter$n,
             meta_vector = this_parameter)
  
}


# save_meta_data ----------------------------------------------------------

saveRDS(PLOT_META, here(glue("{DATA_FOLDER}/meta.rds")))

# mixed_data --------------------------------------------------------------

if (!dir.exists(here(glue("{DATA_FOLDER}/native/mixed")))) dir.create(here(glue("{DATA_FOLDER}/native/mixed")))
if (!dir.exists(here(glue("{DATA_FOLDER}/native/mixed/train")))) dir.create(here(glue("{DATA_FOLDER}/native/mixed/train")))
if (!dir.exists(here(glue("{DATA_FOLDER}/native/mixed/test")))) dir.create(here(glue("{DATA_FOLDER}/native/mixed/test")))

mixed_train_dest <- here(glue("{DATA_FOLDER}/native/mixed/train"))
mixed_test_dest <- here(glue("{DATA_FOLDER}/native/mixed/test"))
for (violation in c("poly", "heter")) {
  train_from <- here(glue("{DATA_FOLDER}/native/{violation}/train/."))
  test_from <- here(glue("{DATA_FOLDER}/native/{violation}/test/."))
  system(glue("cp -r {train_from} {mixed_train_dest}"))
  system(glue("cp -r {test_from} {mixed_test_dest}"))
}


# mixed_low_res -----------------------------------------------------------

PIL <- reticulate::import("PIL")

create_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
}

for (data_type in c("train", "test")) {
  for (response in c("null", "not_null")) {
    for (filename in list.files(here(glue("{DATA_FOLDER}/native/mixed/{data_type}/{response}")))) {
      im <- PIL$Image$open(glue("{DATA_FOLDER}/native/mixed/{data_type}/{response}/{filename}"))
      for (res in c(32L, 64L, 128L, 256L)) {
        new_im <- im$resize(c(res, res))
        create_dir(glue("{DATA_FOLDER}/{res}/mixed/{data_type}/{response}"))
        new_im$save(glue("{DATA_FOLDER}/{res}/mixed/{data_type}/{response}/{filename}"))
      }
      im$close()
    }
  }
}
