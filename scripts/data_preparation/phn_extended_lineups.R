
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
TOTAL_NUM_PARAMETER <- 20

# Data folder for saving plots
DATA_FOLDER <- "data/phn_extended/lineups"

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
                  VI_MODEL$plot_lineup(theme = theme_light(), 
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

get_x_var <- function(dist_name, sigma = 0.3, k = 5, even = TRUE) {
  
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

# general_poly_model ------------------------------------------------------

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

# poly_data ---------------------------------------------------------------

poly_parameter_range <- list(j = parameter_discrete(2, 18), 
                             e_sigma = parameter_continuous(0.5, 2, function(x) x^2),
                             x_dist = parameter_choice(c("uniform", "normal", "lognormal", "even_discrete")),
                             x_sigma = parameter_continuous(0.2, 0.8),
                             x_k = parameter_discrete(5, 20),
                             x_even = parameter_choice(c(FALSE, TRUE)),
                             n = parameter_discrete(30, 1000))

for (i in 1:TOTAL_NUM_PARAMETER) {
  
  this_parameter <- map(poly_parameter_range, ~.x())
  
  draw_plots(violation = "poly",
             not_null = generalized_poly_model(j = this_parameter$j,
                                               x = get_x_var(this_parameter$x_dist,
                                                             this_parameter$x_sigma,
                                                             this_parameter$x_k,
                                                             this_parameter$x_even),
                                               sigma = this_parameter$e_sigma),
             null = generalized_poly_model(j = this_parameter$j,
                                           x = get_x_var(this_parameter$x_dist,
                                                         this_parameter$x_sigma,
                                                         this_parameter$x_k,
                                                         this_parameter$x_even),
                                           include_z = FALSE,
                                           sigma = this_parameter$e_sigma),
             n = this_parameter$n,
             meta_vector = this_parameter)
}

# heter_data --------------------------------------------------------------

heter_parameter_range <- list(a = parameter_discrete(-1, 1), 
                              b = parameter_continuous(0.1, 8, function(x) x^2),
                              x_dist = parameter_choice(c("uniform", "normal", "lognormal", "even_discrete")),
                              x_sigma = parameter_continuous(0.2, 0.8),
                              x_k = parameter_discrete(5, 20),
                              x_even = parameter_choice(c(FALSE, TRUE)),
                              n = parameter_discrete(30, 1000))

for (i in 1:TOTAL_NUM_PARAMETER) {
  
  this_parameter <- map(heter_parameter_range, ~.x())
  
  draw_plots(violation = "heter",
             not_null = heter_model(a = this_parameter$a,
                                    b = this_parameter$b,
                                    x = get_x_var(this_parameter$x_dist,
                                                  this_parameter$x_sigma,
                                                  this_parameter$x_k,
                                                  this_parameter$x_even)),
             null = heter_model(a = this_parameter$a,
                                b = 0,
                                x = get_x_var(this_parameter$x_dist,
                                              this_parameter$x_sigma,
                                              this_parameter$x_k,
                                              this_parameter$x_even)),
             n = this_parameter$n,
             meta_vector = this_parameter)
  
}


# non_normal --------------------------------------------------------------

non_normal_parameter_range <- list(x_dist = parameter_choice(c("uniform", "normal", "lognormal", "even_discrete")),
                                   x_sigma = parameter_continuous(0.2, 0.8),
                                   x_k = parameter_discrete(5, 20),
                                   x_even = parameter_choice(c(FALSE, TRUE)),
                                   e_dist = parameter_choice(c("uniform", "lognormal", "even_discrete", "t")),
                                   e_df = parameter_discrete(3, 10),
                                   e_sigma = parameter_continuous(0.5, 2, function(x) x^2),
                                   n = parameter_discrete(30, 1000))

get_e_var <- function(dist_name, df, e_sigma) {
  
  lognormal_sigma_table <- map_dbl(seq(0.001, 2, 0.001), ~sqrt((exp(.x^2) - 1) * exp(.x^2)))
  names(lognormal_sigma_table) <- seq(0.001, 2, 0.001)
  
  dist_name <- as.character(dist_name)
  
  if (dist_name == "uniform") {
    return(rand_uniform(a = -sqrt(12 * e_sigma^2)/2, 
                        b = sqrt(12 * e_sigma^2)/2,
                        env = new.env(parent = .GlobalEnv)))
  }
  
  if (dist_name == "lognormal") {
    table_index <- which.min(abs(lognormal_sigma_table - e_sigma))
    mod_sigma <- as.numeric(names(lognormal_sigma_table))[table_index]
    return(rand_lognormal(mu = 0,
                          sigma = mod_sigma,
                          env = new.env(parent = .GlobalEnv)))
  }
  
  if (dist_name == "even_discrete") {
    return(rand_uniform_d(a = -sqrt(12 * e_sigma^2)/2, 
                          b = sqrt(12 * e_sigma^2)/2,
                          even = TRUE,
                          env = new.env(parent = .GlobalEnv)))
  }
  
  if (dist_name == "t") {
    tau <- 1
    if (df > 2) tau <- sqrt(e_sigma^2 * (df - 2)/df)
    return(rand_t(tau = tau, 
                  df = df, 
                  env = new.env(parent = .GlobalEnv)))
  }
  
  return(rand_normal(sigma = e_sigma, 
                     env = new.env(parent = .GlobalEnv)))
}

for (i in 1:TOTAL_NUM_PARAMETER) {
  
  this_parameter <- map(non_normal_parameter_range, ~.x())
  
  draw_plots(violation = "non_normal",
             not_null = non_normal_model(x = get_x_var(this_parameter$x_dist,
                                                       this_parameter$x_sigma,
                                                       this_parameter$x_k,
                                                       this_parameter$x_even),
                                         e = get_e_var(this_parameter$e_dist,
                                                       this_parameter$e_df,
                                                       this_parameter$e_sigma)),
             null = non_normal_model(x = get_x_var(this_parameter$x_dist,
                                                   this_parameter$x_sigma,
                                                   this_parameter$x_k,
                                                   this_parameter$x_even),
                                     e = get_e_var("normal", 0, this_parameter$e_sigma)),
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
for (violation in c("poly", "heter", "non_normal")) {
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
