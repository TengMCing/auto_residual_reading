
# library -----------------------------------------------------------------

library(tidyverse)
library(visage)
library(here)
library(glue)
library(cassowaryr)


# multicore ---------------------------------------------------------------

library(doMC)
library(foreach)

registerDoMC()

cat(glue("{getDoParWorkers()} workers is used by `doMC`!"))

set.seed(2023)

# Global setting ----------------------------------------------------------

# Number of samples per parameter
SAMPLE_PER_PARAMETER <- list(train = 10, test = 1)

# Number of parameter sets per model
TOTAL_NUM_PARAMETER <- 4000

# Data folder for saving plots
DATA_FOLDER <- "data/phn/residual_plots"

# draw_plots --------------------------------------------------------------


# The global uid for plots
PLOT_UID <- 0

# The global meta data for plots
PLOT_META <- data.frame()

# Draw plots for a violation model
draw_plots <- function(not_null, null, n, meta_vector) {
  mod <- list()
  mod$not_null <- not_null
  mod$null <- null
  
  result <- vector("list", length = 4)
  result_counter <- 1
  
  for (response in c("not_null", "null")) {
    for (data_type in c("train", "test")) {
      plot_dat <- map(1:SAMPLE_PER_PARAMETER[[data_type]], 
                      ~mod[[response]]$gen(n))
      
      result[[result_counter]] <- plot_dat
      result_counter <- result_counter + 1
      
      es <- map_dbl(plot_dat, ~mod[[response]]$sample_effect_size(.x))
      
      # if (any(is.infinite(es))) {
      #   browser()
      #   map_dbl(plot_dat, ~mod[[response]]$sample_effect_size(.x))
      # }
      
      try_or_zero <- function(fn, ...) {
        try_result <- try(fn(...), silent = TRUE)
        if (inherits(try_result, "try-error")) return(0)
        return(try_result)
      }
      
      measure_clumpy <- map_dbl(plot_dat, ~try_or_zero(sc_clumpy2, .x$.fitted, .x$.resid)) %>% 
        {ifelse(is.na(.), 0, .)}
      measure_dcor <- map_dbl(plot_dat, ~try_or_zero(sc_dcor, .x$.fitted, .x$.resid)) %>% 
        {ifelse(is.na(.), 0, .)}
      measure_monotonic <- map_dbl(plot_dat, ~try_or_zero(sc_monotonic, .x$.fitted, .x$.resid)) %>% 
        {ifelse(is.na(.), 0, .)}
      measure_outlying <- map_dbl(plot_dat, ~try_or_zero(sc_outlying, .x$.fitted, .x$.resid)) %>% 
        {ifelse(is.na(.), 0, .)}
      measure_skewed <- map_dbl(plot_dat, ~try_or_zero(sc_skewed, .x$.fitted, .x$.resid)) %>% 
        {ifelse(is.na(.), 0, .)}
      measure_skinny <- map_dbl(plot_dat, ~try_or_zero(sc_skinny, .x$.fitted, .x$.resid)) %>% 
        {ifelse(is.na(.), 0, .)}
      measure_sparse <- map_dbl(plot_dat, ~try_or_zero(sc_sparse2, .x$.fitted, .x$.resid)) %>% 
        {ifelse(is.na(.), 0, .)}
      measure_splines <- map_dbl(plot_dat, ~try_or_zero(sc_splines, .x$.fitted, .x$.resid)) %>% 
        {ifelse(is.na(.), 0, .)}
      measure_striated <- map_dbl(plot_dat, ~try_or_zero(sc_striated2, .x$.fitted, .x$.resid)) %>% 
        {ifelse(is.na(.), 0, .)}
      measure_stringy <- map_dbl(plot_dat, ~try_or_zero(sc_stringy, .x$.fitted, .x$.resid)) %>% 
        {ifelse(is.na(.), 0, .)}
      measure_striped <- map_dbl(plot_dat, ~try_or_zero(sc_striped, .x$.fitted, .x$.resid)) %>% 
        {ifelse(is.na(.), 0, .)}
      
      # Speed up the plot drawing
      num_plots <- length(plot_dat)
      foreach(this_dat = plot_dat, 
              this_plot_id = (PLOT_UID + 1):(PLOT_UID + num_plots)) %do% {
                this_plot <- this_dat %>%
                  VI_MODEL$plot(theme = theme_light(base_size = 11/5), 
                                remove_axis = TRUE, 
                                remove_legend = TRUE, 
                                remove_grid_line = TRUE)
                
                # The lineup layout contains 4 rows and 5 cols
                ggsave(glue(here("{DATA_FOLDER}/native/{data_type}/{response}/{this_plot_id}.png")), 
                       this_plot, 
                       width = 7/5, 
                       height = 7/4)
              }
      
      for (i in 1:num_plots) {
        PLOT_UID <<- PLOT_UID + 1
        PLOT_META <<- PLOT_META %>%
          bind_rows(c(plot_uid = PLOT_UID, 
                      meta_vector, 
                      data_type = data_type, 
                      response = response,
                      effect_size = es[i],
                      measure_clumpy = measure_clumpy[i],
                      measure_dcor = measure_dcor[i],
                      measure_monotonic = measure_monotonic[i],
                      measure_outlying = measure_outlying[i],
                      measure_skewed = measure_skewed[i],
                      measure_skinny = measure_skinny[i],
                      measure_sparse = measure_sparse[i],
                      measure_splines = measure_splines[i],
                      measure_striated = measure_striated[i],
                      measure_stringy = measure_stringy[i],
                      measure_striped = measure_striped[i]))
      }
    }
  }
  
  return(list_flatten(result))
}


# get_x_var ---------------------------------------------------------------

# Ensure the support of the predictor is [-1, 1]
stand_dist <- function(x) (x - min(x))/max(x - min(x)) * 2 - 1

get_x1_var <- function(dist_name, sigma = 0.3, k = 5, even = TRUE) {
  
  # Define the x variable
  rand_uniform_x1 <- rand_uniform(-1, 1)
  rand_normal_raw_x1 <- rand_normal(sigma = sigma)
  rand_normal_x1 <- closed_form(~stand_dist(rand_normal_raw_x1))
  rand_lognormal_raw_x1 <- rand_lognormal(sigma = sigma)
  rand_lognormal_x1 <- closed_form(~stand_dist(rand_lognormal_raw_x1/3 - 1))
  rand_discrete_x1 <- rand_uniform_d(-1, 1, k = k, even = even)
  
  switch(as.character(dist_name),
         uniform = rand_uniform_x1,
         normal = rand_normal_x1,
         lognormal = rand_lognormal_x1,
         even_discrete = rand_discrete_x1)
} 

get_x2_var <- function(dist_name, sigma = 0.3, k = 5, even = TRUE) {
  
  # Define the x variable
  rand_uniform_x2 <- rand_uniform(-1, 1)
  rand_normal_raw_x2 <- rand_normal(sigma = sigma)
  rand_normal_x2 <- closed_form(~stand_dist(rand_normal_raw_x2))
  rand_lognormal_raw_x2 <- rand_lognormal(sigma = sigma)
  rand_lognormal_x2 <- closed_form(~stand_dist(rand_lognormal_raw_x2/3 - 1))
  rand_discrete_x2 <- rand_uniform_d(-1, 1, k = k, even = even)
  
  switch(as.character(dist_name),
         uniform = rand_uniform_x2,
         normal = rand_normal_x2,
         lognormal = rand_lognormal_x2,
         even_discrete = rand_discrete_x2)
} 


# get_e_var ---------------------------------------------------------------

get_e_var <- function(dist_name, e_sigma, e_k, e_even) {
  
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
                          even = e_even,
                          k = e_k,
                          env = new.env(parent = .GlobalEnv)))
  }
  
  return(rand_normal(sigma = e_sigma,
                     env = new.env(parent = .GlobalEnv)))
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


# phn_data ----------------------------------------------------------------

phn_parameter_range <- list(j = parameter_discrete(2, 18),
                            a = parameter_continuous(-1, 1),
                            b = parameter_continuous(0, 8, function(x) x^2),
                            include_z = parameter_choice(c(FALSE, TRUE)),
                            include_x2 = parameter_choice(c(FALSE, TRUE)),
                            include_heter = parameter_choice(c(FALSE, TRUE)),
                            include_non_normal = parameter_choice(c(FALSE, TRUE)),
                            x1_dist = parameter_choice(c("uniform", "normal", "lognormal", "even_discrete")),
                            x1_sigma = parameter_continuous(0.3, 0.6),
                            x1_k = parameter_discrete(5, 10),
                            x1_even = parameter_choice(c(FALSE, TRUE)),
                            x2_dist = parameter_choice(c("uniform", "normal", "lognormal", "even_discrete")),
                            x2_sigma = parameter_continuous(0.3, 0.6),
                            x2_k = parameter_discrete(5, 10),
                            x2_even = parameter_choice(c(FALSE, TRUE)),
                            e_dist = parameter_choice(c("uniform", "lognormal", "even_discrete")),
                            e_even = parameter_choice(c(FALSE, TRUE)),
                            e_k = parameter_discrete(5, 10),
                            e_sigma = parameter_continuous(0.25, 2, function(x) x^2),
                            n = parameter_discrete(50, 500))

phn_result <- vector("list", length = TOTAL_NUM_PARAMETER)

for (i in 1:TOTAL_NUM_PARAMETER) {
  
  while (TRUE) {
    this_parameter <- map(phn_parameter_range, ~.x())
    if (!(this_parameter$include_z == FALSE && this_parameter$include_heter == FALSE && this_parameter$include_non_normal == FALSE)) break
  }
  
  draw_plots(not_null = phn_model(j = this_parameter$j,
                                  a = this_parameter$a,
                                  b = this_parameter$include_heter * this_parameter$b,
                                  include_z = this_parameter$include_z,
                                  include_x2 = this_parameter$include_x2,
                                  x1 = get_x1_var(this_parameter$x1_dist,
                                                 this_parameter$x1_sigma,
                                                 this_parameter$x1_k,
                                                 this_parameter$x1_even),
                                  x2 = get_x2_var(this_parameter$x2_dist,
                                                 this_parameter$x2_sigma,
                                                 this_parameter$x2_k,
                                                 this_parameter$x2_even),
                                  e = get_e_var(ifelse(this_parameter$include_non_normal == TRUE, this_parameter$e_dist, "normal"),
                                                this_parameter$e_sigma,
                                                this_parameter$e_k,
                                                this_parameter$e_even)),
             null = phn_model(j = this_parameter$j,
                              a = this_parameter$a,
                              b = 0,
                              include_z = FALSE,
                              include_x2 = this_parameter$include_x2,
                              x1 = get_x1_var(this_parameter$x1_dist,
                                             this_parameter$x1_sigma,
                                             this_parameter$x1_k,
                                             this_parameter$x1_even),
                              x2 = get_x2_var(this_parameter$x2_dist,
                                             this_parameter$x2_sigma,
                                             this_parameter$x2_k,
                                             this_parameter$x2_even),
                              e = get_e_var("normal",
                                            this_parameter$e_sigma,
                                            this_parameter$e_k,
                                            this_parameter$e_even)),
             n = this_parameter$n,
             meta_vector = this_parameter) -> phn_result[[i]]
  
}

phn_result <- phn_result %>% list_flatten()

# save_meta_data ----------------------------------------------------------

phn_result %>% saveRDS(file = here(glue("{DATA_FOLDER}/raw.rds")))
write_csv(PLOT_META, here(glue("{DATA_FOLDER}/meta.csv")))

# mixed_low_res -----------------------------------------------------------

PIL <- reticulate::import("PIL")

create_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
}

for (data_type in c("train", "test")) {
  for (response in c("null", "not_null")) {
    for (filename in list.files(here(glue("{DATA_FOLDER}/native/{data_type}/{response}")))) {
      im <- PIL$Image$open(glue("{DATA_FOLDER}/native/{data_type}/{response}/{filename}"))
      for (res in c(32L, 64L, 128L, 256L)) {
        new_im <- im$resize(c(res, res))
        create_dir(glue("{DATA_FOLDER}/{res}/{data_type}/{response}"))
        new_im$save(glue("{DATA_FOLDER}/{res}/{data_type}/{response}/{filename}"))
      }
      im$close()
    }
  }
}
