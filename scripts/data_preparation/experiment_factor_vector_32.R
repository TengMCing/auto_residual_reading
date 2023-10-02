
# library -----------------------------------------------------------------

library(tidyverse)
library(visage)
library(here)
library(glue)

# Global setting ----------------------------------------------------------

# Number of samples per parameter
SAMPLE_PER_PARAMETER <- list(train = 10, test = 1)

# Number of parameter sets per model
TOTAL_NUM_PARAMETER <- 2000

# Data folder for saving plots
DATA_FOLDER <- "data/experiment_factor/vector"

# The global uid for plots
PLOT_UID <- 0

# The global meta data for plots
PLOT_META <- data.frame()

# Target length
TARGET_LEN <- 32

# residual_to_vector ------------------------------------------------------


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

gen_dat <- function(violation, not_null, null, n, meta_vector) {
  mod <- list()
  mod$not_null <- not_null
  mod$null <- null
  
  result_dat <- NULL
  
  for (response in c("not_null", "null")) {
    for (data_type in c("train", "test")) {
      
      plot_dat <- map(1:SAMPLE_PER_PARAMETER[[data_type]], 
                      ~mod[[response]]$gen(n))
      
      num_plots <- length(plot_dat)
      this_dat <- map(plot_dat, 
                      ~residual_to_vector(.x$.resid, .x$.fitted)) %>%
        reduce(cbind) %>%
        t() %>%
        `rownames<-`(NULL) %>%
        cbind((PLOT_UID + 1):(PLOT_UID + num_plots))
      
      if (is.null(result_dat)) result_dat <- this_dat else result_dat <- rbind(result_dat, this_dat)
      
      for (i in 1:num_plots) {
        PLOT_UID <<- PLOT_UID + 1
        PLOT_META <<- PLOT_META %>%
          bind_rows(c(plot_uid = PLOT_UID, 
                      this_parameter, 
                      data_type = data_type, 
                      response = response))
      }
    }
  }
  
  return(result_dat)
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

poly_dat <- NULL

for (i in 1:TOTAL_NUM_PARAMETER) {
  
  this_parameter <- map(poly_parameter_range, ~.x())
  
  poly_dat <- rbind(poly_dat, 
                    gen_dat(violation = "poly",
                            not_null = poly_model(shape = this_parameter$shape,
                                                  x = get_x_var(this_parameter$x_dist),
                                                  sigma = this_parameter$e_sigma),
                            null = poly_model(shape = this_parameter$shape,
                                              x = get_x_var(this_parameter$x_dist),
                                              include_z = FALSE,
                                              sigma = this_parameter$e_sigma),
                            n = this_parameter$n,
                            meta_vector = this_parameter))
}

poly_dat <- poly_dat %>%
  as.data.frame() %>%
  rename(plot_uid = last_col()) %>%
  left_join(select(PLOT_META, plot_uid, data_type, response))
  

# heter_data --------------------------------------------------------------

heter_parameter_range <- list(a = parameter_choice(c(-1, 0, 1)), 
                              b = parameter_choice(c(0.25, 1, 4, 16, 64)),
                              x_dist = parameter_choice(c("uniform", "normal", "lognormal", "even_discrete")),
                              n = parameter_choice(c(50, 100, 300)))

heter_dat <- NULL

for (i in 1:TOTAL_NUM_PARAMETER) {
  
  this_parameter <- map(heter_parameter_range, ~.x())
  
  heter_dat <- rbind(heter_dat, 
        gen_dat(violation = "heter",
                not_null = heter_model(a = this_parameter$a,
                                       b = this_parameter$b,
                                       x = get_x_var(this_parameter$x_dist)),
                null = heter_model(a = this_parameter$a,
                                   b = 0,
                                   x = get_x_var(this_parameter$x_dist)),
                n = this_parameter$n,
                meta_vector = this_parameter))
  
}

heter_dat <- heter_dat %>%
  as.data.frame() %>%
  rename(plot_uid = last_col()) %>%
  left_join(select(PLOT_META, plot_uid, data_type, response))


# save_meta_data ----------------------------------------------------------

dir.create(here(glue("{DATA_FOLDER}")), recursive = TRUE)
saveRDS(PLOT_META, here(glue("{DATA_FOLDER}/meta.rds")))

# mixed -------------------------------------------------------------------

mixed_dat <- bind_rows(poly_dat, heter_dat)
mixed_train_dat <- mixed_dat %>%
  filter(data_type == "train")
mixed_test_dat <- mixed_dat %>%
  filter(data_type == "test")
write_csv(mixed_train_dat, glue("{DATA_FOLDER}/train.csv"))
write_csv(mixed_test_dat, glue("{DATA_FOLDER}/test.csv"))
