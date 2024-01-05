
# library -----------------------------------------------------------------

library(tidyverse)
library(visage)
library(here)
library(glue)
library(cassowaryr)

set.seed(2023)

# Global setting ----------------------------------------------------------

# Number of samples per parameter
SAMPLE_PER_PARAMETER <- list(train = 10, test = 1)

# Number of parameter sets per model
TOTAL_NUM_PARAMETER <- 8000

# Data folder for saving plots
DATA_FOLDER <- "data/phn/es"


# y_check -----------------------------------------------------------------

GLOBAL_Y_TRAIN <- c()
GLOBAL_Y_TEST <- c()
NUM_BIN <- 100
MAX_Y <- 7
BINS <- (1:NUM_BIN - 1) * (MAX_Y / (NUM_BIN - 1))

y_check_train <- function(new_y) {
  target_bin <- sum(new_y >= BINS)
  # if (target_bin == 1 & new_y != 0) return(FALSE)
  
  bin_lower <- (target_bin - 1) * (MAX_Y / (NUM_BIN - 1))
  
  bin_upper <- target_bin * (MAX_Y / (NUM_BIN - 1))
  if (new_y >= MAX_Y) bin_upper <- Inf
  
  bin_freq <- sum(GLOBAL_Y_TRAIN >= bin_lower & GLOBAL_Y_TRAIN < bin_upper)
  max_freq <- floor(TOTAL_NUM_PARAMETER * SAMPLE_PER_PARAMETER$train / NUM_BIN)
  
  if (bin_freq < max_freq) {
    GLOBAL_Y_TRAIN <<- c(GLOBAL_Y_TRAIN, new_y)
    return(TRUE)
  } else {
    return(FALSE)
  }
}

y_check_test <- function(new_y) {
  target_bin <- sum(new_y >= BINS)
  # if (target_bin == 1 & new_y != 0) return(FALSE)
  
  bin_lower <- (target_bin - 1) * (MAX_Y / (NUM_BIN - 1))
  
  bin_upper <- target_bin * (MAX_Y / (NUM_BIN - 1))
  if (new_y >= MAX_Y) bin_upper <- Inf
  
  bin_freq <- sum(GLOBAL_Y_TEST >= bin_lower & GLOBAL_Y_TEST < bin_upper)
  max_freq <- floor(TOTAL_NUM_PARAMETER * SAMPLE_PER_PARAMETER$test / NUM_BIN)
  
  if (bin_freq < max_freq) {
    GLOBAL_Y_TEST <<- c(GLOBAL_Y_TEST, new_y)
    return(TRUE)
  } else {
    return(FALSE)
  }
}


# save_plot_data ----------------------------------------------------------



RAW_DATA <- list()

# The global uid for plots
PLOT_UID <- 0

# The global meta data for plots
PLOT_META <- data.frame()

save_plot_data <- function(dat, es, data_type, meta_vector) {
  
  PLOT_UID <<- PLOT_UID + 1
  
  PLOT_META <<- PLOT_META %>%
    bind_rows(c(plot_uid = PLOT_UID, 
                meta_vector, 
                data_type = data_type, 
                effect_size = es))
  
  RAW_DATA[[PLOT_UID]] <<- dat
  
  return(invisible(NULL))
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
                            b = parameter_continuous(0, 10, function(x) x^2),
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
                            e_sigma = parameter_continuous(0.25, 3, function(x) x^2),
                            n = parameter_discrete(50, 500))

phn_result <- vector("list", length = TOTAL_NUM_PARAMETER)

for (i in 1:TOTAL_NUM_PARAMETER) {
  
  print(i)
  current_num_plots <- 0
  
  while (TRUE) {
    this_parameter <- map(phn_parameter_range, ~.x())
    
    mod <- phn_model(j = this_parameter$j,
                     a = this_parameter$a,
                     b = (this_parameter$include_heter > 0) * this_parameter$b,
                     include_z = this_parameter$include_z > 0,
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
                                   this_parameter$e_even))
    
    dat <- mod$gen(this_parameter$n)
    
    signal_strength <- log(mod$sample_effect_size(dat) + 1)
    
    if (y_check_train(signal_strength)) {
      current_num_plots <- current_num_plots + 1
      save_plot_data(dat, signal_strength, "train", this_parameter)
    }
    
    if (current_num_plots >= SAMPLE_PER_PARAMETER$train) break
  }
  
  print(table(GLOBAL_Y_TRAIN >= BINS))
  
  current_num_plots <- 0
  
  while (TRUE) {
    this_parameter <- map(phn_parameter_range, ~.x())
    
    mod <- phn_model(j = this_parameter$j,
                     a = this_parameter$a,
                     b = (this_parameter$include_heter > 0) * this_parameter$b,
                     include_z = this_parameter$include_z > 0,
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
                                   this_parameter$e_even))
    
    dat <- mod$gen(this_parameter$n)
    
    signal_strength <- log(mod$sample_effect_size(dat) + 1)
    
    if (y_check_test(signal_strength)) {
      current_num_plots <- current_num_plots + 1
      save_plot_data(dat, signal_strength, "test", this_parameter)
    }
    
    if (current_num_plots >= SAMPLE_PER_PARAMETER$test) break
  }
  
  print(table(GLOBAL_Y_TEST >= BINS))
}


# save a copy -------------------------------------------------------------

if (!dir.exists(here(glue("{DATA_FOLDER}")))) dir.create(here(glue("{DATA_FOLDER}")), recursive = TRUE)
RAW_DATA %>% saveRDS(file = here(glue("{DATA_FOLDER}/raw.rds")))
write_csv(PLOT_META, here(glue("{DATA_FOLDER}/meta.csv")))
