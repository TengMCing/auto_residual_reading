
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
DATA_FOLDER <- "data/phn_v2/residual_plots"


# y_check -----------------------------------------------------------------

GLOBAL_Y_TRAIN <- c()
GLOBAL_Y_TEST <- c()
NUM_BIN <- 50
MAX_Y <- 7

y_check_train <- function(new_y) {
  target_bin <- floor(min(new_y, MAX_Y) / MAX_Y * (NUM_BIN - 1)) + 1
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
  target_bin <- floor(min(new_y, MAX_Y) / MAX_Y * (NUM_BIN - 1)) + 1
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
  
  table(map_dbl(GLOBAL_Y_TRAIN, ~floor(min(.x, MAX_Y) / MAX_Y * (NUM_BIN - 1)) + 1)) %>%
    print()
  
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
  
  table(map_dbl(GLOBAL_Y_TEST, ~floor(min(.x, MAX_Y) / MAX_Y * (NUM_BIN - 1)) + 1)) %>%
    print()
}


# save a copy -------------------------------------------------------------

if (!dir.exists(here(glue("{DATA_FOLDER}")))) dir.create(here(glue("{DATA_FOLDER}")), recursive = TRUE)
RAW_DATA %>% saveRDS(file = here(glue("{DATA_FOLDER}/raw.rds")))
write_csv(PLOT_META, here(glue("{DATA_FOLDER}/meta.csv")))


# draw all plots ----------------------------------------------------------

for (i in 1:length(RAW_DATA)) {
  dat <- RAW_DATA[[i]]
  data_type <- PLOT_META$data_type[i]
  
  VI_MODEL$plot(dat,
                theme = theme_light(base_size = 11/5),
                remove_axis = TRUE,
                remove_legend = TRUE,
                remove_grid_line = TRUE) -> this_plot
  
  # The lineup layout contains 4 rows and 5 cols
  ggsave(glue(here("{DATA_FOLDER}/native/{data_type}/0/{i}.png")),
         this_plot,
         width = 7/5,
         height = 7/4)
}


# compute scagnostics -----------------------------------------------------

RAW_DATA <- readRDS(here(glue("{DATA_FOLDER}/raw.rds")))
PLOT_META <- read_csv(here(glue("{DATA_FOLDER}/meta.csv")))

try_or_zero <- function(fn, ...) {
  print(as.character(substitute(fn)))
  try_result <- try(fn(...), silent = TRUE)
  if (inherits(try_result, "try-error")) return(0)
  return(try_result)
}

PLOT_META$measure_monotonic <- 0
PLOT_META$measure_sparse <- 0
PLOT_META$measure_splines <- 0
PLOT_META$measure_striped <- 0

for (i in 70000:length(RAW_DATA)) {
  print(i)
  dat <- RAW_DATA[[i]]
  PLOT_META$measure_monotonic[i] <- try_or_zero(sc_monotonic, dat$.fitted, dat$.resid) %>% 
    {ifelse(is.na(.), 0, .)}
  
  if (i == 3180 || i == 11890) {
    PLOT_META$measure_sparse[i] <- 0
  } else {
    PLOT_META$measure_sparse[i] <- try_or_zero(sc_sparse2, dat$.fitted, dat$.resid) %>% 
      {ifelse(is.na(.), 0, .)}
  }
  
  if (i == 11890) {
    PLOT_META$measure_splines[i] <- 0
  } else {
    PLOT_META$measure_splines[i] <- try_or_zero(sc_splines, dat$.fitted, dat$.resid) %>% 
      {ifelse(is.na(.), 0, .)}
  }
  
  if (i == 8070 || i == 11889 || i == 11890 || i == 70059) {
    PLOT_META$measure_striped[i] <- 0
  } else {
    PLOT_META$measure_striped[i] <- try_or_zero(sc_striped, dat$.fitted, dat$.resid) %>% 
      {ifelse(is.na(.), 0, .)}
  }
  
}


# save_meta_data ----------------------------------------------------------

write_csv(PLOT_META, here(glue("{DATA_FOLDER}/meta.csv")))

# mixed_low_res -----------------------------------------------------------

PIL <- reticulate::import("PIL")

create_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
}

for (data_type in c("train", "test")) {
  for (filename in list.files(here(glue("{DATA_FOLDER}/native/{data_type}/0")))) {
    im <- PIL$Image$open(glue("{DATA_FOLDER}/native/{data_type}/0/{filename}"))
    for (res in c(32L, 64L, 128L, 256L)) {
      new_im <- im$resize(c(res, res))
      create_dir(glue("{DATA_FOLDER}/{res}/{data_type}/0"))
      new_im$save(glue("{DATA_FOLDER}/{res}/{data_type}/0/{filename}"))
    }
    im$close()
  }
}
