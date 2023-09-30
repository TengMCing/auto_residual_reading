
# Setup the R environment
default_repo <- "https://cloud.r-project.org"
if (!requireNamespace("haven", quietly = TRUE)) install.packages("haven", repos = default_repo)
if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse", repos = default_repo)
if (!requireNamespace("here", quietly = TRUE)) install.packages("here", repos = default_repo)
if (!requireNamespace("glue", quietly = TRUE)) install.packages("glue", repos = default_repo)
if (!requireNamespace("progress", quietly = TRUE)) install.packages("progress", repos = default_repo)
if (!requireNamespace("doMC", quietly = TRUE)) install.packages("doMC", repos = default_repo)
if (!requireNamespace("visage", quietly = TRUE)) {
  remotes::install_github("TengMCing/bandicoot")
  remotes::install_github("TengMCing/visage")
}

# User also needs to ensure Tensorflow, Keras, PIL, etc. 
# are installed for the conda environment

# Setup folder structure

library(here)
library(glue)
proj_dir <- here()

create_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
}

create_dir(glue("{proj_dir}/data"))
create_dir(glue("{proj_dir}/scripts/data_preparation"))
create_dir(glue("{proj_dir}/scripts/training"))
create_dir(glue("{proj_dir}/scripts/evaluation"))
create_dir(glue("{proj_dir}/scripts/slurm"))
create_dir(glue("{proj_dir}/keras_tuner/best_logs"))
create_dir(glue("{proj_dir}/keras_tuner/best_models"))
create_dir(glue("{proj_dir}/keras_tuner/logs"))
create_dir(glue("{proj_dir}/keras_tuner/tuner"))
