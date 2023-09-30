# This script generates all the residual plots used in the human subject experiment

library(visage)
library(here)
library(glue)
library(tidyverse)

vi_lineup <- get_vi_lineup()
proj_dir <- here()

map(vi_lineup, function(lineup) {
  for (i in 1:20) {
    this_plot <- lineup$data %>%
      filter(k == i) %>%
      VI_MODEL$plot(theme = theme_light(base_size = 11/5), 
                    remove_axis = TRUE, 
                    remove_legend = TRUE, 
                    remove_grid_line = TRUE)
    ggsave(glue("{proj_dir}/data/source/experiment/residual_plots/{lineup$metadata$name}_{i}.png"), 
          this_plot, 
          width = 7/5, 
          height = 7/4)
  }
  
})


# low_res -----------------------------------------------------------------

PIL <- reticulate::import("PIL")

create_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
}

for (filename in list.files(glue("{proj_dir}/data/source/experiment/residual_plots"))) {
  im <- PIL$Image$open(glue("{proj_dir}/data/source/experiment/residual_plots/{filename}"))
  for (res in c(32L, 64L, 128L, 256L)) {
    new_im <- im$resize(c(res, res))
    create_dir(glue("{proj_dir}/data/{res}/experiment/residual_plots"))
    new_im$save(glue("{proj_dir}/data/{res}/experiment/residual_plots/{filename}"))
  }
  im$close()
}

