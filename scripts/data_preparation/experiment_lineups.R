# This script generates all the residual plots used in the human subject experiment

library(visage)
library(here)
library(glue)
library(tidyverse)

vi_lineup <- get_vi_lineup()
proj_dir <- here()

map(vi_lineup, function(lineup) {
    this_plot <- lineup$data %>%
      VI_MODEL$plot_lineup(theme = theme_light(), 
                           remove_axis = TRUE, 
                           remove_legend = TRUE, 
                           remove_grid_line = TRUE)
    ggsave(glue("{proj_dir}/data/source/experiment/lineups/{lineup$metadata$name}.png"), 
           this_plot, 
           width = 7, 
           height = 7)
})


# low_res -----------------------------------------------------------------

PIL <- reticulate::import("PIL")

create_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
}

for (filename in list.files(glue("{proj_dir}/data/source/experiment/lineups"))) {
  im <- PIL$Image$open(glue("{proj_dir}/data/source/experiment/lineups/{filename}"))
  for (res in c(32L, 64L, 128L, 256L)) {
    new_im <- im$resize(c(res, res))
    create_dir(glue("{proj_dir}/data/{res}/experiment/lineups"))
    new_im$save(glue("{proj_dir}/data/{res}/experiment/lineups/{filename}"))
  }
  im$close()
}

