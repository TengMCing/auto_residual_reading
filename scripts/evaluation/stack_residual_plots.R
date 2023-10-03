library(visage)


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

mod_ <- poly_model(shape = 1, x = get_x_var("uniform"), sigma = 0.5)
map(1:500, ~mod_$gen(300, test = FALSE) %>% 
      mutate(.resid = stand_dist(.resid), 
             .fitted = stand_dist(.fitted)) %>%
      select(.resid, .fitted)) %>%
  reduce(bind_rows) %>%
  ggplot() +
  geom_point(aes(.fitted, .resid), alpha = 0.1)

mod_ <- poly_model(shape = 1, x = get_x_var("uniform"), sigma = 1)
map(1:500, ~mod_$gen(300, test = FALSE) %>% 
      mutate(.resid = stand_dist(.resid), 
             .fitted = stand_dist(.fitted)) %>%
      select(.resid, .fitted)) %>%
  reduce(bind_rows) %>%
  ggplot() +
  geom_point(aes(.fitted, .resid), alpha = 0.1)

mod_ <- poly_model(shape = 1, x = get_x_var("uniform"), sigma = 2)
map(1:500, ~mod_$gen(300, test = FALSE) %>% 
      mutate(.resid = stand_dist(.resid), 
             .fitted = stand_dist(.fitted)) %>%
      select(.resid, .fitted)) %>%
  reduce(bind_rows) %>%
  ggplot() +
  geom_point(aes(.fitted, .resid), alpha = 0.1)

mod_ <- poly_model(shape = 1, x = get_x_var("uniform"), sigma = 4)
map(1:500, ~mod_$gen(300, test = FALSE) %>% 
      mutate(.resid = stand_dist(.resid), 
             .fitted = stand_dist(.fitted)) %>%
      select(.resid, .fitted)) %>%
  reduce(bind_rows) %>%
  ggplot() +
  geom_point(aes(.fitted, .resid), alpha = 0.1)

mod_ <- poly_model(shape = 2, x = get_x_var("uniform"), sigma = 0.5)
map(1:500, ~mod_$gen(300, test = FALSE) %>% 
      mutate(.resid = stand_dist(.resid), 
             .fitted = stand_dist(.fitted)) %>%
      select(.resid, .fitted)) %>%
  reduce(bind_rows) %>%
  ggplot() +
  geom_point(aes(.fitted, .resid), alpha = 0.1)

mod_ <- poly_model(shape = 3, x = get_x_var("uniform"), sigma = 0.5)
map(1:500, ~mod_$gen(300, test = FALSE) %>% 
      mutate(.resid = stand_dist(.resid), 
             .fitted = stand_dist(.fitted)) %>%
      select(.resid, .fitted)) %>%
  reduce(bind_rows) %>%
  ggplot() +
  geom_point(aes(.fitted, .resid), alpha = 0.1)

mod_ <- poly_model(shape = 4, x = get_x_var("uniform"), sigma = 0.5)
map(1:500, ~mod_$gen(300, test = FALSE) %>% 
      mutate(.resid = stand_dist(.resid), 
             .fitted = stand_dist(.fitted)) %>%
      select(.resid, .fitted)) %>%
  reduce(bind_rows) %>%
  ggplot() +
  geom_point(aes(.fitted, .resid), alpha = 0.1)
