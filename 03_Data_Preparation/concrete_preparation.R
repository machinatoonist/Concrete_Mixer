# CONCRETE UNDERSTANDING ----
# Libraries

set.seed(1)

library(tidyverse)
library(tidyquant)
library(readxl)
library(stringr)
library(skimr)
library(GGally)
library(caret)
library(recipes)
library(stats)

# Load Data

path_train <- "00_Data/Concrete_Data.xls"
train_raw_tbl <- read_excel(path_train, sheet = 1)

glimpse(train_raw_tbl)

names(train_raw_tbl) <- c("Cement_kg", 
                          "Slag_kg", 
                          "Fly_Ash_kg", 
                          "Water_kg", 
                          "Superplasticizer_kg",
                          "Coarse_Aggregate_kg",
                          "Fine_Aggregate_kg",
                          "Age_day",
                          "Compressive_Strength_MPa")


# Uncover Problems and Opportunities ----

calculate_concrete_cost <- function(
  
  # Direct Costs	
  cement_cost = 500,
  slag_cost = 100,
  fly_ash_cost = 400,
  plasticizer_cost = 3500,
  coarse_agg_cost = 100,
  fine_agg_cost = 150
  
) {
  
  # Direct Costs
  direct_cost <- sum(cement_cost, slag_cost, fly_ash_cost, 
                      plasticizer_cost, coarse_agg_cost, fine_agg_cost)
  
  return(direct_cost)

}

calculate_concrete_cost(cement_cost = 200)



# Workflow  ----

concrete_tbl <- train_raw_tbl %>%
  mutate(
    cost = calculate_concrete_cost() 
  )

  
# Visualisation of Data  ----

skim(train_raw_tbl)

# train_raw_tbl %>%
#   mutate(
#     cost = calculate_concrete_cost()
#     )
#   
train_raw_tbl %>%
  select_if(is.numeric) %>%
  map(~ unique(.) %>% length())

# Determine if any numeric data needs to be considered as factors

train_raw_tbl %>%
  select_if(is.numeric) %>%
  map_df(~ unique(.) %>% length()) %>%
  gather() %>%
  arrange(value)

train_raw_tbl %>%
  select(Compressive_Strength_MPa, everything()) %>%
  ggpairs()

plot_ggpairs <- function(data, 
                         colour = NULL,
                         densityAlpha = 0.5){
  colour_expr <- enquo(colour)
  
  if(rlang::quo_is_null(colour_expr)){
    
    g <- data %>%
      ggpairs(lower = "blank")
    
  } else {
    
    colour_name <- quo_name(colour_expr)
    g <- data %>%
      ggpairs(mapping = aes_string(colour = colour_name),
              lower = "blank", legend = 1,
              diag = list(continuous = wrap("densityDiag", 
                                            alpha = densityAlpha))) +
      theme(legend.position = "bottom")
  }
  
  return(g)
  
}

train_raw_tbl %>%
  select(Compressive_Strength_MPa, everything()) %>%
  plot_ggpairs()


# Processing Pipeline

train_indices <- createDataPartition(y = train_raw_tbl[["Compressive_Strength_MPa"]],
                                     p = 0.7,
                                     list = FALSE)

train_data <- train_raw_tbl[train_indices,]

test_data <- train_raw_tbl[-train_indices,]

train_control <- trainControl(method = "none")


data <- train_raw_tbl

plot_hist_facet <- function(data,
                            bins = 10,
                            ncol = 5,
                            fct_reorder = FALSE,
                            fct_rev = FALSE,
                            fill = palette_light()[[3]],
                            color = "white",
                            scale = "free"){
  
  data_factored <- data %>%
    mutate_if(is.character, as.factor) %>%
    mutate_if(is.factor, as.numeric) %>%
    gather(key = key, value = value, factor_key = TRUE)
  
  if(fct_reorder) {
    data_factored <- data_factored %>%
      mutate(key = as.character(key) %>% as.factor())
    
  }
  
  if(fct_rev) {
    data_factored <- data_factored %>%
      mutate(key = fct_rev(key))
  }
  
  g <- data_factored %>%
    ggplot(aes(x = value, group = key)) +
    geom_histogram(bins = bins, fill = fill, color = color) +
    facet_wrap(~ key, ncol = ncol, scale = scale) +
    theme_tq() +
    labs(title = "Modeling Concrete Compressive Strength",
         subtitle = "Drivers")
  
  return(g)
  
}

train_raw_tbl %>%
  relocate(Compressive_Strength_MPa) %>%
  plot_hist_facet(bins = 10, ncol = 3)

# Data preprocessing with recipes ----

# Plan
# 1. Impute for missing data or remove features with zero variance
# 2. Transformations
# 3. Discretise - binning can hurt correlation analysis and should be avoided for regression analysis
# 4. Dummy Variables
# 5. Interaction variable
# 6. Normalisation
# 7. Multivariate Transformations

# Plan: Correlation Analysis

# 1. Zero variance features ----
# Using recipe() from the purrr package

train_raw_tbl

recipe_obj <- recipe(Compressive_Strength_MPa ~ ., data = train_raw_tbl) %>%
  step_zv(all_predictors())

recipe_obj

prep_train_raw_tbl <- recipe_obj %>%
  prep() %>%
  bake(new_data = train_raw_tbl) %>%
  select(where(is.numeric)) %>%
  plot_hist_facet(bins = 10, ncol = 3)  


# 2. Transformations ----

skewed_feature_names <- train_raw_tbl %>%
  select_if(is.numeric) %>%
  map_df(skewness) %>%
  gather(factor_key = T) %>%
  arrange(desc(value)) %>%
  filter(value >= 0.8) %>%
  # filter(!key %in% c("Age", "Superplasticizer")) %>%
  pull(key) %>%
  as.character()

train_raw_tbl %>%
  select(all_of(skewed_feature_names)) %>%
  plot_hist_facet()

recipe_obj <- recipe(Compressive_Strength_MPa ~ ., data = train_raw_tbl) %>%
  step_zv(all_predictors()) %>%
  step_YeoJohnson(skewed_feature_names) %>%
  # step_mutate_at(factor_names, fn = as.factor)
  step_center(all_numeric()) %>%
  step_scale(all_numeric())

prepared_recipe <- recipe_obj %>% 
  prep() 

#Before prep()
recipe_obj$steps[[4]]
# After prep()
prepared_recipe$steps[[4]]

prepared_recipe %>%
  bake(new_data = train_raw_tbl) %>%
  select_if(is.numeric) %>%
  plot_hist_facet()

## Final Recipe ----

recipe_obj <- recipe(Compressive_Strength_MPa ~ ., data = train_raw_tbl) %>%
  step_zv(all_predictors()) %>%
  step_YeoJohnson(skewed_feature_names) %>%
  # step_mutate_at(factor_names, fn = as.factor)
  step_center(all_numeric()) %>%
  step_scale(all_numeric()) %>%
  prep()

recipe_obj

train_tbl_bake <- bake(recipe_obj, train_data)

train_tbl_bake %>% glimpse()

test_tbl_bake <- bake(recipe_obj, test_data)

# Correlation Analysis ----

data <- train_tbl_bake
glimpse(train_tbl_bake)

# Quoting an expression outside of a function you use quo().  Inside a function use enquo()
feature_expr <- quo(Compressive_Strength_MPa)

get_cor = function(data,
                   target,
                   use = "pairwise.complete.obs",
                   fct_reorder = FALSE,
                   fct_rev = FALSE) {
  
  feature_expr <- enquo(target)
  feature_name <- quo_name(feature_expr)  # takes quoted expression and converts into text
  
  data_cor <- data %>%
    mutate_if(is.character, as_factor) %>%
    mutate_if(is.factor, as.numeric) %>%
    cor(use = use) %>%
    as_tibble() %>% 
    mutate(feature = names(.)) %>% # the dot (.) passes the data frame into the expression
    select(feature, !! feature_expr) %>% 
    filter(!(feature == feature_name)) %>%
    mutate_if(is.character, as_factor)
  
  # Adjust levels by the magnitude of the feature_expr correlation coefficient - Attrition
  if(fct_reorder) {
    data_cor <- data_cor %>%
      mutate(feature = fct_reorder(feature, !! feature_expr)) %>%
      arrange(feature)
    
  }
  
  if(fct_rev) {
    data_cor <- data_cor %>%
      mutate(feature = fct_rev(feature)) %>%
      arrange(feature)
  }
  
  return(data_cor)
  
}

train_tbl_bake %>%
  get_cor(Compressive_Strength_MPa,
          use = "pairwise.complete.obs",
          fct_reorder = T,
          fct_rev = T)

data <- train_tbl_bake
glimpse(train_tbl_bake)
feature_expr <- quo(Compressive_Strength)

plot_cor <- function (data,
                      target,
                      fct_reorder = FALSE,
                      fct_rev = FALSE,
                      include_lbl = TRUE,
                      lbl_precision = 2,
                      lbl_position = "outward",
                      size = 2,
                      line_size = 1,
                      vert_size = 1,
                      color_pos = palette_light()[[1]],
                      color_neg = palette_light()[[2]]
) {
  feature_expr <- enquo(target)
  feature_name <- quo_name(feature_expr)  # takes quoted expression and converts into text
  
  data_cor <- data %>%
    get_cor(!! feature_expr, fct_reorder = fct_reorder, fct_rev = fct_rev) %>%
    mutate(feature_name_text = round(!! feature_expr, lbl_precision)) %>%
    mutate(Correlation = case_when(
      !! feature_expr >= 0 ~ "Positive",
      TRUE ~ "Negative") %>% as_factor())
  
  g <- data_cor %>%
    ggplot(aes_string(x = feature_name, y = "feature", group = "feature")) +
    geom_point(aes(color = Correlation), size = size) +
    geom_segment(aes(xend = 0, yend = feature, color = Correlation), size = line_size) +
    geom_vline(xintercept = 0, color = palette_light()[[1]], size = vert_size) +
    expand_limits(x = c(-1, 1)) +
    theme_tq() + 
    scale_color_manual(values = c(color_neg, color_pos))
  
  if(include_lbl) g <- g + geom_label(aes(label = feature_name_text), hjust = lbl_position)
  
  return(g)
}

train_tbl_bake %>%
  plot_cor(Compressive_Strength_MPa, fct_reorder = TRUE, fct_rev = FALSE)
