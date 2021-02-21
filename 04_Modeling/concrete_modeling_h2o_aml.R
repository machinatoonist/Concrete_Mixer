# H2O MODELING ----

# Install H2O ----
# First install a version of Java that is compatible with H2O

# if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
# if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
# 
# pkgs <- c("RCurl","jsonlite")
# for (pkg in pkgs) {
#   if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
# }
# 
# install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")))

# h2o.init()
# demo(h2o.kmeans)

# 1. Setup ----
# Libraries
library(h2o)
library(recipes)
library(readxl)
library(tidyverse)
library(stringr)
library(tidyquant)
library(forcats)
library(cowplot)
library(fs)
library(glue)
library(recipes)
library(caret)
h2o.init()

wd = getwd()
# Load data
path_train <- "00_Data/Concrete_Data.xls"
raw_tbl <- read_excel(path_train, sheet = 1)

# Processing Pipeline
source("00_Scripts/data_processing_pipeline.R")

readable_tbl <- process_data_readable(raw_tbl)
train_indices <- createDataPartition(y = readable_tbl[["Compressive_Strength_MPa"]],
                                     p = 0.7,
                                     list = FALSE)
train_data_tbl <- readable_tbl[train_indices,]

test_data_tbl <- readable_tbl[-train_indices,]

# ML Preprocessing

# Commenting out skewed factor references that should not be required for h2o automl
# skewed_feature_names <- train_data_tbl %>%
#   select_if(is.numeric) %>%
#   map_df(skewness) %>%
#   gather(factor_key = T) %>%
#   arrange(desc(value)) %>%
#   filter(value >= 0.8) %>%
#   # filter(!key %in% c("Age", "Superplasticizer")) %>%
#   pull(key) %>%
#   as.character()

# Commenting out the data transformation steps because h2o should take care of these steps for us
recipe_obj <- recipe(Compressive_Strength_MPa ~ ., data = train_data_tbl) %>%
  step_zv(all_predictors()) %>%
  # step_YeoJohnson(all_of(skewed_feature_names)) %>%
  # # step_mutate_at(factor_names, fn = as.factor)
  # step_center(all_numeric()) %>%
  # step_scale(all_numeric()) %>%
  prep()

  
train_tbl <- bake(recipe_obj, new_data = train_data_tbl)
test_tbl <- bake(recipe_obj, new_data = test_data_tbl)
train_tbl %>% glimpse()

# 2. Modeling ----

h2o.init()

# Not using h2o leaderboard
# split_h2o <- h2o.splitFrame(as.h2o(train_tbl), ratios = c(0.85), seed = 321)  

# train_h2o <- split_h2o[[1]]  
# valid_h2o <- split_h2o[[2]]  
# test_h2o <- as.h2o(test_tbl)

train_h2o <- as.h2o(train_tbl)
valid_h2o <- as.h2o(test_tbl)

y <- "Compressive_Strength_MPa"
x <- setdiff(names(train_h2o), y) # Get the difference between two arguments

# To get the value for lambda use the function: tidy(recipe_obj, number = step_number).  
# You can also get it by pulling out the steps element since your recipe object is a list 
# (e.g. recipe_obj$steps)

automl_models_h2o <- h2o.automl(
    x = x,
    y = y,
    training_frame = train_h2o,
    validation_frame = valid_h2o,
    max_runtime_secs = 60,
    nfolds = 5
)

# Inspect the h2o results object

typeof(automl_models_h2o)

slotNames(automl_models_h2o)

automl_models_h2o@leaderboard

automl_models_h2o@leader

# h2o.getModel("GLM_1_AutoML_20210101_073531")
# 
# h2o.getModel("StackedEnsemble_BestOfFamily_AutoML_20210101_073531")
# 
# h2o.getModel("DeepLearning_grid__2_AutoML_20210101_073531_model_1")

automl_models_h2o@leaderboard %>%
  as_tibble() %>% 
  slice(1:6) %>%
  pull(model_id) %>%
  h2o.getModel()

extract_model_name_by_position <- function (h2o_leaderboard, n = 1, verbose = TRUE){
  
  model_name <- h2o_leaderboard %>%
  as_tibble() %>% 
  slice(n) %>%
  pull(model_id) 
  
  if(verbose) message(model_name)
  
  return(model_name)
}

automl_models_h2o@leaderboard %>% 
  extract_model_name_by_position(1) %>%
  h2o.getModel() %>%
  h2o.saveModel(path = glue("{wd}/04_Modeling/h2o_models/"))

# Save Models
for(i in 1:6) {
  automl_models_h2o@leaderboard %>% 
    extract_model_name_by_position(i) %>%
    h2o.getModel() %>%
    h2o.saveModel(path = glue("{wd}/04_Modeling/h2o_models/"))
}

# Load Models
for(i in 1) {
  h2o.loadModel(glue("{wd}/04_Modeling/h2o_models/{
                      automl_models_h2o@leaderboard %>% 
                      extract_model_name_by_position(i)
                      }")
                )
}

# This is an alternative approach to loading a single model
ensemble_model <- h2o.loadModel(path = glue("{wd}/04_Modeling/h2o_models/StackedEnsemble_AllModels_AutoML_20210102_060236"))


# Making Predictions

ensemble_model

predictions <- h2o.predict(ensemble_model,  newdata = as.h2o(test_tbl))

predictions_tbl <- predictions %>% as_tibble()

# Get parameters of model
ensemble_model@allparameters

# No AUC for regression model
# h2o.auc(ensemble_model, train = T, valid = T, xval = T)

h2o.mae(ensemble_model, train = T, valid = T, xval = T)

# 3. Visualising the leaderboard ----

data_transformed <- automl_models_h2o@leaderboard %>%
  as_tibble() %>%
  mutate(model_type = str_split(string = model_id, pattern = "_", simplify = T)[,1]) %>%
  slice(1:5) %>%
  rownames_to_column() %>%
  mutate(model_id = as_factor(model_id) %>% reorder(mae),
         model_type = as_factor(model_type)) %>%
  pivot_longer(cols = c(mean_residual_deviance:rmsle), names_to = "key", values_to = "value") %>%
  mutate(key = as_factor(key),
         model_id = paste0(rowname,". ", model_id) %>% as_factor() %>% fct_rev())

data_transformed %>%
  ggplot(aes(x = value, y = model_id , color = model_type)) +
  geom_point(size = 3) + 
  geom_label(aes(label = round(value, 2), hjust = "inward")) +
  facet_wrap(~ key, scales = "free_x") +
  theme_tq() +
  scale_color_tq() + 
  labs(title = "H2O Leaderboard Visualisation",
      subtitle = "Ordered by: mae",
      y = "Model Position. Model ID",
      x = ""
      )

# Create object for stepping through model
h2o_leaderboard <- automl_models_h2o@leaderboard

plot_h2o_leaderboard <- function(h2o_leaderboard, order_by = c("mae", "rmse"), 
                                 n_max = 20, size = 4, include_lbl = TRUE) {
  
  # Setup inputs
  order_by <- tolower(order_by[[1]])
  
  leaderboard_tbl <- h2o_leaderboard %>%
    as_tibble() %>%
    mutate(model_type = str_split(model_id, "_", simplify = T)[,1]) %>%
    rownames_to_column(var = "rowname") %>%
    mutate(model_id = paste0(rowname, ". ", as.character(model_id)) %>% as_factor())
  
  # Transformation
  if (order_by == "mae") {
    
    data_transformed_tbl <- leaderboard_tbl %>%
      slice(1:n_max) %>%
      mutate(
        model_id   = as_factor(model_id) %>% reorder(mae) %>% fct_rev(),
        model_type = as_factor(model_type)
      ) %>%
      # gather(key = key, value = value, 
      #        -c(model_id, model_type, rowname), factor_key = T) 
    pivot_longer(cols = c(mean_residual_deviance:rmsle), names_to = "key", values_to = "value") %>%
      mutate(key = as_factor(key)) %>% arrange(key)
    
  } else if (order_by == "rmse") {
    
    data_transformed_tbl <- leaderboard_tbl %>%
      slice(1:n_max) %>%
      mutate(
        model_id   = as_factor(model_id) %>% reorder(rmse) %>% fct_rev(),
        model_type = as_factor(model_type)
      ) %>%
      # gather(key = key, value = value, 
      #        -c(model_id, model_type, rowname), factor_key = T) 
      pivot_longer(cols = c(auc:mse), names_to = "key", values_to = "value") %>%
      mutate(key = as_factor(key)) %>% arrange(key)
    
  } else {
    stop(paste0("order_by = '", order_by, "' is not a permitted option."))
  }
  
  # Visualization
  g <- data_transformed_tbl %>%
    ggplot(aes(value, model_id, color = model_type)) +
    geom_point(size = size) +
    facet_wrap(~ key, scales = "free_x") +
    theme_tq() +
    scale_color_tq() +
    labs(title = "Concrete Machine Learning Model Leaderboard Metrics",
         subtitle = paste0("Ordered by: ", toupper(order_by)),
         y = "Model Postion, Model ID", x = "")
  
  if (include_lbl) g <- g + geom_label(aes(label = round(value, 2), hjust = "inward"))
  
  return(g)
  
}
automl_models_h2o@leaderboard %>% 
  plot_h2o_leaderboard(order_by = "mae", n_max = 8)

# CODE BELOW IS NOT APPLICABLE FOR ASSESSING REGRESSION MODEL PERFORMANCE

# 4. Assessing Performance ----

# Grid Search & CV

xgboost_model <- h2o.loadModel("04_Modeling/h2o_models/XGBoost_grid__1_AutoML_20201214_054322_model_3")
xgboost_model

glm_model <- h2o.loadModel("04_Modeling/h2o_models/GLM_1_AutoML_20210101_073531")
glm_model

test_tbl

h2o.performance(xgboost_model, newdata = as.h2o(test_tbl))

h2o.performance(glm_model, newdata = as.h2o(test_tbl))

?h2o.grid
?h2o.glm
glm_model@allparameters

glm_grid_01 <- h2o.grid(
      algorithm = "glm",
      grid_id = "glm_grid_01",
      
      # h2o.glm()
      x = x,
      y = y,
      training_frame = train_h2o,
      validation_frame = valid_h2o,
      nfolds = 5,
      
      hyper_params = list(
        alpha = list(c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), c(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                     c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), c(0.6, 0.5, 0.4, 0.3, 0.2, 0.0),
                     c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5), c(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        stopping_metric = c("AUC", "logloss")
      )
  
  )

glm_grid_01

h2o.getGrid("glm_grid_01", sort_by = "auc", decreasing = TRUE)

glm_grid_01_model_7 <- h2o.getModel("glm_grid_01_model_7")

glm_grid_01_model_7 %>% h2o.auc(train = T, valid = T, xval = T)

glm_grid_01_model_7 %>%
  h2o.performance(newdata = as.h2o(test_tbl))

glm_grid_01_model_7 %>%
h2o.saveModel(path = "04_Modeling/h2o_models/")

# Load Models
xgboost_model <-  h2o.loadModel(path = "04_Modeling/h2o_models/XGBoost_grid__1_AutoML_20201214_054322_model_1")
glm_model <- h2o.loadModel(path = "04_Modeling/h2o_models/glm_grid_01_model_7")
ensemble_model <- h2o.loadModel(path = "04_Modeling/h2o_models/StackedEnsemble_BestOfFamily_AutoML_20201214_054322")

performance_h2o <- h2o.performance(glm_model, newdata = as.h2o(test_tbl))
performance_h2o

typeof(performance_h2o)
performance_h2o %>% slotNames()
performance_h2o@algorithm
performance_h2o@metrics

# Classifier Summary Metrics
h2o.auc(performance_h2o)
h2o.auc(glm_model, train = T, valid = T, xval = T)
h2o.giniCoef(performance_h2o)
h2o.logloss(performance_h2o)

# Performance on training data
h2o.confusionMatrix(glm_model)
# Evaluate performance on the hold out set
h2o.confusionMatrix(performance_h2o)

# Understand how confusion matrix changes as a function of threshold
# f1: typically the threshold that maximises f1 is used by this is not necessarily the best
# An expected value optimisation is recommended when costs of false positives and false negatives are known

# Precision vs Recall Plot

performance_tbl <- performance_h2o %>%
  h2o.metric() %>%
  as_tibble() 

performance_tbl %>%
  arrange(desc(f1)) %>%
  glimpse()


performance_tbl %>%
  ggplot(aes(x = threshold)) +
  geom_line(aes(y = precision, color = "orange")) +
  geom_line(aes(y = recall, color = "blue")) +
  geom_vline(xintercept = h2o.find_threshold_by_max_metric(performance_h2o, "f1")) + 
  theme_tq() +
  labs(
    title = "Precision vs Recall",
    y = "value"
  )

# ROC Plot (receiver operating characteristic curve) is a graph showing the performance of a 
# classification model at all classification thresholds. 
# This curve plots two parameters: True Positive Rate. False Positive Rate.
# A perfect model has an AUC of 1.  A model with no predictive power has an AUC of 0.5.

path <- "04_Modeling/h2o_models/glm_grid_01_model_7"

load_model_performance_metrics <- function (path, test_tbl) {
  
  model_h2o <- h2o.loadModel(path)
  perf_h2o <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl))
  
  perf_h2o %>%
    h2o.metric() %>% 
    as_tibble() %>%
    mutate(auc = h2o.auc(perf_h2o)) %>%
    select(tpr, fpr, auc)
}

load_model_performance_metrics(path, test_tbl) 

# Get the path to each model and load performance metrics
model_metrics_tbl <- fs::dir_info(path = "04_Modeling/h2o_models/") %>%
  select(path) %>%
  mutate(metrics = map(path, load_model_performance_metrics, test_tbl)) %>%
  unnest(cols = c(metrics))


model_metrics_tbl <- model_metrics_tbl %>%
  mutate(
    path = str_split(string = path, pattern = "/",simplify = T)[,3] %>% as_factor(),
    auc = auc %>% round(3) %>% as.character() %>% as_factor()) 

model_metrics_tbl %>%
  ggplot(aes(x = fpr, y = tpr, color = path, linetype = auc)) +
  geom_line(size = 0.5) +
  theme_tq() +
  scale_color_tq() +
  theme(legend.direction = "vertical") +
  labs(
      title = "ROC Plot",
      subtitle = "Performance of models"
  )

# Precision vs Recall
# Recall or not missing a positive outcome (false negative) is often more desirable than 
# Precision if the cost of a false positive is not too expensive

load_model_performance_metrics <- function (path, test_tbl) {
  
  model_h2o <- h2o.loadModel(path)
  perf_h2o <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl))
  
  perf_h2o %>%
    h2o.metric() %>% 
    as_tibble() %>%
    mutate(auc = h2o.auc(perf_h2o)) %>%
    select(tpr, fpr, auc, precision, recall)
}

load_model_performance_metrics(path, test_tbl) 

model_metrics_tbl <- fs::dir_info(path = "04_Modeling/h2o_models/") %>%
  select(path) %>%
  mutate(metrics = map(path, load_model_performance_metrics, test_tbl)) %>%
  unnest(cols = c(metrics)) 

model_metrics_tbl %>%
  mutate(
    path = str_split(string = path, pattern = "/",simplify = T)[,3] %>% as_factor(),
    auc = auc %>% round(3) %>% as.character() %>% as_factor()) %>%
  ggplot(aes(x = recall, y = precision, color = path, linetype = auc)) +
  geom_line(size = 0.5) +
  theme_tq() +
  scale_color_tq() +
  theme(legend.direction = "vertical") +
  labs(
    title = "Precision vs Recall Plot",
    subtitle = "Performance of models"
  )

# Gain and Lift
predictions <- h2o.predict(glm_grid_01_model_7,  newdata = as.h2o(test_tbl))
predictions_tbl <- predictions %>% 
  as_tibble() 

ranked_predictions_tbl <- predictions_tbl %>%
  bind_cols(test_tbl) %>%
  select(Attrition, predict:Yes) %>%
  arrange(desc(Yes))

# Calculate overall attrition rate
combined_tbl <- train_readable_tbl %>%
  rbind(test_readable_tbl) %>%
  group_by(Attrition) %>%
  summarize(n = n()) %>%
  ungroup() %>%   
  mutate(pct = n/sum(n)) %>%
  ungroup()

# Explanation of Gain
# If you select 10 employees at random we would only expect to get around 16% to be positive for attrition 
# based on the overall calculated attrition rate.  Using the model, however, we can get much higher.
# Using the model to sort our employees by likelihood for attrition the gain is the measure of how well
# we can predict employees will be to leave the company versus random guessing.
#
# Explanation of Lift
# This is the ratio of people predicted to leave in a model prioritised sample versus the number expected to leave 
# the company based on the average attrition rate.

calculated_gain_lift_tbl <- ranked_predictions_tbl %>%
  mutate(ntile = ntile(Yes, n = 16)) %>%
  group_by(ntile) %>%
  summarize(
    cases = n(),
    responses = sum(Attrition == "Yes")
  ) %>%
  arrange(desc(ntile)) %>%
  mutate(group = row_number()) %>%
  select(-ntile) %>% relocate(group) %>%
  mutate(
    cumulative_responses = cumsum(responses),
    pct_responses = responses/sum(responses),
    gain = cumsum(pct_responses),
    cumulative_pct_cases = cumsum(cases)/sum(cases),
    lift = gain/cumulative_pct_cases,
    gain_baseline = cumulative_pct_cases,
    lift_baseline = gain_baseline/cumulative_pct_cases
  )

# The gain shows how in the first two cohorts of people predicted by the model to be likely to leave
# we are detecting 66.7% of all those who actually leave the company.  This is far greater than the baseline.
# The lift is the improvement over randomly selecting people.
# The gain lift table shows the benefit of having a model to predict those who are likely to leave the company.
# The ntile() function was used to group and sort by probability of attrition

gain_lift_tbl <- glm_grid_01_model_7 %>%
  h2o.performance(newdata = as.h2o(test_tbl)) %>%
  h2o.gainsLift() %>%
  tibble()

glimpse(gain_lift_tbl)

# capture_rate is equivalent to the gain calculated above
# cumulative_lift is equivalent to the lift

gain_transformed_tbl <- gain_lift_tbl %>%
  select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift) %>%
  select(-cumulative_lift) %>%
  mutate(baseline = cumulative_data_fraction) %>%
  rename(gain = cumulative_capture_rate) %>%
  pivot_longer(names_to = "key", values_to = "value", cols = c(gain:baseline)) %>%
  arrange(key)

gain_transformed_tbl %>%
  ggplot(aes(x = cumulative_data_fraction, y = value, color = key)) +
  geom_line(size = 0.5) +
  theme_tq() +
  scale_color_tq() + 
  labs(
    title = "Gain Chart",
    subtitle = "h2o model",
    x = "Cumulative Data Fraction",
    y = "Gain"
  )

# Communicating to stakeholders
# A gain lift chart helps you focus on the highest leverage opportunities for improvement

lift_transformed_tbl <- gain_lift_tbl %>%
  select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift) %>%
  select(-cumulative_capture_rate) %>%
  mutate(baseline = 1) %>%
  rename(lift = cumulative_lift) %>%
  pivot_longer(names_to = "key", values_to = "value", cols = c(lift:baseline)) %>%
  arrange(key)

lift_transformed_tbl %>%
  ggplot(aes(x = cumulative_data_fraction, y = value, color = key)) +
  geom_line(size = 0.5) +
  theme_tq() +
  scale_color_tq(theme = "light") + 
  labs(
    title = "Lift Chart",
    subtitle = "h2o model",
    x = "Cumulative Data Fraction",
    y = "Lift"
  )

# 5. Performance Visualisation ----

h2o_leaderboard <- automl_models_h2o@leaderboard
newdata <- test_tbl
order_by <- "auc"
max_models <- 4
size <- 1.0


plot_h2o_performance <- function(h2o_leaderboard, newdata, order_by = c("auc", "logloss"),
                                 max_models = 3, size = 1.5) {
  
  # Inputs
  leaderboard_tbl <- h2o_leaderboard %>%
    as_tibble() %>%
    slice(1:max_models)
  
  newdata_tbl <- newdata %>%
    as_tibble()
  
  order_by <- tolower(order_by[[1]])
  order_by_expr <- rlang::sym(order_by)
  
  h2o.no_progress()
  
  # 1. Model Metrics
  get_model_performance_metrics <- function (model_id, test_tbl) {
    
    model_h2o <- h2o.getModel(model_id)
    perf_h2o <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl))
    
    perf_h2o %>%
      h2o.metric() %>% 
      as_tibble() %>%
      select(threshold, tpr, fpr, precision, recall)
  }
  
  model_metrics_tbl <- leaderboard_tbl %>%
    mutate(metrics = map(model_id, get_model_performance_metrics, newdata_tbl)) %>%
    unnest(cols = c(metrics)) %>%
    mutate(
      model_id = as_factor(model_id) %>%
        fct_reorder(!! order_by_expr, .desc = ifelse(order_by == "auc", TRUE, FALSE)),
      auc = auc %>%
        round(3) %>%
        as.character() %>%
        as_factor() %>%
        fct_reorder(as.numeric(model_id)),
      logloss = logloss %>%
        round(4) %>%
        as.character() %>%
        as_factor() %>%
        fct_reorder(as.numeric(model_id))
    )
  
  # 1A. ROC Plot - Receiver Operating Characteristics Plot
  p1 <- model_metrics_tbl %>%
    ggplot(aes_string("fpr", "tpr", color = "model_id", linetype = order_by))+
    geom_line(size = size) +
    theme_tq() +
    scale_color_tq() +
    labs(title = "ROC", x = "False Positive Rate", y = "True Positive Rate") +
    theme(legend.direction = "vertical")
  
  # 1B. Precision vs Recall
  p2 <- model_metrics_tbl %>%
    ggplot(aes_string("recall", "precision", color = "model_id", linetype = order_by))+
    geom_line(size = size) +
    theme_tq() +
    scale_color_tq() +
    labs(title = "Precision vs Recall", x = "Recall", y = "Precision") +
    theme(legend.position = "none")
  
  # Gain / Lift
  get_gain_lift <- function(model_id, test_tbl) {
    
    model_h2o <- h2o.getModel(model_id)
    perf_h2o <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl))
    
    perf_h2o %>%
      h2o.gainsLift() %>% 
      as_tibble() %>%
      select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift)
  }
  
  gain_lift_tbl <- leaderboard_tbl %>%
    mutate(metrics = map(model_id, get_gain_lift, newdata_tbl)) %>%
    unnest(cols = c(metrics)) %>%
    mutate(
      model_id = as_factor(model_id) %>%
        fct_reorder(!! order_by_expr, .desc = ifelse(order_by == "auc", TRUE, FALSE)),
      auc = auc %>%
        round(3) %>%
        as.character() %>%
        as_factor() %>%
        fct_reorder(as.numeric(model_id)),
      logloss = logloss %>%
        round(4) %>%
        as.character() %>%
        as_factor() %>%
        fct_reorder(as.numeric(model_id))
    ) %>%
    rename(
      gain = cumulative_capture_rate,
      lift = cumulative_lift
    )
  
  # 2A. Gain Plot
  p3 <- gain_lift_tbl %>%
    ggplot(aes_string("cumulative_data_fraction", "gain", 
                      color = "model_id", linetype = order_by)) +
    geom_line(size = size) + 
    geom_segment(x = 0, y = 0, xend = 1, yend = 1,
                 color = "black", size = size) +
    theme_tq() +
    scale_color_tq() +
    expand_limits(x = c(0,1), y = c(0,1)) + 
    labs(title = "Gain", x = "Cumulative Data Fraction", y = "Gain") +
    theme(legend.position = "none")
  
  # 2B. Lift Plot
  p4 <- gain_lift_tbl %>%
    ggplot(aes_string("cumulative_data_fraction", "lift", 
                      color = "model_id", linetype = order_by)) +
    geom_line(size = size) + 
    geom_segment(x = 0, y = 1, xend = 1, yend = 1,
                 color = "black", size = size) +
    theme_tq() +
    scale_color_tq() +
    scale_y_continuous(labels = scales::number_format(accuracy = 0.01)) + 
    expand_limits(x = c(0,1), y = c(0.5,1)) +
    labs(title = "Lift", x = "Cumulative Data Fraction", y = "Lift") +
    theme(legend.position = "none")
  
  # Combine using cowplot
  p_legend <- get_legend(p1)
  p1 <- p1 +  theme(legend.position = "none")
  
  p <- cowplot::plot_grid(p1, p2, p3, p4, ncol = 2)
  
  p_title <- ggdraw() +
    draw_label("H2O Model Metrics", size = 18, fontface = "bold",
               colour = palette_light()[[1]])
  
  p_subtitle <- ggdraw() +
    draw_label(glue("Ordered by {toupper(order_by)}"), size = 10,
               colour = palette_light()[[1]])
  
  ret <- plot_grid(p_title, p_subtitle, p, p_legend,
                   ncol = 1, rel_heights = c(0.05, 0.05, 1, 0.05 * max_models))
  
  h2o.show_progress()
  
  return(ret)
  
}

automl_models_h2o@leaderboard %>%
  plot_h2o_performance(newdata = test_tbl, order_by = "auc", max_models = 5, size = 0.8)





