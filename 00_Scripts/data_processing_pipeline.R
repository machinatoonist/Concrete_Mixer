# PREDICTING CONCRETE COMPRESSIVE STRENGTH WITH H2O AND LIME ----
# DATA PREPARATION ----
# data_processing_pipeline.R ----


library(tidyverse)
library(tidyquant)
library(stringr)
library(forcats)

# Processing pipeline for Concrete Data
process_data_readable <- function(data) {
    
    names(data) <- c("Cement_kg", 
                  "Slag_kg", 
                  "Fly_Ash_kg", 
                  "Water_kg", 
                  "Superplasticizer_kg",
                  "Coarse_Aggregate_kg",
                  "Fine_Aggregate_kg",
                  "Age_day",
                  "Compressive_Strength_MPa")
    
    return(data)
    
}