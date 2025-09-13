
# Packages and Reading in Data --------------------------------------------

library(tidyverse)
library(tidymodels)
library(vroom)

train_data <- vroom("C:/Users/rkrum/Documents/Stat 348/BikeShare/train.csv")
test_data  <- vroom("C:/Users/rkrum/Documents/Stat 348/BikeShare/test.csv")

# Cleaning Data -----------------------------------------------------------
train_data <- train_data %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))

# Recipe Linear Model ------------------------------------------------------------------


bike_recipe <- recipe(count ~ ., data = train_data) %>%
  step_mutate(season = as.factor(season)) %>%  # make season factor
  step_time(datetime, features = c("hour")) %>% # extract hour
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% # recode weather
  step_date(datetime, features = "dow") %>% # extract day of week
  step_rm(datetime) # remove original datetime

# Define a model
lin_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

# Combine into a Workflow and fit
bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(lin_model) %>%
  fit(data = train_data)

# Run all the steps on test data
lin_preds <- predict(bike_workflow, new_data = test_data) %>%
  mutate(.pred = exp(.pred))

kaggle_submission <- lin_preds %>%
  bind_cols(., test_data) %>% 
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

## Write out the file
vroom_write(x=kaggle_submission, file="./LinearPredsWrangled.csv", delim=",")


prepped_recipe <- prep(bike_recipe) # Sets up the preprocessing using myDataSet13
baked_train_data<-bake(prepped_recipe, new_data=train_data)


# Basic Linear Regression Model -------------------------------------------


## Setup and Fit the Linear Regression Model
my_linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression") %>%
  fit(formula=count~ ., data=train_data)

## Generate Predictions Using Linear Model
bike_predictions <- predict(my_linear_model,
                            new_data=test_data)
kaggle_submission <- bike_predictions %>%
bind_cols(., test_data) %>% 
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))
## Write out the file
vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")
