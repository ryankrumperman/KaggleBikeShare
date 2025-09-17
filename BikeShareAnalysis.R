
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


# Penalized Regression ----------------------------------------------------

bike_penalized_recipe <- recipe(count ~ ., data = train_data) %>%
  step_mutate(season = as.factor(season)) %>%  # make season factor
  step_time(datetime, features = c("hour")) %>% # extract hour
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% # recode weather
  step_date(datetime, features = "dow") %>% # extract day of week
  step_rm(datetime) %>% # remove original datetime
  step_dummy(all_nominal_predictors()) %>% #make dummy variables
  step_normalize(all_numeric_predictors()) # Make mean 0, sd=1

## Penalized regression model
preg_model <- linear_reg(penalty= 10, mixture=0) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R
preg_wf <- workflow() %>%
  add_recipe(bike_penalized_recipe) %>%
  add_model(preg_model) %>%
  fit(data= train_data)

pred_5<-predict(preg_wf, new_data=test_data)%>%
  mutate(.pred = exp(.pred))

kaggle_submission <- pred_5 %>%
  bind_cols(., test_data) %>% 
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

## Write out the file
vroom_write(x=kaggle_submission, file="./Penalize_Pred5.csv", delim=",")



# Cross Validation --------------------------------------------------------

## Penalized regression model3
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R6

## Set Workflow
preg_wf <- workflow() %>%
add_recipe(bike_penalized_recipe) %>%
add_model(preg_model)

## Grid of values to tune over13
grid_of_tuning_params <- grid_regular(penalty(),
                                      mixture(),
                                      levels = 5) ## L^2 total tuning possibilities16

## Split data for CV18
folds <- vfold_cv(train_data, v = 10, repeats=1)

# Run the CV1
CV_results <- preg_wf %>%
tune_grid(resamples=folds,
          grid=grid_of_tuning_params,
          metrics=metric_set(rmse, mae)) #Or leave metrics NULL

## Plot Results (example)
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric ="rmse")


## Finalize the Workflow & fit it1
final_wf <-preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

## Predict7
cv_guesses<- final_wf %>%
  predict(new_data = test_data) %>%
  mutate(.pred = exp(.pred))

kaggle_submission <- cv_guesses %>%
  bind_cols(., test_data) %>% 
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

## Write out the file
vroom_write(x=kaggle_submission, file="./cv_preds.csv", delim=",")
