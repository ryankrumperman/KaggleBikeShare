
# Packages and Reading in Data --------------------------------------------

library(tidyverse)
library(tidymodels)
library(vroom)
install.packages("rpart")
library(rpart)
install.packages("ranger")
library(ranger)
install.packages("agua")
library(agua)

train_data <- vroom("C:/Users/rkrum/Documents/Stat 348/BikeShare/train.csv")
test_data  <- vroom("C:/Users/rkrum/Documents/Stat 348/BikeShare/test.csv")

# Cleaning Data -----------------------------------------------------------
train_data <- train_data %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))

# Recipe Linear Model ------------------------------------------------------------------

bike_recipe <- recipe(count ~ ., data = train_data) %>%
  step_mutate(season = as.factor(season)) %>%
  step_mutate(weather = as.factor(ifelse(weather == 4, 3, weather))) %>%
  
  # extract time features
  step_time(datetime, features = c("hour")) %>%
  step_date(datetime, features = c("dow", "month", "year")) %>%
  
  # coerce to factors (use the actual column names created)
  step_mutate_at(c("datetime_hour", "datetime_dow", "datetime_month", "datetime_year"),
                 fn = as.factor) %>%
  
  # interaction between hour and dow
  step_interact(~ datetime_hour:datetime_dow) %>%
  
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())





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
baked_test_data<- bake(prepped_recipe, new_data = test_data)


write.csv(baked_test_data, "baked_test.csv")
write.csv(baked_train_data, "baked_train.csv")

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


# Regression Trees --------------------------------------------------------



my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # What R function to use
  set_mode("regression")

## Create a workflow with model & recipe
tree_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(my_mod)
## Set up grid of tuning values
grid_of_tuning_params <- grid_regular(tree_depth(),
                                      cost_complexity(),
                                      min_n(),
                                      levels = 3)
## Set up K-fold CV
folds <- vfold_cv(train_data, v = 10, repeats=1)
CV_results <- tree_workflow %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse, mae))
bestTune <- CV_results %>%
  select_best(metric ="rmse")

final_wf <-tree_workflow %>%
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
vroom_write(x=kaggle_submission, file="./tree_preds.csv", delim=",")

## Find best tuning parameters
## Finalize workflow and predict

# Random Forests ----------------------------------------------------------
my_mod <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500
) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Workflow
tree_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(my_mod)

# Grid for tuning only tunable parameters
grid_of_tuning_params <- expand_grid(
  mtry = c(1,3,7,10),
  min_n = c(1,5,10)
)

# K-fold CV
folds <- vfold_cv(train_data, v = 10, repeats = 1)

# Tune
CV_results <- tree_workflow %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params,
    metrics = metric_set(rmse, mae)
  )

# Select best tuning
bestTune <- CV_results %>%
  select_best(metric = "rmse")

# Final workflow
final_wf <- tree_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

# Predictions
cv_guesses <- final_wf %>%
  predict(new_data = test_data) %>%
  mutate(.pred = exp(.pred))  # only do this if your model was trained on log(count)

# Prepare submission
kaggle_submission <- cv_guesses %>%
  bind_cols(test_data) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(
    count = pmax(0, count),
    datetime = as.character(format(datetime))
  )

# Write out the file
vroom_write(x = kaggle_submission, file = "./forest_preds.csv", delim = ",")

# Boosting ----------------------------------------------------------------
install.packages("bonsai")
install.packages("lightgbm")
install.packages("dbarts")
library(dbarts)
library(bonsai)
library(lightgbm)


bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate9
  set_engine("dbarts") %>% # might need to install10
  set_mode("regression")

# Workflow
bart_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(bart_model)

# Grid for tuning only tunable parameters
grid_of_tuning_params <- expand_grid( trees = c(50,100,250)
)

# K-fold CV
folds <- vfold_cv(train_data, v = 5, repeats = 1)

# Tune
CV_results <- bart_workflow %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params,
    metrics = metric_set(rmse, mae)
  )

# Select best tuning
bestTune <- CV_results %>%
  select_best(metric = "rmse")

# Final workflow
final_wf <- bart_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

# Predictions
cv_guesses <- final_wf %>%
  predict(new_data = test_data) %>%
  mutate(.pred = exp(.pred))  # only do this if your model was trained on log(count)

# Prepare submission
kaggle_submission <- cv_guesses %>%
  bind_cols(test_data) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(
    count = pmax(0, count),
    datetime = as.character(format(datetime))
  )

# Write out the file
vroom_write(x = kaggle_submission, file = "./BART2_preds.csv", delim = ",")



# Stacking ----------------------------------------------------------------

## Initialize an h2o session
h2o::h2o.init()

auto_model <- auto_ml() %>%
set_engine("h2o", max_runtime_secs=180, max_models = 5) %>%
set_mode("regression")

## Combine into Workflow
automl_wf <- workflow() %>%
add_recipe(bike_recipe) %>%
add_model(auto_model) %>%
fit(data=train_data)

cv_guesses<- automl_wf %>%
  predict(new_data = test_data) %>%
  mutate(.pred = exp(.pred))

kaggle_submission <- cv_guesses %>%
  bind_cols(., test_data) %>% 
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))





# datarobot ---------------------------------------------------------------

data_robot<-vroom("C:\\Users\\rkrum\\Downloads\\result-68dae154984c1666ce94922c.csv")

data<-data.frame(count = data_robot$count_PREDICTION, datetime=test_data$datetime)

data<- data %>%
  mutate(count = exp(count))


kaggle_submission <- data %>% 
  select(datetime, count) %>%
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))
vroom_write(x=kaggle_submission, file="./data_preds.csv", delim=",")
