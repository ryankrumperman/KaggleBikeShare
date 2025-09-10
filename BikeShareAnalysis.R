library(tidyverse)
library(tidymodels)
library(vroom)

train_data<- vroom("C:/Users/rkrum/Documents/Stat 348/BikeShare/train.csv")
test_data<- vroom("C:/Users/rkrum/Documents/Stat 348/BikeShare/test.csv")

train_data<- train_data %>%
  mutate(weather = factor(weather))
test_data<- test_data %>%
  mutate(weather = factor(weather))

#dump casual and registered

train_data<-train_data %>%
  select(-casual, -registered)

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
