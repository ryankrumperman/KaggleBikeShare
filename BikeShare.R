library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(patchwork)
library(gridExtra)

train_data<- vroom("C:/Users/rkrum/Documents/Stat 348/BikeShare/train.csv")
test_data<- vroom("C:/Users/rkrum/Documents/Stat 348/BikeShare/test.csv")

train_data<- train_data %>%
  mutate(weather = factor(weather))
p1<- ggplot(train_data, aes(x=weather))+
  geom_bar()+
  labs( title = "Barplot of Weather")
p2<-ggplot(train_data, aes(x= temp, y= count))+
  geom_point()+
  geom_smooth()+
  labs(title= "Temperature vs Number of Riders", x= "Outside Temp", y= "Number of Users")
p3<-plot_missing(train_data)
p4<-ggplot(train_data, aes(x= windspeed, y= count))+
  geom_point()+
  labs(title= "Windspeed vs number of Riders", x= "Winspeed", y= "Number of Riders")
grid.arrange(p1, p2, p3, p4)


