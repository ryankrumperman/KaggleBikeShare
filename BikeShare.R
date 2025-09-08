library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(patchwork)
library(gridExtra)

data<- vroom("C:/Users/rkrum/Documents/Stat 348/BikeShare/train.csv")
data<- data %>%
  mutate(weather = factor(weather))


p1<- ggplot(data, aes(x=weather))+
  geom_bar()+
  labs( title = "Barplot of Weather")

p2<-ggplot(data, aes(x= temp, y= count))+
  geom_point()+
  geom_smooth()+
  labs(title= "Temperature vs Number of Riders", x= "Outside Temp", y= "Number of Users")

p3<-plot_missing(data)

p4<-ggplot(data, aes(x= windspeed, y= count))+
  geom_point()+
  labs(title= "Windspeed vs number of Riders", x= "Winspeed", y= "Number of Riders")


grid.arrange(p1, p2, p3, p4)
