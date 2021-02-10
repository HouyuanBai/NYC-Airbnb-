#---- Load Required Packages ----
library(dplyr)
library(tidyr)
library(readxl)
library(glmnet)
library(ggplot2)
library(rpart.plot)
library(caret)
library(e1071)
library(gbm)
library(forecast)
library(randomForest)

##---- Read and clean data ----##
ab <- read.csv("~/Desktop/AB_NYC_2019.csv",header = T)
glimpse(ab)
# select variables, delete meaningless variables
drop <-c("id","name","host_id","host_name","last_review", "neighbourhood")
ab1=ab[,!(names(ab)%in%drop)]
# check missing value
summary(is.na(ab1))
ab1 <- ab1%>% mutate(reviews_per_month =replace_na(reviews_per_month,0))
summary(is.na(ab1))
# convert room_type to numeric 
unique(ab1$room_type)
ab1$room_type_new <-
  ifelse(
    ab1$room_type == 'Private room', yes = 1,
  ifelse(
    ab1$room_type == 'Entire home/apt', yes = 2,
  ifelse(
    ab1$room_type == 'Shared room ',yes = 3, no=0)
    )
  )
# convert neighbourhood to numeric
unique(ab1$neighbourhood_group)
ab1$neighbourhood_new <-
  ifelse(
    ab1$neighbourhood_group == 'Manhattan',yes = 1,
  ifelse(
    ab1$neighbourhood_group == 'Brooklyn',yes = 2,
  ifelse(
    ab1$neighbourhood_group == 'Queens',yes = 3,
  ifelse(
    ab1$neighbourhood_group == 'Staten Island',yes = 4,
  ifelse(
    ab1$neighbourhood_group == 'Bronx', yes = 5, no=0)
        )
      )
    )
  )
drop2 <-c("neighbourhood_group", "room_type")
ab2=ab1[,!(names(ab1)%in%drop2)]

##---- EDA ----##
summary(ab2$price)
# ggplot for price
price_ggplot= ggplot(ab2, aes(x=price)) + 
  geom_histogram(colour = "dark blue", fill ="#56B4E9", binwidth = 500) + 
  geom_vline(aes(xintercept=mean(price, na.rm = TRUE)), 
             color="red", linetype="dashed",size=1) + 
  annotate(geom = 'text', x = mean(ab1$price), y = 1000,
           label = paste('Mean Price =',round(mean(ab2$price), 2)), 
           colour = "red", hjust = -1, vjust=-15)
print(price_ggplot + labs(title = "The histogram of price for NYC Airbnb in 2019",
                          y="count"))

# Using log to replace the price
# ggplot for log price
ab3 <- ab2%>%
  filter(price > 0) %>%
  mutate(log_price = log(price));
ab3 %>%
  ggplot(mapping = aes(x = log_price)) +
  geom_histogram(binwidth = 0.5, color="dark blue", fill="#56B4E9")+
  labs(title="The histogram of log price for NYC Airbnb in 2019",
       y="count")

#------Build Model------#
# split train and test
drop3 <-c("price")
ab4=ab3[,!(names(ab3)%in%drop3)]
sub_ab <- sample(nrow(ab4), 0.8*nrow(ab4))
test_ab <- ab4[-sub_ab,]
train_ab <- ab4[sub_ab,]

#----------GLM---------#
model1 <- glm(log_price ~ ., data = train_ab)
summary(model1)
pred1 <- predict(model1, newdata = test_ab)
table1 <- table(test_ab$log_price, pred1)

# R squared
mse1= sum((pred1-test_ab$log_price)^2)/nrow(test_ab)
var1=sum((test_ab$log_price-mean(test_ab$log_price))^2)/(nrow(test_ab)-1)
Rs1=1-(mse1/var1)
Rs1
confusionMatrix(pred1, test_ab)
# RMSE
accuracy(pred1, test_ab$log_price)

#---- Modeling: Decision Tress ----
# Decision tree
model2<- rpart(log_price ~ ., data = train_ab, method = "anova")
rpart.plot(model2, roundint = FALSE)
pred2 <- predict(model2, newdata = test_ab)
# accuracy evaluate
# R squared
mse2= sum((pred2-test_ab$log_price)^2)/nrow(test_ab)
var2=sum((test_ab$log_price-mean(test_ab$log_price))^2)/(nrow(test_ab)-1)
Rs2=1-(mse2/var2)
Rs2
# RMSE
accuracy(pred2, test_ab$log_price)

#---- Modeling: random forest ----

model3<- randomForest(log_price~., train_ab, ntree = 500)
model3
pred3 <- predict(model3, test_ab, type = 'class')
# accuracy evaluate
# R squared
mse3= sum((pred3-test_ab$log_price)^2)/nrow(test_ab)
var3=sum((test_ab$log_price-mean(test_ab$log_price))^2)/(nrow(test_ab)-1)
Rs3=1-(mse3/var3)
Rs3
# RMSE
accuracy(pred3, test_ab$log_price)

#------- Model: Gradient Boosted---------#
gbm <- train(log_price~ ., data = train_ab, method = "gbm",
                  trControl = trainControl("cv", number = 10))
print(gbm)             
plot(gbm)             
pred4 <- predict(gbm, test_ab)
# accuracy evaluate
# R squared
mse4= sum((pred4-test_ab$log_price)^2)/nrow(test_ab)
var4=sum((test_ab$log_price-mean(test_ab$log_price))^2)/(nrow(test_ab)-1)
Rs4=1-(mse4/var4)
Rs4
# RMSE
accuracy(pred4, test_ab$log_price)

#----- variable importance-----#
varImp1<- varImp(model2, scale=FALSE)
varImp1
