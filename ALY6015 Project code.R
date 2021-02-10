# Load Packages
library(dplyr)
library(tidyr)
library(readxl)
library(glmnet)
library(ggplot2)
library(rpart.plot)
library(forecast)

# Read and clean data
data_raw <- read_excel("AB_NYC_2019.xlsx") %>%
  select(-c(id, name, host_name, last_review)) %>%
  replace_na(list(reviews_per_month = 0));

########### EDA ###########
# price in each group
aggregate(data_raw$price,
          list(data_raw$neighbourhood_group),
          summary)
data_raw %>%
  select(neighbourhood_group, price) %>%
  filter(price < 500) %>%
  ggplot(mapping = aes(x = price,
                       y = neighbourhood_group)) +
  geom_violin(draw_quantiles = c(0.25, 0.5, 0.75))

# price distribution
# Price should be greater than 0, so we drop price <= 0.
data_raw %>%
  filter( price > 0) %>%
  filter(price < 2200) %>%
  ggplot(mapping = aes(x = price)) +
  geom_histogram(binwidth = 50,color="black", fill="blue" )

# Using log to replace the price, we find it is longtail distribution
data_clean <- data_raw %>%
  filter(price > 0) %>%
  mutate(log_price = log(price));
data_clean %>%
  ggplot(mapping = aes(x = log_price)) +
  geom_histogram(binwidth = 0.1, color="black", fill="steelblue")

# log_price vs room_type
data_clean <- data_clean %>%
  mutate(room_type = case_when(room_type == "Shared room" ~ "3",
                               room_type == "Private room" ~ "2",
                               TRUE ~ "1"));
data_clean %>%
  select(room_type, log_price) %>%
  ggplot(mapping = aes(x = room_type,
                       y = log_price)) +
  geom_violin(draw_quantiles = c(0.25, 0.5, 0.75))

# log_price vs minimum_nights
data_clean <- data_clean %>%
  mutate(minimum_nights_bin = cut(minimum_nights,
                                  breaks = c(-Inf, 10, 20 , 30, Inf),
                                  labels = c(1, 2, 3 , 4)));
data_clean %>%
  select(minimum_nights_bin, log_price) %>%
  ggplot(mapping = aes(x = minimum_nights_bin,
                       y = log_price)) +
  geom_violin(draw_quantiles = c(0.25, 0.5, 0.75))

########### Modeling ###########
# Prepare the data
data_clean <- data_clean %>%
  select(neighbourhood_group, room_type, minimum_nights_bin,
         log_price);

# Train test split
sample_size <- nrow(data_clean);
train_ind <- sample(seq(sample_size), size = floor(sample_size * 0.8));

# Multiple Linear Regression
model_MLR <- lm(log_price ~ ., data = data_clean[train_ind,]);
summary(model_MLR)
y_pred_MLR <- predict(model_MLR, newdata = data_clean[-train_ind,]);
accuracy(y_pred_MLR, data_clean[-train_ind,]$log_price)

# LASSO
data_x <- model.matrix(log_price ~ ., data = data_clean)[, -1];
data_y <- data_clean$log_price;
lambda_seq <- 10^seq(-2, 2, by = 0.1);

data_x_train <- data_x[train_ind,];
data_y_train <- data_y[train_ind];

model_cv <- cv.glmnet(data_x_train,
                      data_y_train,
                      alpha = 1,
                      lambda = lambda_seq);
lambda_best <- model_cv$lambda.min;
lasso_best <- glmnet(data_x_train,
                     data_y_train,
                     alpha = 1,
                     lambda = lambda_seq);
plot(lasso_best, "lambda", label = TRUE)
y_pred_LASSO <- predict(lasso_best, s = lambda_best, newx = data_x[-train_ind,]);
accuracy(as.numeric(y_pred_LASSO), data_y[-train_ind])

# Decision tree
model_decisiontree <- rpart(log_price ~ ., data = data_clean[train_ind,],
                  method = "anova");
rpart.plot(model_decisiontree, roundint = FALSE)
print(model_decisiontree)
y_pred_dt <- predict(model_decisiontree, newdata = data_clean[-train_ind,]);
accuracy(y_pred_dt, data_clean[-train_ind,]$log_price)

