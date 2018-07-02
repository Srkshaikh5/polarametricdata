# Packages
library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)

# Data
data1 <- read.csv("D:\\Aegis\\ml\\ml evaluation\\train\\train.csv", header = T)

#data<-data[1:1500000,]
summary(data1)
data<-data1
boxplot(data$Expected)

quantile(data$Expected, c(0.6,0.7,0.75,0.78,0.80,0.85,0.90, 0.95, 0.97, 0.98, 0.99))

data$Expected<-ifelse(data$Expected >=4.57,
                 4.57,data$Expected)

boxplot(data$Expected)
max(data$Expected)
summary(data)
# Partition data
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
train <- data[ind==1,]
test <- data[ind==2,]

# Create matrix - One-Hot Encoding for Factor variables


train_label <- train[,"Expected"]
train_matrix <- xgb.DMatrix(data = as.matrix(train), label = train_label)

test_label <- test[,"Expected"]
test_matrix <- xgb.DMatrix(data = as.matrix(test), label = test_label)

# Parameters

xgb_params <- list("objective" = "reg:linear",
                   "eval_metric" = "rmse")
                   
watchlist <- list(train = train_matrix, test = test_matrix)

# eXtreme Gradient Boosting Model
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = 1000,
                       watchlist = watchlist,
                       eta = 0.001,
                       max.depth = 5,
                       subsample = 1,
                       colsample_bytree = 1,
                       missing = NA,
                       seed = 333)
# boxplot(train$Expected)

# Prediction & confusion matrix - test data
p <- predict(bst_model, newdata = test_matrix)
d<-data.frame(Actual=test$Expected,Predicted=p,Error=(test$Expected-p))
d

# RMSE
RMSE = function(x, y){
  sqrt(mean((x - y)^2))
}

RMSE(test$Expected,p)


# Training & test error plot
e <- data.frame(bst_model$evaluation_log)
e
plot(e$iter, e$train_rmse, col = 'blue')
lines(e$iter, e$test_, col = 'red')

m<-min(e$test_rmse)
m
e[e$test_rmse == m,]




