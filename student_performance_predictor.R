
# student_performance_predictor.R

# Load required libraries
library(tidyverse)
library(caret)
library(pROC)

# Load dataset
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
temp <- tempfile()
download.file(url, temp)
unzip(temp, files = "student-mat.csv")
student_data <- read.csv("student-mat.csv", sep = ";")
unlink(temp)

# Preprocess data
student_data$G3 <- ifelse(student_data$G3 >= 10, "Pass", "Fail")
student_data$G3 <- as.factor(student_data$G3)

# Split data
set.seed(123)
trainIndex <- createDataPartition(student_data$G3, p = 0.8, list = FALSE)
trainData <- student_data[trainIndex, ]
testData <- student_data[-trainIndex, ]

# Train model
model <- train(G3 ~ ., data = trainData, method = "rf", trControl = trainControl(method = "cv", number = 5))

# Predict
predictions <- predict(model, testData)
confMat <- confusionMatrix(predictions, testData$G3)
print(confMat)

# ROC curve
probs <- predict(model, testData, type = "prob")
roc_curve <- roc(response = testData$G3, predictor = probs[, "Pass"], levels = rev(levels(testData$G3)))
plot(roc_curve, main = "ROC Curve")
