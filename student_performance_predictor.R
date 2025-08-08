# Load required libraries
library(tidyverse)
library(caret)
library(e1071)
library(randomForest)
library(pROC)
library(cowplot)

# Load dataset
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv"
students <- read.csv(url, sep = ";")

# View summary
glimpse(students)

# Add target variable: Pass/Fail based on G3 (Final Grade)
students$Pass <- ifelse(students$G3 >= 10, "Pass", "Fail")
students$Pass <- factor(students$Pass)

# Drop G1, G2, G3 to avoid data leakage
students <- students %>% select(-c(G1, G2, G3))

# Convert character columns to factors
students <- students %>% mutate_if(is.character, as.factor)

# Train-Test Split
set.seed(123)
train_index <- createDataPartition(students$Pass, p = 0.8, list = FALSE)
train_data <- students[train_index, ]
test_data <- students[-train_index, ]

# Train Logistic Regression Model
log_model <- train(Pass ~ ., data = train_data, method = "glm", family = "binomial")

# Train Random Forest Model
rf_model <- train(Pass ~ ., data = train_data, method = "rf", ntree = 100)

# Train SVM Model
svm_model <- train(Pass ~ ., data = train_data, method = "svmLinear")

# Predict on test data
log_preds <- predict(log_model, test_data)
rf_preds <- predict(rf_model, test_data)
svm_preds <- predict(svm_model, test_data)

# Accuracy
cat("Logistic Regression Accuracy:", mean(log_preds == test_data$Pass), "\n")
cat("Random Forest Accuracy:", mean(rf_preds == test_data$Pass), "\n")
cat("SVM Accuracy:", mean(svm_preds == test_data$Pass), "\n")

# Confusion Matrix for Random Forest
cat("\nConfusion Matrix for Random Forest:\n")
print(confusionMatrix(rf_preds, test_data$Pass))

# ROC Curve for all models
log_probs <- predict(log_model, test_data, type = "prob")[, "Pass"]
rf_probs <- predict(rf_model, test_data, type = "prob")[, "Pass"]
svm_probs <- predict(svm_model, test_data, type = "prob")[, "Pass"]

log_roc <- roc(test_data$Pass, log_probs)
rf_roc <- roc(test_data$Pass, rf_probs)
svm_roc <- roc(test_data$Pass, svm_probs)

plot(log_roc, col = "blue", main = "ROC Curve")
lines(rf_roc, col = "green")
lines(svm_roc, col = "red")
legend("bottomright", legend = c("Logistic", "Random Forest", "SVM"),
       col = c("blue", "green", "red"), lwd = 2)

# Feature Importance - Random Forest
importance <- varImp(rf_model)
plot(importance, main = "Feature Importance (Random Forest)")

# Predict on new student data
new_student <- data.frame(
  school = factor("GP", levels = levels(students$school)),
  sex = factor("F", levels = levels(students$sex)),
  age = 17,
  address = factor("U", levels = levels(students$address)),
  famsize = factor("GT3", levels = levels(students$famsize)),
  Pstatus = factor("T", levels = levels(students$Pstatus)),
  Medu = 4,
  Fedu = 4,
  Mjob = factor("teacher", levels = levels(students$Mjob)),
  Fjob = factor("services", levels = levels(students$Fjob)),
  reason = factor("course", levels = levels(students$reason)),
  guardian = factor("mother", levels = levels(students$guardian)),
  traveltime = 1,
  studytime = 3,
  failures = 0,
  schoolsup = factor("no", levels = levels(students$schoolsup)),
  famsup = factor("yes", levels = levels(students$famsup)),
  paid = factor("no", levels = levels(students$paid)),
  activities = factor("yes", levels = levels(students$activities)),
  nursery = factor("yes", levels = levels(students$nursery)),
  higher = factor("yes", levels = levels(students$higher)),
  internet = factor("yes", levels = levels(students$internet)),
  romantic = factor("no", levels = levels(students$romantic)),
  famrel = 4,
  freetime = 3,
  goout = 2,
  Dalc = 1,
  Walc = 1,
  health = 4,
  absences = 2
)

new_pred <- predict(rf_model, new_student)
cat("\nPrediction for new student:", new_pred, "\n")
