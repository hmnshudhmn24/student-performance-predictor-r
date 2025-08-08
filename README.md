# 🎓 Student Performance Predictor

Predict final grades of students using machine learning based on study habits, attendance, and social factors. Built with R using caret, the model evaluates performance via confusion matrix and ROC curves. Great for showcasing classification skills, feature engineering, and model evaluation.

## 📌 Overview
This project uses the Student Performance Dataset from UCI to predict student final grades (G3) as Pass or Fail. It involves:
- Data preprocessing
- Feature engineering
- Classification using Random Forest, Logistic Regression, and SVM
- Model evaluation with confusion matrix and ROC curve
- Predicting unseen student records
- Built using R and the caret package

## 📁 Dataset
- **Source**: [UCI Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
- **Data File**: `student-mat.csv` (Math course students)

## 🛠️ Features Used
- `studytime`: Weekly study time
- `failures`: Number of past class failures
- `schoolsup`: Extra educational support
- `famsup`: Family educational support
- `goout`: Going out with friends
- `Dalc`: Workday alcohol consumption
- `Walc`: Weekend alcohol consumption
- `health`: Current health status
- `absences`: Number of school absences
- `G1`, `G2`, `G3`: Grades from different periods (G3 is used to create target label)

## 🧪 Model Pipeline
- **Binary Classification Label**: Students with G3 >= 10 are marked as "Pass", otherwise "Fail".
- **Preprocessing**: Encoding categorical variables, removing G1/G2/G3 to avoid leakage.
- **Training**: Using Random Forest, Logistic Regression, and SVM via `caret`.
- **Evaluation**:
  - Confusion matrix
  - Accuracy score
  - ROC curve & AUC
  - Feature importance visualization

## 📊 Output
- Accuracy scores for each model
- Confusion Matrix for Random Forest
- ROC AUC score
- ROC Curve comparison for all models
- Feature importance chart
- Prediction for new student profile

## 💻 Requirements
- R >= 4.0
- R packages:
  ```r
  install.packages(c("tidyverse", "caret", "randomForest", "pROC", "ggplot2", "e1071"))
  ```

## 🚀 Run the Project
1. Download `student-mat.csv` from the UCI website.
2. Place it in your working directory.
3. Run the R script:
   ```r
   source("student_performance_predictor.R")
   ```

## 📈 Sample ROC Curve Output
The script will generate a multi-model ROC curve with AUC values like:
```
AUC:
- Logistic Regression: ~0.87
- Random Forest: ~0.91
- SVM: ~0.85
```

## 📌 Project Structure
```
student-performance-predictor/
│
├── student_performance_predictor.R   # Main R script with all logic
├── README.md                         # Project documentation
└── student-mat.csv                   # Dataset (to be manually downloaded)
```
