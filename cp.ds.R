# Reset graphics device
graphics.off()

# Load libraries
library(missForest)
library(ROSE)
library(randomForest)
library(caret)
library(caTools)
library(xgboost)
library(adabag)
library(pROC)
library(reshape2)  # for melt function
library(FSelector)
library(rpart)
library(e1071)
library(ggplot2)  # Load ggplot2 last

# Load your dataset
f <- read.csv("ds11.csv", na.strings = "")

# Preprocess your dataset
# Remove columns not needed
columns_to_delete <- c("Timestamp", "column_to_delete", "Permissions")
f <- f[, !(names(f) %in% columns_to_delete)]

# Display class statistics
cat("Class distribution:\n")
cat(table(f$Music.effects), "\n")

# Filter out rows with "Worsen" class
f <- f[f$Music.effects != "Worsen", ]

# Impute missing values
character_columns <- sapply(f, is.character)
f[character_columns] <- lapply(f[character_columns], as.factor)
imputed_data <- missForest(f)$ximp
balanced_data <- ROSE(Music.effects ~ ., data = imputed_data, N = 700)$data

# Display class distribution after balancing
cat("\nClass distribution after balancing:\n")
cat(table(balanced_data$Music.effects), "\n")

# Split data
set.seed(123)
index <- createDataPartition(balanced_data$Music.effects, p = 0.7, list = FALSE)
train_data <- balanced_data[index, ]
test_data <- balanced_data[-index, ]

# Feature Selection using randomForest importance
rf_model <- randomForest(Music.effects ~ ., data = train_data)
importance <- as.data.frame(importance(rf_model))
top_features <- rownames(importance)[order(-importance$MeanDecreaseGini)][1:10]

# Print selected features
cat("\nSelected features:\n")
print(top_features)

# Train Random Forest model with tuning
random_Forest <- train(
  Music.effects ~ .,
  data = train_data,
  method = "rf",
  trControl = trainControl(method = "cv", number = 10, savePredictions = "final"),
  tuneGrid = expand.grid(
    mtry = c(2, 4, 6) 
  )
)


predictions_rf <- predict(random_Forest, newdata = test_data)


confusion_matrix_rf <- confusionMatrix(predictions_rf, reference = test_data$Music.effects)
cat("\nRandom Forest Confusion Matrix:\n")
print(confusion_matrix_rf)

# Train SVM classifier
svm_model <- train(
  Music.effects ~ .,
  data = train_data,
  method = "svmLinear",
  trControl = trainControl(method = "cv", number = 10, savePredictions = "final")
)

# Make predictions on the test data for SVM
predictions_svm <- predict(svm_model, newdata = test_data)

# Create the confusion matrix for SVM
confusion_matrix_svm <- confusionMatrix(predictions_svm, reference = test_data$Music.effects)
cat("\nSVM Confusion Matrix:\n")
print(confusion_matrix_svm)

# Train Decision Tree model
decision_tree_model <- train(
  Music.effects ~ .,
  data = train_data,
  method = "rpart",
  trControl = trainControl(method = "cv", number = 7, savePredictions = "final")
)

# Make predictions on the test data using Decision Tree
predictions_dt <- predict(decision_tree_model, newdata = test_data)

# Create the confusion matrix for Decision Tree
confusion_matrix_dt <- confusionMatrix(predictions_dt, reference = test_data$Music.effects)
cat("\nDecision Tree Confusion Matrix:\n")
print(confusion_matrix_dt)

# Train Logistic Regression model
logistic_model <- train(
  Music.effects ~ .,
  data = train_data,
  method = "glm",
  trControl = trainControl(method = "cv", number = 10, savePredictions = "final")
)

# Make predictions on the test data for Logistic Regression
predictions_logistic <- predict(logistic_model, newdata = test_data)

# Create the confusion matrix for Logistic Regression
confusion_matrix_logistic <- confusionMatrix(predictions_logistic, reference = test_data$Music.effects)
cat("\nLogistic Regression Confusion Matrix:\n")
print(confusion_matrix_logistic)

# Train XGBoost model
xgboost_model <- train(
  Music.effects ~ .,
  data = train_data,
  method = "xgbTree",

  tuneGrid = expand.grid(
    nrounds = 10,
    max_depth = 3,
    eta = 0.3,
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1
  )
)

# Make predictions on the test data for XGBoost
predictions_xgboost <- predict(xgboost_model, newdata = test_data)

# Create the confusion matrix for XGBoost
confusion_matrix_xgboost <- confusionMatrix(predictions_xgboost, reference = test_data$Music.effects)
cat("\nXGBoost Confusion Matrix:\n")
print(confusion_matrix_xgboost)

# Train AdaBoost model with different parameters
adaboost_model <- boosting(
  Music.effects ~ .,
  data = train_data,
  mfinal = 10,  # Example: Try different values of mfinal
  control = rpart.control(maxdepth = 1)  # Example: Adjust rpart control parameters
)

# Make predictions on the test data for AdaBoost
predictions_adaboost <- predict(adaboost_model, newdata = test_data)

# Ensure factor levels match
predictions_adaboost$class <- factor(predictions_adaboost$class, levels = levels(test_data$Music.effects))

# Create the confusion matrix for AdaBoost
confusion_matrix_adaboost <- confusionMatrix(predictions_adaboost$class, reference = test_data$Music.effects)
cat("\nAdaBoost Confusion Matrix:\n")
print(confusion_matrix_adaboost)

# Function to plot ROC curve
plot_roc_curve <- function(predictions, reference, model_name) {
  # Calculate the probabilities if not already given
  if (!is.numeric(predictions)) {
    predictions <- as.numeric(predictions) - 1
  }
  roc_obj <- roc(reference, predictions)
  plot(roc_obj, main = paste("ROC Curve for", model_name))
  print(auc(roc_obj))
}

# Random Forest ROC Curve
rf_probabilities <- predict(random_Forest, newdata = test_data, type = "prob")
plot_roc_curve(rf_probabilities[, 2], test_data$Music.effects, "Random Forest")

# SVM ROC Curve
svm_probabilities <- predict(svm_model, newdata = test_data, decision.values = TRUE)
plot_roc_curve(svm_probabilities, test_data$Music.effects, "SVM")

# Decision Tree ROC Curve
dt_probabilities <- predict(decision_tree_model, newdata = test_data, type = "prob")
plot_roc_curve(dt_probabilities[, 2], test_data$Music.effects, "Decision Tree")

# Logistic Regression ROC Curve
logistic_probabilities <- predict(logistic_model, newdata = test_data, type = "prob")
plot_roc_curve(logistic_probabilities[, 2], test_data$Music.effects, "Logistic Regression")

# XGBoost ROC Curve
xgboost_probabilities <- predict(xgboost_model, newdata = test_data, type = "prob")
plot_roc_curve(xgboost_probabilities[, 2], test_data$Music.effects, "XGBoost")

# AdaBoost ROC Curve
adaboost_probabilities <- predict(adaboost_model, newdata = test_data, type = "prob")
plot_roc_curve(adaboost_probabilities[, 2], test_data$Music.effects, "AdaBoost")

