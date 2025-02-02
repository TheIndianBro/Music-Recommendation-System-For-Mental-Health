# Music-Recommendation-System-For-Mental-Health

#R File Description
Description of the R Script (cp.ds.R)
This R script is designed for music effects classification using machine learning models. It preprocesses data, applies feature selection, balances classes, and trains multiple classifiers.

Key Features & Steps:
Libraries Used:

Machine learning: randomForest, xgboost, adabag, e1071
Data handling: missForest, ROSE, caret, caTools
Visualization: ggplot2, reshape2
Feature selection: FSelector
Data Preprocessing:

Loads dataset ds11.csv
Removes unnecessary columns (Timestamp, column_to_delete, Permissions)
Filters out rows with the "Worsen" class in Music.effects
Converts categorical columns into factors
Imputes missing values using missForest
Balances the dataset using ROSE
Feature Selection:

Uses randomForest to rank features
Selects the top 10 features based on MeanDecreaseGini
Machine Learning Models:

Random Forest (RF): Hyperparameter tuning using mtry
Support Vector Machine (SVM): Uses svmLinear
Decision Tree (rpart)
Logistic Regression (glm)
XGBoost (xgbTree): Tuned with parameters (nrounds, max_depth, eta, etc.)
AdaBoost (boosting): Uses rpart for base learners
Model Evaluation:

Splits data into train (70%) and test (30%)
Evaluates models using Confusion Matrix
Computes ROC Curves & AUC Scores for all models
Conclusion
This script applies advanced ML techniques to classify music effects on individuals. It ensures data quality, balances classes, selects optimal features, and evaluates models comprehensively. Let me know if you need modifications or further analysis! 🚀
