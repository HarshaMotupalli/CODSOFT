# CODSOFT Machine Learning Tasks Completion
Welcome to my CODSOFT repository! Here, I'll be uploading my solutions for the tasks that were completed as part of an internship at CODSOFT.
## Task-2 : CREDIT CARD FRAUD DETECTION
### Overview
This project focuses on the task of detecting credit card fraud. It uses the fraudTest.csv dataset, which contains information about various credit card transactions, including transaction details, customer information, and whether the transaction was fraudulent or not.
### File
Task2.ipynb : This file tackles the problem of credit card fraud detection.It preprocess the transaction data, handles missing values, and trains a Logistic Regression model.
### Steps
Data Preprocessing:  The dataset is loaded and preprocessed, including handling missing values and encoding categorical variables.

Feature Engineering: The dataset is transformed and encoded to prepare it for the machine learning model.

Model Training: A Logistic Regression model is trained on the preprocessed data.

Model Evaluation: The model's accuracy is evaluated on the test set.
### Result
The Logistic Regression model achieved an accuracy of 99.61% on the test set.
## Task-3 : CUSTOMER CHURN PREDICTION
### Overview
This project aims to predict customer churn for a bank. It uses the Churn_Modelling.csv dataset, which includes various customer-related features such as credit score, geography, gender, age, tenure, balance, number of products, credit card status, active member status, and estimated salary.
### File
Task3.ipynb : This file addresses the problem of customer churn prediction in the banking industry.It preprocesses the data, encodes categorical features, and trains a Random Forest Classifier model.
### Steps
Data Preprocessing: The dataset is loaded and preprocessed, including handling missing values and encoding categorical variables.

Feature Engineering: Additional features are created based on the provided information.

Model Training: A Random Forest Classifier is trained on the preprocessed data.

Model Evaluation: The model's accuracy, confusion matrix, and classification report are generated to evaluate its performance.
### Result
The Random Forest Classifier achieved an accuracy of 87% on the test set.
## Task-4 : SPAM SMS DETECTION
### Overview
This project focuses on the task of detecting spam SMS messages. It utilizes the spam.csv dataset to build a model that can accurately classify SMS messages as either "ham" (legitimate) or "spam".
### File
Task4.ipynb : This file presents a solution for spam SMS detection.It performs text cleaning, TF-IDF vectorization, and uses an SVM model to classify SMS messages as spam or ham.
### Steps
Data Preprocessing: The dataset is loaded and preprocessed, including handling missing values, removing duplicates, and cleaning the text data.

Feature Extraction: The text data is converted to numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) technique.

Model Training: A Support Vector Machine (SVM) classifier is trained on the preprocessed data.

Model Evaluation: The model's accuracy is evaluated on the test set.
### Result
The Support Vector Machine (SVM) classifier achieved an accuracy of 98.16% on the test set.
