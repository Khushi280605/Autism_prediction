# 🧠 Autism Spectrum Disorder Prediction using Machine Learning

## 📌 Overview

This project presents a Machine Learning-based system for predicting Autism Spectrum Disorder (ASD) using behavioral screening questionnaire responses and demographic features.

The goal of this project is to build a reliable predictive model that can assist healthcare professionals in early autism screening. The system follows a complete machine learning pipeline including preprocessing, class imbalance handling, model training, hyperparameter tuning, and evaluation.



## 🎯 Problem Statement

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition that affects communication, behavior, and social interaction. Early detection plays a crucial role in timely intervention.

Traditional diagnostic processes are time consuming and require trained professionals. This project aims to develop a machine learning based supportive screening system using structured questionnaire and demographic data.



## 📊 Dataset

The dataset contains:

- Behavioral screening scores
- Demographic features (gender, ethnicity, country, etc.)
- Target variable (ASD / Non-ASD)

### Preprocessing Steps:

- Removed irrelevant features (e.g., ID, age_desc)
- Handled missing and inconsistent values
- Applied Label Encoding to categorical features
- Treated outliers using the Interquartile Range (IQR) method
- Performed Exploratory Data Analysis (EDA)



## ⚖️ Handling Class Imbalance

The dataset was imbalanced, with fewer ASD cases compared to Non-ASD cases.

To address this, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to the training dataset to generate synthetic samples for the minority class and improve model performance.



## 🤖 Machine Learning Models Implemented

The following classification algorithms were trained and compared:

- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier



## 🔧 Model Optimization

Hyperparameter tuning was performed using **RandomizedSearchCV** with **5-fold cross-validation** to:

- Improve model performance
- Reduce overfitting
- Select the best performing model

### Optimized Parameters Included:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `min_samples_split`

After evaluation, **Random Forest** achieved the best performance and was selected as the final model.



## 📈 Model Evaluation

The selected Random Forest model achieved:

- **Accuracy:** 81.88%
- **AUC Score:** ~0.88

### Evaluation Metrics Used:

- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- ROC Curve
- Feature Importance Analysis

### Confusion Matrix Summary:

- True Negatives: 108
- True Positives: 23
- False Positives: 16
- False Negatives: 13

The model demonstrates balanced performance across classes and effectively distinguishes between ASD and Non-ASD cases.



## 📊 Feature Importance

Feature importance analysis was conducted to identify the most influential features contributing to ASD prediction.

Behavioral screening scores were found to be the most significant predictors.



## 🧠 Project Workflow

```
Dataset
   ↓
Data Cleaning
   ↓
Label Encoding
   ↓
Outlier Treatment (IQR)
   ↓
Train-Test Split (80:20)
   ↓
SMOTE (Training Data Only)
   ↓
Model Training
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Best Model Selection (Random Forest)
   ↓
Evaluation (ROC, Confusion Matrix, F1)
   ↓
Model Saving (Pickle)
```



## 💾 Model Saving

The final trained Random Forest model was saved using the `pickle` library for future reuse and deployment.



## 🚀 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- imbalanced-learn (SMOTE)



## 📌 Key Highlights

✔ Complete end-to-end ML pipeline  
✔ Class imbalance handled using SMOTE  
✔ Multiple model comparison  
✔ Cross-validation for reliable evaluation  
✔ Hyperparameter tuning  
✔ ROC curve and feature importance analysis  
✔ Model saved for deployment  



## 🔮 Future Improvements

- Use larger and more diverse datasets
- Perform advanced feature engineering
- Deploy as a web application for real-time screening
- Explore ensemble stacking methods

