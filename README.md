# Credit Card Fraud Machine Learning Model Analysis

![image](https://github.com/user-attachments/assets/ad09e755-2cfd-4818-9468-60c48528a855)


## Table of Contents

- [Project Introduction](#project-introduction)
    - [Credit Card Fraud Machine Learning Model Jupyter Notebook](#credit-card-fraud-machine-learning-model-jupyter-notebook)
    - [Credit Card Fraud Machine Learning Model Dataset](#credit-card-fraud-machine-learning-model-dataset)
- [Objective](#objective)
- [Analysis Outline](#analysis-outline)
- [Conclusion](#conclusion)

## Project Introduction

2024 has been a difficult year for entry-level data science jobs and for this project, I am interested in analyzing the data science field job market. For this project, I am utilizing a Kaggle-based dataset that web-scraped Canadian job postings for data using Selenium and BeautifulSoup. This dataset provides multiple interesting insights into the data science job market such as in-demand technical skills, expected work experience, and salary ranges. 

### Credit Card Fraud Machine Learning Model Jupyter Notebook

Link: [Credit Card Fraud Machine Learning Model](https://github.com/jasondo-da/Canadian_Data_Analyst_Online_Job_Posting_Analysis/blob/main/da_job_posts_canada_analysis.ipynb)

### Credit Card Fraud Machine Learning Model Dataset

Link: [Original Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

| Columns | Description |
| :------------- | :------------ |
| Time | The number of seconds between this transaction and the first transaction in the dataset |
| V1 | may be result of a PCA Dimensionality reduction to protect user identities and sensitive features (v1-v28) |
| V... | may be result of a PCA Dimensionality reduction to protect user identities and sensitive features (v1-v28) |
| V28 | may be result of a PCA Dimensionality reduction to protect user identities and sensitive features (v1-v28) |
| Amount | Transaction amount |
| Class | 1 is for fraudulent transactions and 0 is for non-fraudulent transactions |


## Objective

The objective of this analysis is to train multiple machine learning models from scikit-learn to find the optimal model to detect fraudulent credit card transactions


## Analysis Outline

- Allocated data into a testing, validation, and testing dataset at a 3-1-1 ratio

- The machine learning models used are the decision tree, random forest, and logistic regression models and all are from scikit-learn

- Trained, validated, and tuned hyperparameters for various machine-learning models to get the optimal accuracy score for each model

- Utilized the optimal machine learning model parameters and input the testing dataset to find the most accurate score for each machine learning model

- Utilize the optimal machine learning model parameters and input the testing dataset to find the highest accuracy score for each machine learning model

- Discovered the optimal machine learning model for the dataset

- Performed a sanity check to compare to the previous optimal model


## Conclusion

In conclusion, the random forest model is the optimal model to use to detect credit card fraud transactions based on the dataset provided. The random forest model recorded a 0.999365 accuracy score much higher than the minimum expected 0.8 accuracy score threshold. This value is significant because, in every 1 million transactions, the model will identify fraud correctly 999365 of the time. 

### Random Forest Model Visualization
![image](https://github.com/user-attachments/assets/1a928f1e-cb1e-47f5-be4e-668c7f44d79f)

