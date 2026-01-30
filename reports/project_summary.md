## 1. Problem Definition

Vehicle insurance fraud is a major challenge for insurance providers, leading to substantial financial losses every year. Fraudulent claims often go undetected due to limitations of manual review processes and rule-based systems.

This project aims to develop an end-to-end machine learning solution that predicts whether a vehicle insurance claim is fraudulent or genuine based on historical claim data.

### Stakeholders
- Insurance companies
- Fraud investigation teams

### Business Impact
- Faster identification of suspicious claims
- Reduced operational costs
- Improved fraud detection accuracy

### Machine Learning Perspective
The task is modeled as a **binary classification problem**, where the model predicts:
- `1` → Fraudulent claim  
- `0` → Genuine claim

### Evaluation Criteria
Due to class imbalance in fraud detection problems, model performance will primarily focus on:
- High **Recall** to capture most fraudulent cases
- Balanced **Precision** to reduce unnecessary investigations
- Supporting metrics such as F1-score and ROC-AUC

## 2. Data Loading and Understanding

The dataset was loaded from the raw data directory and inspected to understand its structure and composition. This step aimed to identify feature types, data distributions, and potential quality issues.

### Dataset Characteristics
- Number of records: To be determined during exploration
- Number of features: 33
- Feature types:
  - Numerical features
  - Categorical features
  - One binary target variable


### Initial Findings
- The dataset includes multiple categorical attributes related to vehicle, policyholder, and claim details.
- Several features contain missing values.
- Fraud detection is expected to be an imbalanced classification problem.

These observations guide the exploratory data analysis and preprocessing steps that follow.

