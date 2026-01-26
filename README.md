## 📌 Problem Definition

Insurance companies face significant financial losses due to fraudulent vehicle insurance claims. Traditional manual fraud detection methods are time-consuming, expensive, and often fail to detect sophisticated fraud patterns.

The objective of this project is to build a machine learning-based system that can automatically identify potentially fraudulent vehicle insurance claims using historical claim data. This system aims to assist insurance companies in prioritizing high-risk claims for investigation, thereby reducing losses and improving operational efficiency.

### 🎯 Business Objective
- Detect fraudulent vehicle insurance claims early
- Reduce financial loss due to fraud
- Support fraud investigation teams with data-driven insights

### 🧠 Machine Learning Formulation
This problem is formulated as a **supervised binary classification task**:
- `1` → Fraudulent claim  
- `0` → Genuine claim

### 📊 Success Metrics
Since fraudulent claims form a minority class, accuracy alone is not sufficient. The model will be evaluated using:
- **Recall** (primary metric) – to detect as many fraudulent claims as possible
- **Precision** – to minimize false fraud alerts
- **F1-Score**
- **ROC-AUC**

## 📊 Step 2: Data Loading and Understanding

The raw vehicle insurance claim dataset was loaded and examined to understand its structure, feature types, and overall data quality. This step focuses on gaining familiarity with the dataset before performing any cleaning or transformations.

### Dataset Overview
- Source: Kaggle – Vehicle Insurance Claim Fraud Detection
- Total attributes: 33
- Target variable: Fraud indicator (binary)

### Key Observations
- The dataset contains both numerical and categorical features.
- The target variable represents whether a claim is fraudulent or genuine.
- Initial inspection revealed the presence of missing values and categorical dominance, which will be addressed in later steps.
