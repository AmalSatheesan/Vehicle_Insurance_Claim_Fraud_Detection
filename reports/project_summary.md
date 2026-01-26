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

