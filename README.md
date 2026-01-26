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

