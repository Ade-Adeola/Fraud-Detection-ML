# Fraud-Detection-ML

End-to-end fraud detection analysis: data preprocessing, ML modeling, evaluation, and Streamlit deployment.

## 📊 Project Overview
This project demonstrates how to build a machine learning model that predicts whether a financial transaction is fraudulent.  
- Handles **imbalanced data (~4% fraud cases)**  
- Uses **Logistic Regression with class weights** (best performing)  
- Evaluated with metrics like **recall, F1, PR AUC**  
- Deployed using **Streamlit** for interactive predictions  

## ⚙️ Tech Stack
- Python (pandas, numpy, scikit-learn, xgboost)  
- Streamlit (deployment)

## 📈 Key Results
- Accuracy: ~0.78  
- Recall: ~0.46 (priority metric)  
- ROC AUC: ~0.64  
- PR AUC: ~0.08 (≈2× random baseline)  

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app/Fraud_app.py
