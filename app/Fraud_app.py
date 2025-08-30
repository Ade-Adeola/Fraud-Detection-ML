import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load pipeline + threshold (from your .pkl file)
with open('fraud_threshold.pkl', 'rb') as f:
    art = pickle.load(f)

model = art['pipepline']
THRESH = art['threshold']

# Streamlit UI
st.title('Fraud Detection App')
st.subheader('Predict Fraudulent Transactions using individual inputs or full dataset.')

# Option selection
option = st.radio('Choose prediction mode:', ['Single input', 'Upload CSV file'])

if option == 'Single input':
    # Add input fields (from dataset features)

    transaction_time = st.selectbox('Transaction Time',
                                    [f"{h:02d}:00:00" for h in range(24)],  # hourly options
                                    index=12)
    transaction_amount = st.number_input('Transaction Amount', value=0.0, step=10.0)
    transaction_type = st.selectbox('Transaction Type', ['deposit', 'purchase', 'transfer', 'withdrawal'])
    transaction_channel = st.selectbox('Transaction Channel', ['ATM', 'POS', 'mobile_app', 'online'])
    merchant_category = st.selectbox('Merchant Category',
                                     ['digital_goods','electronics','fashion','fuel','gambling',
                                      'groceries','healthcare','restaurants','travel','utilities'])
    is_high_risk_merchant = st.selectbox('High Risk Merchant?', [0, 1])
    customer_age = st.number_input('Customer Age', value=30, step=1)
    customer_income_monthly = st.number_input('Customer Income Monthly', value=0.0, step=100.0)
    customer_tenure_months = st.number_input('Customer Tenure (Months)', value=12, step=1)
    customer_location = st.selectbox('Customer Location',
                                     ['AE-Dubai','GB-London','GH-Accra','KE-Nairobi','NG-Abuja',
                                      'NG-Kano','NG-Lagos','NG-Rivers','US-NY','ZA-Gauteng'])
    email_domain = st.selectbox('Email Domain',
                                ['gmail.com','outlook.com','proton.me','tempmail.net','yahoo.com'])
    chargeback_history_count = st.number_input('Chargeback History Count', value=0, step=1)
    account_balance_before = st.number_input('Account Balance Before', value=0.0, step=100.0)
    account_balance_after = st.number_input('Account Balance After', value=0.0, step=100.0)
    avg_transaction_amount_30d = st.number_input('Avg Transaction Amount (30d)', value=0.0, step=10.0)
    num_transactions_last_24h = st.number_input('Num Transactions (Last 24h)', value=0, step=1)
    velocity_1h = st.number_input('Velocity (1h)', value=0, step=1)
    failed_login_attempts_24h = st.number_input('Failed Login Attempts (24h)', value=0, step=1)
    txn_hour = st.number_input('Transaction Hour (0-23)', value=12, min_value=0, max_value=23, step=1)
    txn_dayofweek = st.number_input('Transaction Day of Week (0=Mon .. 6=Sun)', value=3, min_value=0, max_value=6, step=1)
    distance_from_home_km = st.number_input('Distance from Home (km)', value=0.0, step=0.1)
    device_trust_score = st.number_input('Device Trust Score', value=0.0, step=0.01)
    device_age_days = st.number_input('Device Age (Days)', value=0, step=1)
    is_new_device = st.selectbox('Is New Device?', [0, 1])
    is_foreign_transaction = st.selectbox('Is Foreign Transaction?', [0, 1])

    if st.button('Predict Fraud'):
        # Using correct column names in the order the model was trained
        input_data = {
            'transaction_amount': transaction_amount,
            'transaction_type': transaction_type,
            'transaction_channel': transaction_channel,
            'merchant_category': merchant_category,
            'is_high_risk_merchant': is_high_risk_merchant,
            'customer_age': customer_age,
            'customer_income_monthly': customer_income_monthly,
            'customer_tenure_months': customer_tenure_months,
            'customer_location': customer_location,
            'email_domain': email_domain,
            'chargeback_history_count': chargeback_history_count,
            'account_balance_before': account_balance_before,
            'account_balance_after': account_balance_after,
            'avg_transaction_amount_30d': avg_transaction_amount_30d,
            'num_transactions_last_24h': num_transactions_last_24h,
            'velocity_1h': velocity_1h,
            'failed_login_attempts_24h': failed_login_attempts_24h,
            'txn_hour': txn_hour,
            'txn_dayofweek': txn_dayofweek,
            'distance_from_home_km': distance_from_home_km,
            'device_trust_score': device_trust_score,
            'device_age_days': device_age_days,
            'is_new_device': is_new_device,
            'is_foreign_transaction': is_foreign_transaction
        }

        # convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # predict
        proba = model.predict_proba(input_df)[:, 1][0]
        pred = int(proba >= THRESH)

        st.metric("Fraud Probability", f"{proba:.3f}")
        st.metric("Prediction", "FRAUD" if pred==1 else "NOT FRAUD")

elif option == 'Upload CSV file':
    file = st.file_uploader('Upload your CSV file', type=['csv'])
    if file:
        data = pd.read_csv(file)
        st.write('Uploaded Data Preview:', data.head())
        if st.button('Predict from CSV'):
            proba = model.predict_proba(data)[:, 1]
            preds = (proba >= THRESH).astype(int)
            data['fraud_proba'] = proba
            data['fraud_pred'] = preds
            st.write('Prediction Results:', data.head())
            st.download_button('Download Results',
                               data.to_csv(index=False),
                               file_name='Fraud_Predictions.csv')