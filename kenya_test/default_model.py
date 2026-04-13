#!/usr/bin/env python3

import pandas as pd
import numpy as np
import psycopg2
from clickhouse_connect import get_client
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")

CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD")
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST")
CLICKHOUSE_PORT = os.getenv("CLICKHOUSE_PORT")
# --------------------------
# Command-line arguments
# --------------------------
parser = argparse.ArgumentParser(description="PAR30 Training + Prediction Pipeline")
parser.add_argument("--as_of_month", required=True, help="YYYY-MM cutoff (e.g. 2026-02)")
parser.add_argument("--save_model", default="xgb_par30_model.pkl")
parser.add_argument("--predict_next_month", action="store_true")
args = parser.parse_args()

# --------------------------
# Database connections
# --------------------------
def get_postgres_conn():
    return psycopg2.connect(
    host=POSTGRES_HOST,
    database=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    port=POSTGRES_PORT
)

def get_clickhouse_client():
    return get_client(
    host=CLICKHOUSE_HOST,
    port=CLICKHOUSE_PORT,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASSWORD,
    secure=True
)

# --------------------------
# Data Extraction
# --------------------------
def extract_data():
    conn = get_postgres_conn()
    ch = get_clickhouse_client()

    # CDS
    cds_df = pd.read_sql("SELECT customerid FROM data_science.cds", conn)

    # Accounts
    accounts = ch.query_df("""
        SELECT account_id, customerId AS customer_id,
               status AS current_account_status, accountType, updatedAt,
               companyRegion, createdAt AS account_created_at
        FROM marts.accounts_mv
    """)
    accounts = accounts[accounts['companyRegion'] == 'kenya']
    accounts['customer_id'] = accounts['customer_id'].astype(int)
    accounts = accounts.sort_values(['customer_id','updatedAt'], ascending=[True, False])
    accounts = accounts.drop_duplicates('customer_id')
    accounts = accounts.drop(['updatedAt','companyRegion'], axis=1)

    # Credit
    credit = pd.read_sql("""
        SELECT customer_id, account_id, account_type, installment_type,
               expected_date, expected_amount, final_amount_paid,
               final_paid_date, amount_due, days_late, total_balance
        FROM data_science.agg_credit_history_v1
        WHERE country='kenya'
    """, conn)
    credit['expected_date'] = pd.to_datetime(credit['expected_date'])
    credit['final_paid_date'] = pd.to_datetime(credit['final_paid_date'])
    credit['month'] = credit['expected_date'].dt.to_period('M').dt.to_timestamp()

    # Payments
    pmts = pd.read_sql("""
        SELECT *
        FROM data_science.agg_monthly_payments_v1
        WHERE country='kenya'
    """, conn)
    pmts['payment_month'] = pd.to_datetime(pmts['payment_month'])
    pmts = pmts.drop(['country','sync_timestamp','customer_type'], axis=1, errors='ignore')

    # IOT
    iot = pd.read_sql("""
        SELECT *
        FROM data_science.agg_device_telemetry_v1
        WHERE country='kenya'
    """, conn)
    iot['device_timestamp_month'] = pd.to_datetime(iot['device_timestamp_month'])
    iot = iot.drop(['country','sync_timestamp','total_telemetry_record_count',
                    'device_id_count','account_id_count'], axis=1, errors='ignore')

    return cds_df, accounts, credit, pmts, iot

# --------------------------
# Transform + Features
# --------------------------
def transform_data(accounts, credit, pmts, iot):

    # Credit features
    credit['dpd'] = (credit['days_late'])
    credit['missed_payment'] = (credit['final_amount_paid'] == 0).astype(int)
    credit['partial_payment'] = ((credit['final_amount_paid'] < credit['expected_amount']) & (credit['final_amount_paid'] != 0)).astype(int)
    credit['payment_ratio'] = credit['final_amount_paid'] / credit['expected_amount']
    credit['par30'] = (credit['dpd'] >= 30).astype(int)

    # Merge all sources
    df = accounts.merge(credit, on=['customer_id','account_id'], how='left')
    df = df.merge(pmts, left_on=['customer_id','month'], right_on=['customer_id','payment_month'], how='left')
    df = df.merge(iot, left_on=['customer_id','month'], right_on=['customer_id','device_timestamp_month'], how='left')
    df = df.drop_duplicates()
    df = df.sort_values(['customer_id','month'])

    # Rolling / lag features
    df['par30_next_month'] = df.groupby('customer_id')['par30'].shift(-1)
    df['payment_ratio_1m'] = df.groupby('customer_id')['payment_ratio'].shift(1)
    df['avg_payment_ratio_3m'] = df.groupby('customer_id')['payment_ratio'].shift(1).rolling(3).mean()
    df['avg_payment_ratio_6m'] = df.groupby('customer_id')['payment_ratio'].shift(1).rolling(6).mean()
    df['missed_payments_3m'] = df.groupby('customer_id')['missed_payment'].shift(1).rolling(3).sum()
    df['missed_payments_6m'] = df.groupby('customer_id')['missed_payment'].shift(1).rolling(6).sum()
    df['partial_payments_3m'] = df.groupby('customer_id')['partial_payment'].shift(1).rolling(3).sum()
    df['payment_gap'] = df['expected_amount'] - df['final_amount_paid']
    df['avg_gap_3m'] = df.groupby('customer_id')['payment_gap'].shift(1).rolling(3).mean()
    df['avg_balance_3m'] = df.groupby('customer_id')['total_balance'].shift(1).rolling(3).mean()
    df['payment_count_3m'] = df.groupby('customer_id')['total_payment_count'].shift(1).rolling(3).sum()
    df['payment_amount_3m'] = df.groupby('customer_id')['total_payment_amount'].shift(1).rolling(3).sum()
    df['days_late_last_month'] = df.groupby('customer_id')['days_late'].shift(1)
    df['max_days_late_3m'] = (df.groupby('customer_id')['days_late'].shift(1).rolling(3).max())
    df['max_days_late_6m'] = (df.groupby('customer_id')['days_late'].shift(1).rolling(6).max())

    return df

# --------------------------
# Model Training
# --------------------------
def train_model(df):
    features = [
        'payment_ratio_1m','avg_payment_ratio_3m','avg_payment_ratio_6m',
        'missed_payments_3m','missed_payments_6m','partial_payments_3m',
        'avg_gap_3m','avg_balance_3m','payment_count_3m','payment_amount_3m',
        'days_late_last_month', 'max_days_late_3m','max_days_late_6m'
    ]

    model_df = df.dropna(subset=['par30_next_month'])
    train = model_df[(model_df['month'] >= '2023-01-01') & (model_df['month'] < '2025-09-01')]
    val = model_df[(model_df['month'] >= '2025-09-01') & (model_df['month'] <= '2025-11-30')]

    X_train, y_train = train[features], train['par30_next_month']
    X_val, y_val = val[features], val['par30_next_month']

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    print("Validation AUC:", roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))
    return model, features

# --------------------------
# Get expected dates for prediction month
# --------------------------
def get_expected_dates(credit, prediction_month):
    temp = credit[credit['month'] == prediction_month]
    expected = temp.sort_values(['customer_id','expected_date']).groupby('customer_id').first().reset_index()
    return expected[['customer_id','expected_date']]

# --------------------------
# Predict Next Month
# --------------------------
def predict_all(model, df, features, cutoff):
    prediction_month = cutoff + pd.offsets.MonthBegin(1)
    df = df[(df['accountType']=="PAYG") & ((df["current_account_status"]=="Arrears") | (df["current_account_status"]=="Current") | (df["current_account_status"]=="Pending Repossession"))]
    latest = df[df['month'] <= cutoff].sort_values(['customer_id','month']).groupby('customer_id').tail(1).copy()
    latest['par30_probability'] = model.predict_proba(latest[features])[:,1]

    def segment(p):
        if p < 0.2: return "Low"
        elif p < 0.5: return "Medium"
        else: return "High"

    latest['risk_segment'] = latest['par30_probability'].apply(segment)
    expected_dates = get_expected_dates(df, prediction_month)
    latest = latest.drop(['expected_date'], axis=1).merge(expected_dates, on='customer_id', how='left')
    latest['prediction_month'] = prediction_month
    return latest[['customer_id', 'accountType','current_account_status', 'prediction_month','expected_date','par30_probability','risk_segment']]

# --------------------------
# Main
# --------------------------
def main():
    cutoff = pd.to_datetime(args.as_of_month).to_period('M').to_timestamp()
    print("Extracting data...")
    cds_df, accounts, credit, pmts, iot = extract_data()

    print("Transforming and feature engineering...")
    df = transform_data(accounts, credit, pmts, iot)

    print("Training model...")
    model, features = train_model(df)
    joblib.dump(model, args.save_model)
    print(f"Model saved to {args.save_model}")

    if args.predict_next_month:
        print("Predicting next month...")
        preds = predict_all(model, df, features, cutoff)
        preds.to_csv("next_month_predictions.csv", index=False)
        print("Predictions saved to next_month_predictions.csv")

if __name__ == "__main__":
    main()