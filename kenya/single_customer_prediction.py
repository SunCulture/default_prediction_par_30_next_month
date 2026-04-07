#!/usr/bin/env python3

import pandas as pd
import numpy as np
import psycopg2
from clickhouse_connect import get_client
import joblib
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
# CLI
# --------------------------
parser = argparse.ArgumentParser(description="Predict + Explain Single Customer")
parser.add_argument("--model_path", required=True)
parser.add_argument("--customer_id", required=True, type=int)
parser.add_argument("--as_of_month", required=True)
args = parser.parse_args()

# --------------------------
# Connections
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
# Extract
# --------------------------
def extract_data():
    conn = get_postgres_conn()
    ch = get_clickhouse_client()

    accounts = ch.query_df("""
        SELECT account_id, customerId AS customer_id,
               status AS current_account_status,
               accountType, updatedAt, companyRegion,
               createdAt AS account_created_at
        FROM marts.accounts_mv
    """)

    accounts = accounts[accounts['companyRegion'] == 'kenya']
    accounts['customer_id'] = accounts['customer_id'].astype(int)
    accounts = accounts.sort_values(['customer_id','updatedAt'], ascending=[True, False])
    accounts = accounts.drop_duplicates('customer_id')

    credit = pd.read_sql("""
        SELECT *
        FROM data_science.agg_credit_history_v1
        WHERE country='kenya'
    """, conn)

    credit['expected_date'] = pd.to_datetime(credit['expected_date'])
    credit['final_paid_date'] = pd.to_datetime(credit['final_paid_date'])
    credit['month'] = credit['expected_date'].dt.to_period('M').dt.to_timestamp()

    pmts = pd.read_sql("""
        SELECT *
        FROM data_science.agg_monthly_payments_v1
        WHERE country='kenya'
    """, conn)

    pmts['payment_month'] = pd.to_datetime(pmts['payment_month'])
    pmts = pmts.drop(['country', 'sync_timestamp', 'customer_type'], axis=1)

    iot = pd.read_sql("""
        SELECT *
        FROM data_science.agg_device_telemetry_v1
        WHERE country='kenya'
    """, conn)

    iot['device_timestamp_month'] = pd.to_datetime(iot['device_timestamp_month'])
    iot = iot.drop(['country','sync_timestamp','total_telemetry_record_count',
                    'device_id_count','account_id_count'], axis=1, errors='ignore')

    return accounts, credit, pmts, iot

# --------------------------
# Feature Engineering
# --------------------------
def transform(accounts, credit, pmts, iot, cutoff):

    credit = credit[credit['month'] <= cutoff]
    pmts = pmts[pmts['payment_month'] <= cutoff]
    iot = iot[iot['device_timestamp_month'] <= cutoff]

    credit['dpd'] = (credit['days_late'])
    credit['missed_payment'] = (credit['final_amount_paid'] == 0).astype(int)
    credit['partial_payment'] = (credit['final_amount_paid'] < credit['expected_amount']).astype(int)
    credit['payment_ratio'] = credit['final_amount_paid'] / credit['expected_amount']
    credit['par30'] = (credit['dpd'] >= 30).astype(int)

    df = accounts.merge(credit, on=['customer_id','account_id'], how='left')

    df = df.merge(
        pmts,
        left_on=['customer_id','month'],
        right_on=['customer_id','payment_month'],
        how='left'
    )

    df = df.merge(
        iot,
        left_on=['customer_id','month'],
        right_on=['customer_id','device_timestamp_month'],
        how='left'
    )

    df = df.drop_duplicates()

    df = df.sort_values(['customer_id','month'])

    # rolling features
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
# Predict + Explain
# --------------------------
def predict_customer(model, df, customer_id, cutoff, credit):

    features = [
        'payment_ratio_1m','avg_payment_ratio_3m','avg_payment_ratio_6m',
        'missed_payments_3m','missed_payments_6m','partial_payments_3m',
        'avg_gap_3m','avg_balance_3m',
        'payment_count_3m','payment_amount_3m','days_late_last_month',
        'max_days_late_3m','max_days_late_6m'
    ]

    cust_df = df[df['customer_id'] == customer_id]

    latest = cust_df[cust_df['month'] <= cutoff].sort_values('month').tail(1).copy()

    # prediction
    latest['par30_probability'] = model.predict_proba(latest[features])[:,1]

    def segment(p):
        if p < 0.2: return "Low"
        elif p < 0.5: return "Medium"
        else: return "High"

    latest['risk_segment'] = latest['par30_probability'].apply(segment)

    # expected date
    prediction_month = cutoff + pd.offsets.MonthBegin(1)
    expected = credit[
        (credit['customer_id'] == customer_id) &
        (credit['month'] == prediction_month)
    ][['expected_date']].head(1)

    latest['prediction_month'] = prediction_month
    latest['expected_date'] = expected['expected_date'].values[0] if not expected.empty else None

    # raw data used
    raw_history = cust_df[cust_df['month'] <= cutoff].copy()

    return latest[[
        'customer_id','prediction_month','expected_date',
        'par30_probability','risk_segment'
    ]], latest[features], raw_history

# --------------------------
# Main
# --------------------------
def main():

    cutoff = pd.to_datetime(args.as_of_month).to_period('M').to_timestamp()

    print("Loading model...")
    model = joblib.load(args.model_path)

    print("Extracting data...")
    accounts, credit, pmts, iot = extract_data()

    print("Transforming...")
    df = transform(accounts, credit, pmts, iot, cutoff)

    print("Predicting customer...")
    pred, features_used, raw_data = predict_customer(
        model, df, args.customer_id, cutoff, credit
    )

    pred.to_csv("single_customer_prediction.csv", index=False)
    features_used.to_csv("single_customer_features_used.csv", index=False)
    raw_data.to_csv("single_customer_raw_history.csv", index=False)

    print("Saved:")
    print(" - prediction.csv")
    print(" - features_used.csv")
    print(" - raw_history.csv")

if __name__ == "__main__":
    main()