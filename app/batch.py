from app.db import get_postgres_conn, get_clickhouse_client
from app.utils import compute_features
from app.train import train_model, save_model
import pandas as pd
import numpy as np
import joblib
import argparse
import psycopg2.extras as extras

def parse_args():
    parser = argparse.ArgumentParser(description="Monthly PAR30 Batch Pipeline")
    parser.add_argument(
        "--as_of_month",
        required=True,
        help="Cutoff month in YYYY-MM-DD format (e.g. 2026-03-01)"
    )
    return parser.parse_args()


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

    return accounts, credit, pmts, iot

def save_features_to_postgres(df):
    conn = get_postgres_conn()
    cursor = conn.cursor()
    df = df.replace({np.nan: None})
    df = df.where(pd.notnull(df), None)
    # Clear table before insert
    cursor.execute("TRUNCATE TABLE data_science.feature_store_par30;")
    # Convert dataframe to list of tuples
    tuples = [tuple(x) for x in df.to_numpy()]
    cols = ",".join(df.columns)
    query = f"""
        INSERT INTO data_science.feature_store_par30 ({cols})
        VALUES %s
    """
    extras.execute_values(cursor, query, tuples)

    conn.commit()
    cursor.close()
    conn.close()

def save_predictions_to_postgres(df):
    conn = get_postgres_conn()
    cursor = conn.cursor()
    df = df.replace({np.nan: None})
    df = df.where(pd.notnull(df), None)
    cursor.execute("TRUNCATE TABLE data_science.par30_predictions;")
    tuples = [tuple(x) for x in df.to_numpy()]
    cols = ",".join(df.columns)
    query = f"""
        INSERT INTO data_science.par30_predictions ({cols})
        VALUES %s
    """
    extras.execute_values(cursor, query, tuples)

    conn.commit()
    cursor.close()
    conn.close()

def load_model(path="models/xgb_par30_model.pkl"):
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]

def run_batch(as_of_month):
    accounts, credit, pmts, iot = extract_data()
    print("Computing features...")
    df = compute_features(accounts, credit, pmts, iot, as_of_month)

    print("Loading model...")
    model, features = load_model()

    # Optional: compute predictions for next month and store
    print("Preparing prediction dataset...")
    def segment(p):
        if p < 0.3: return "Low"
        elif p < 0.6: return "Medium"
        else: return "High"

    prediction_month = pd.to_datetime(as_of_month) + pd.offsets.MonthBegin(1)
    df = df[(df['accountType'] == "PAYG") & df['current_account_status'].isin(["Advance", "Current"])]
    latest = df[df['month'] <= as_of_month].sort_values(['customer_id','month']).groupby('customer_id').tail(1)
    print("Saving features...")
    save_features_to_postgres(latest[['customer_id', 'account_id', 'month',
        'payment_ratio_1m','avg_payment_ratio_3m','avg_payment_ratio_6m',
        'missed_payments_3m','missed_payments_6m','partial_payments_3m',
        'avg_gap_3m','avg_balance_3m','payment_count_3m','payment_amount_3m',
        'days_late_last_month','max_days_late_3m','max_days_late_6m'
    ]])
    print("Predicting...")
    latest['par30_probability'] = model.predict_proba(latest[features])[:,1]
    latest['risk_segment'] = latest['par30_probability'].apply(segment)
    expected_dates = credit[credit['month'] == prediction_month].sort_values(['customer_id','expected_date']).groupby('customer_id').first().reset_index()
    latest = latest.drop(['expected_date'], axis=1).merge(expected_dates[['customer_id','expected_date']], on='customer_id', how='left')
    latest['prediction_month'] = prediction_month
    latest = latest[['customer_id', 'account_type','current_account_status', 'prediction_month','expected_date','par30_probability','risk_segment']]
    print("Storing predictions...")
    save_predictions_to_postgres(latest)
    print("Batch run complete.")
    return latest


if __name__ == "__main__":
    args = parse_args()
    run_batch(args.as_of_month)