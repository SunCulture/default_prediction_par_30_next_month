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
               companyRegion, createdAt AS account_created_at,
                phoneNumber AS phone_number
        FROM marts.accounts_mv
    """)
    accounts = accounts[accounts['companyRegion'] == 'kenya']
    accounts['customer_id'] = accounts['customer_id'].astype(int)
    accounts = accounts.sort_values(['customer_id','updatedAt'], ascending=[True, False])
    accounts = accounts.drop_duplicates('customer_id')
    accounts = accounts.drop(['updatedAt','companyRegion'], axis=1)

    # Credit
    credit = ch.query_df("""
         SELECT customerId AS customer_id, accountId AS account_id, accountType AS account_type, installmentType AS installment_type,
                expectedDate AS expected_date, expectedAmount AS expected_amount, finalAmountPaid AS final_amount_paid,
                finalPaidDate AS final_paid_date, amountDue AS amount_due, daysLate AS days_late, total_balance
        FROM credit_score_model.agg_credit_history_v1
    """)
    credit['expected_date'] = pd.to_datetime(credit['expected_date'])
    credit['final_paid_date'] = pd.to_datetime(credit['final_paid_date'])
    credit['month'] = credit['expected_date'].dt.to_period('M').dt.to_timestamp()
    credit['expected_amount'] = credit['expected_amount'].astype(float)
    credit['final_amount_paid'] = credit['final_amount_paid'].astype(float)

    # Payments
    pmts = ch.query_df("""
        SELECT country, customerId AS customer_id, paymentMonth AS payment_month,
                customerType AS customer_type, totalPaymentCount AS total_payment_count,
                totalPaymentAmount AS total_payment_amount
        FROM credit_score_model.agg_monthly_payments_v1
        WHERE country='kenya'
    """)
    pmts['payment_month'] = pd.to_datetime(pmts['payment_month'])
    pmts = pmts.drop(['country','customer_type'], axis=1, errors='ignore')

    # IOT
    iot = ch.query_df("""
        SELECT country, deviceTimestampMonth AS device_timestamp_month, customerId AS customer_id,
                deviceIdCount AS device_id_count, accountIdCount AS account_id_count, daysWithData AS days_with_data,
                totalTimeIntervalMins AS total_time_interval_mins, avgTimeIntervalMins AS avg_time_interval_mins,
                avgEnergyConsumptionKwh AS avg_energy_consumption_kwh, totalEnergyConsumptionKwh AS total_energy_consumption_kwh,
                totalTelemetryRecordCount AS total_telemetry_record_count
        FROM credit_score_model.agg_device_telemetry_v1
        WHERE country='kenya'
    """)
    iot['device_timestamp_month'] = pd.to_datetime(iot['device_timestamp_month'])
    iot = iot.drop(['country','sync_timestamp','total_telemetry_record_count',
                    'device_id_count','account_id_count'], axis=1, errors='ignore')

    return accounts, credit, pmts, iot

def save_features_to_postgres(df):
    client = get_clickhouse_client()

    # Clean NaNs for ClickHouse compatibility
    df = df.replace({np.nan: None})
    df = df.where(pd.notnull(df), None)

    # OPTIONAL: clear table (only if you want full overwrite like Postgres TRUNCATE)
    # client.command("TRUNCATE TABLE credit_score_model.feature_store_par30")

    # Insert dataframe
    client.insert_df(
        'credit_score_model.feature_store_par30',
        df
    )

    client.close()

def save_predictions_to_postgres(df):
    client = get_clickhouse_client()
    # Replace NaNs with None (ClickHouse handles None as NULL if column is Nullable)
    df = df.replace({np.nan: None})
    df = df.where(pd.notnull(df), None)

    # Optional: truncate table (only if you want full overwrite)
    # client.command("TRUNCATE TABLE credit_score_model.par30_predictions")

    # Insert DataFrame directly
    client.insert_df(
        'credit_score_model.par30_predictions',
        df
    )

    client.close()
    

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
    def suggest_action(p):
        if p < 0.3: return "Soft SMS Reminder"
        elif p < 0.6: return "SMS Reminder"
        else: return "Follow Up Gentle Call"

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
    latest['action'] = latest['par30_probability'].apply(suggest_action)
    expected_dates = credit[credit['month'] == prediction_month].sort_values(['customer_id','expected_date']).groupby('customer_id').first().reset_index()
    latest = latest.drop(['expected_date'], axis=1).merge(expected_dates[['customer_id','expected_date']], on='customer_id', how='left')
    latest['prediction_month'] = prediction_month
    latest = latest[['customer_id', 'phone_number', 'account_type','current_account_status', 'prediction_month','expected_date','par30_probability','risk_segment', 'action']]
    print("Storing predictions...")
    save_predictions_to_postgres(latest)
    print("Batch run complete.")
    return latest


if __name__ == "__main__":
    args = parse_args()
    run_batch(args.as_of_month)