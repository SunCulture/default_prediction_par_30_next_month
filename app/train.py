import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from app.utils import compute_features
from app.db import get_postgres_conn, get_clickhouse_client
import pandas as pd


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

def train_model(df):
    features = [
        'payment_ratio_1m','avg_payment_ratio_3m','avg_payment_ratio_6m',
        'missed_payments_3m','missed_payments_6m','partial_payments_3m',
        'avg_gap_3m','avg_balance_3m','payment_count_3m','payment_amount_3m',
        'days_late_last_month','max_days_late_3m','max_days_late_6m'
    ]
    model_df = df.dropna(subset=['par30_next_month'])
    train = model_df[(model_df['month'] >= '2023-01-01') & (model_df['month'] < '2025-09-01')]
    val = model_df[(model_df['month'] >= '2025-09-01') & (model_df['month'] <= '2025-11-30')]

    X_train, y_train = train[features], train['par30_next_month']
    X_val, y_val = val[features], val['par30_next_month']

    model = XGBClassifier(
        objective= 'binary:logistic',
        scale_pos_weight = (len(y_train[y_train==0]) / len(y_train[y_train==1])) * 0.75,
        n_estimators=250,
        max_depth=3,
        min_child_weight=10,
        gamma=1,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="error",
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    print("Validation AUC:", roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))
    return model, features

def save_model(model, features, path="models/xgb_par30_model.pkl"):
    joblib.dump({"model": model,"features": features}, path)


def main(cutoff):
    print("Extracting data for training...")
    accounts, credit, pmts, iot = extract_data()

    print("Computing features...")
    df = compute_features(accounts, credit, pmts, iot, cutoff)

    print("Training model...")
    model, features = train_model(df)

    print("Saving model...")
    save_model(model, features)

    print("Done training.")

if __name__ == "__main__":
    main()