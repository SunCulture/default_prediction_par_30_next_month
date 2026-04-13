import pandas as pd

def compute_features(accounts, credit, pmts, iot, cutoff):
    credit = credit[credit['month'] <= cutoff]
    pmts = pmts[pmts['payment_month'] <= cutoff]
    iot = iot[iot['device_timestamp_month'] <= cutoff]

    credit['dpd'] = credit['days_late']
    credit['missed_payment'] = (credit['final_amount_paid'] == 0).astype(int)
    credit['partial_payment'] = ((credit['final_amount_paid'] < credit['expected_amount']) & (credit['final_amount_paid'] != 0)).astype(int)
    credit['payment_ratio'] = credit['final_amount_paid'] / credit['expected_amount']
    credit['par30'] = (credit['dpd'] >= 30).astype(int)

    df = accounts.merge(credit, on=['customer_id','account_id'], how='left')
    df = df.merge(pmts, left_on=['customer_id','month'], right_on=['customer_id','payment_month'], how='left')
    df = df.merge(iot, left_on=['customer_id','month'], right_on=['customer_id','device_timestamp_month'], how='left')
    df = df.sort_values(['customer_id','month'])
    df = df.drop_duplicates().sort_values(['customer_id','month'])

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
    df['max_days_late_3m'] = df.groupby('customer_id')['days_late'].shift(1).rolling(3).max()
    df['max_days_late_6m'] = df.groupby('customer_id')['days_late'].shift(1).rolling(6).max()
    df = df.dropna(subset=['par30_next_month'])

    return df