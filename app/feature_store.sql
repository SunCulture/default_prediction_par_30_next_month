CREATE TABLE feature_store_par30 (
    customer_id BIGINT,
    account_id BIGINT,
    month DATE,

    -- Features
    payment_ratio_1m FLOAT,
    avg_payment_ratio_3m FLOAT,
    avg_payment_ratio_6m FLOAT,
    missed_payments_3m FLOAT,
    missed_payments_6m FLOAT,
    partial_payments_3m FLOAT,
    avg_gap_3m FLOAT,
    avg_balance_3m FLOAT,
    payment_count_3m FLOAT,
    payment_amount_3m FLOAT,
    days_late_last_month FLOAT,
    max_days_late_3m FLOAT,
    max_days_late_6m FLOAT

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (customer_id, account_id)
);