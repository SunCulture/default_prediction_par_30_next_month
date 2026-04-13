CREATE TABLE par30_predictions (
    customer_id BIGINT,
    account_type TEXT,
    current_account_status TEXT,
    prediction_month DATE,
    expected_date DATE,
    par30_probability FLOAT,
    risk_segment TEXT,
    created_at TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (customer_id, prediction_month)
);