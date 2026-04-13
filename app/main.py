from fastapi import FastAPI, BackgroundTasks, HTTPException
import pandas as pd
import joblib
from app.db import get_postgres_conn
from app.batch import run_batch, load_model
from app.train import main
from pydantic import BaseModel


app = FastAPI()
model, features = load_model()
features = [
        'payment_ratio_1m','avg_payment_ratio_3m','avg_payment_ratio_6m',
        'missed_payments_3m','missed_payments_6m','partial_payments_3m',
        'avg_gap_3m','avg_balance_3m','payment_count_3m','payment_amount_3m',
        'days_late_last_month', 'max_days_late_3m','max_days_late_6m']  # same as train features

@app.get("/")
def home():
    return {"status": "PAR30 API running"}


@app.get("/predict_all_from_db")
def predict_all_customers():
    conn = get_postgres_conn()
    df = pd.read_sql("SELECT * FROM data_science.par30_predictions", conn)
    latest = df.groupby('customer_id').tail(1)
    return latest[['customer_id','risk_segment']].to_dict(orient='records')

@app.get("/predict_customer_from_db/{customer_id}")
def predict_single(customer_id: int):
    conn = get_postgres_conn()
    df = pd.read_sql(f"SELECT * FROM data_science.par30_predictions WHERE customer_id={customer_id}", conn)
    if df.empty:
        return {"error":"Customer not found"}
    df = df.tail(1)
    return df[['customer_id','risk_segment']].to_dict(orient='records')


@app.post("/predict/all/{as_of_month}")
def predict_all(as_of_month: str):

    preds = run_batch(as_of_month)

    return {
            "status": "Finished",
            "message": "Prediction Done"
        }


@app.get("/predict/customer/{customer_id}/{as_of_month}")
def predict_customer(customer_id: int, as_of_month: str):

    preds = run_batch(as_of_month)
    customer = preds[preds['customer_id'] == customer_id]

    if customer.empty:
        return {"message": "Customer not found"}

    return customer.to_dict(orient="records")


@app.post("/train/{as_of_month}")
def train_endpoint(as_of_month: str, background_tasks: BackgroundTasks):
    try:
        # Trigger async training
        background_tasks.add_task(main, as_of_month)

        return {
            "status": "started",
            "message": "Training started"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))