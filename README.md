# Default Prediction – PAR30 (Next Month)

This project predicts whether a customer will default (reach PAR30) in the next month using a trained machine learning model.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/SunCulture/default_prediction_par_30_next_month.git
cd default_prediction_par_30_next_month
```

---

### 2. Configure Environment Variables

Create a `.env` file in the root directory and add the following:

```env
POSTGRES_USER=""
POSTGRES_PASSWORD=""
POSTGRES_HOST=""
POSTGRES_PORT=5432
POSTGRES_DB="reporting-service"

CLICKHOUSE_HOST=""
CLICKHOUSE_PORT=8443
CLICKHOUSE_USER=""
CLICKHOUSE_PASSWORD=""
```

---

### 3. Run Docker

#### Linux / WSL
```bash
docker compose up -d
```

---


## 🧠 Running Predictions

Run the prediction script in a different terminal using:

```bash
curl -X POST "http://localhost:8000/predict/all/{as_of_month}"
```

### Parameters:
- `--as_of_month`: The cutoff date for input data (YYYY-MM-DD)

For example to predict for April 2026 the command run was;

```bash
curl -X POST "http://localhost:8000/predict/all/2026-03-01"
```
---

## 📁 Output

- The predictions and updated features will be saved to the reporting-service database in postgres.
- Each row corresponds to a customer with their predicted likelihood of defaulting (PAR30).
- This should update the PowerBI Dashboard with the lates predictions

---

## 📝 Notes
- Ensure database credentials are correct before running the script.
- The model uses **historical data up to the specified `as_of_month`** to predict the following month.
