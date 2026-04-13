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

### 2. Create and Activate a Virtual Environment

#### Linux / WSL
```bash
python -m venv venv
source venv/bin/activate
```

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

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

## 🧠 Running Predictions

Run the prediction script using:

```bash
python predict.py --as_of_month 2026-03-01 --model_path xgb_par30_model.pkl
```

### Parameters:
- `--as_of_month`: The cutoff date for input data (YYYY-MM-DD)
- `--model_path`: Path to the trained model file (`.pkl`)

---

## 📁 Output

- A new **CSV file** containing predictions will be generated after execution.
- Each row corresponds to a customer with their predicted likelihood of defaulting (PAR30).

---

## 📝 Notes
- Ensure database credentials are correct before running the script.
- The model uses **historical data up to the specified `as_of_month`** to predict the following month.
