# default_prediction_par_30_next_month

To predict
clone the repository
git clone https://github.com/SunCulture/default_prediction_par_30_next_month.git

create a virtual environment - Linux or WSL instructions
python -m venv venv
source venv/bin/activate

install requirements
pip install -r requirements.txt

add a .env file with these credentials
POSTGRES_USER=""
POSTGRES_PASSWORD=""
POSTGRES_HOST=""
POSTGRES_PORT=5432
POSTGRES_DB="reporting-service"
CLICKHOUSE_HOST=""
CLICKHOUSE_PORT=8443
CLICKHOUSE_USER=""
CLICKHOUSE_PASSWORD=""


run this command;
python predict.py --as_of_month 2026-03-01 --model_path xgb_par30_model.pkl

a new csv file with the predictions will be generated.