import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = "/opt/airflow/project"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}


def run_batch_predictions():
    from Productionization_flight_price_prediction.predict import batch_predictions

    batch_predictions()


with DAG(
    dag_id="flight_price_batch_predictions",
    default_args=default_args,
    description="Score all known flight routes for the current day and upsert predictions.csv",
    schedule="*/10 * * * *",  # demo cadence; switch to "@daily" for production
    start_date=datetime(2024, 8, 15),
    catchup=False,
    tags=["flight-price", "batch-scoring"],
) as dag:
    run_predictions = PythonOperator(
        task_id="run_batch_predictions",
        python_callable=run_batch_predictions,
    )
