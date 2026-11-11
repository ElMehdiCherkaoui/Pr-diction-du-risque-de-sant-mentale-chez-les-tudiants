from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline import (
    evaluate_model,
    load_model,
    load_processed_data,
    preprocess_data,
    save_model,
    train_model,
)


def prepare_data():
    preprocess_data()
    print("Raw data was preprocessed and saved.")


def train():
    X_train, _, y_train, _ = load_processed_data()
    best_name, model = train_model(X_train, y_train)
    save_model(model)
    print(f"Selected model: {best_name}")


def evaluate():
    _, X_test, _, y_test = load_processed_data()
    model = load_model()
    metrics, matrix = evaluate_model(model, X_test, y_test)
    print("Metrics:", metrics)
    print("Confusion matrix:\n", matrix)


with DAG(
    dag_id="student_mental_health_pipeline",
    start_date=datetime(2026, 11, 11),
    schedule="@daily",
    catchup=False,
    tags=["student-mental-health"],
) as dag:
    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
    )
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train,
    )
    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate,
    )

    prepare >> train_task >> evaluate_task