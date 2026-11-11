from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline import evaluate_model, load_processed_data, save_model, train_model


def prepare_data():
    """Check that Notebook 02 created the processed data files."""
    load_processed_data()
    print("Processed data is ready.")


def train():
    """Train and select the best model."""
    X_train, _, y_train, _ = load_processed_data()
    best_name, model, searches = train_model(X_train, y_train)
    save_model(model)
    print(f"Selected model: {best_name}")
    print({name: search.best_score_ for name, search in searches.items()})


def evaluate():
    """Evaluate the saved training result on the independent test data."""
    X_train, X_test, y_train, y_test = load_processed_data()
    _, model, _ = train_model(X_train, y_train)
    metrics, matrix = evaluate_model(model, X_test, y_test)
    print("Metrics:", metrics)
    print("Confusion matrix:\n", matrix)


with DAG(
    dag_id="student_mental_health_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
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