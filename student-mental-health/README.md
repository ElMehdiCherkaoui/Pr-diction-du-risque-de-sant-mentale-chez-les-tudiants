# Student Mental Health Risk

This is a simple educational machine-learning project. It predicts `Low`,
`Medium`, or `High` risk from the prepared student data.

It is not a medical diagnosis tool and must not be used to make decisions
about real students.

## Run the pipeline

From the project folder:

```bash
python src/pipeline.py
```

The pipeline reads the current file in `data/raw/`, preprocesses it, saves
new train/test CSV files in `data/processed/`, trains two models, selects the
best one, evaluates it, and saves the model in `models/`.

If you add new labelled rows to the raw CSV, run the pipeline again. The
processed data and model will be rebuilt using the new rows.

## Run Streamlit

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`.

## Run with Docker

```bash
docker compose up --build
```

The application also uses port `8501`.

## Airflow

Copy or mount `airflow/dags/mental_health_pipeline.py` into the Airflow DAGs
folder. It runs these tasks in order:

1. Prepare raw data and save processed data
2. Train and save the model
3. Evaluate the model

The DAG is scheduled weekly and does not catch up old runs.