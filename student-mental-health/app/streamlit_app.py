from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.pipeline import MODEL_PATH, evaluate_model, load_processed_data


st.set_page_config(page_title="Student Mental Health Risk", layout="wide")
st.title("Student Mental Health Risk")
st.warning(
    "Educational prototype only. This is not a medical diagnosis or a tool "
    "for making decisions about real students."
)


@st.cache_resource
def load_model():
    import joblib

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_results():
    _, X_test, _, y_test = load_processed_data()
    model = load_model()
    metrics, matrix = evaluate_model(model, X_test, y_test)
    return X_test, y_test, metrics, matrix


if not Path(MODEL_PATH).exists():
    st.error("The model is missing. Run the Airflow DAG or src/pipeline.py first.")
    st.stop()

X_test, y_test, metrics, matrix = load_results()
model = load_model()

st.header("Model performance")
metric_columns = st.columns(5)
for column, (name, value) in zip(metric_columns, metrics.items()):
    column.metric(name.replace("_", " ").upper(), f"{value:.3f}")

st.subheader("Confusion matrix")
figure, axis = plt.subplots(figsize=(5, 4))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_,
    ax=axis,
)
axis.set_xlabel("Predicted")
axis.set_ylabel("Actual")
st.pyplot(figure)
plt.close(figure)

st.header("Try a new prediction")
st.write("Enter values using the same prepared columns used during training.")

with st.form("prediction_form"):
    values = {}
    columns = st.columns(2)
    for index, name in enumerate(X_test.columns):
        default = float(X_test[name].median())
        values[name] = columns[index % 2].number_input(
            name,
            value=default,
            format="%.3f",
        )
    submitted = st.form_submit_button("Predict")

if submitted:
    sample = pd.DataFrame([values], columns=X_test.columns)
    prediction = model.predict(sample)[0]
    st.success(f"Predicted risk category: {prediction}")