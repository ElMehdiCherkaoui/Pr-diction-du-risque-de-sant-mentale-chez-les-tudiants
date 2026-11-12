
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from src.pipeline import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    evaluate_model,
    load_model,
    load_processed_data,
)

st.title("Student Mental Health Risk")

@st.cache_data
def load_model_cached():
    return load_model()
def load_processed_data_cached():
    return load_processed_data()

model = load_model_cached()

X_train, X_test, y_train, y_test =  load_processed_data_cached()

st.caption(f"Loaded Airflow model: {MODEL_PATH.name}")

st.header("Model Performance")

metrics, cm = evaluate_model(model, X_test, y_test)

st.write(metrics)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(plt.gcf())
plt.close()

st.header("Make a Prediction")

profiles = {
    "Low-risk example": {
        "Age": 22, "Year_of_Study": 3, "Sleep_Hours": 8.0,
        "Study_Hours_Per_Week": 15.0, "GPA": 3.5,
        "Social_Media_Hours": 2.0, "Screen_Time_Hours": 4.0,
        "Financial_Stress": 2.0, "Family_Support": 9.0,
        "Attendance_Rate": 90.0, "Gender": "Female",
        "Physical_Activity": "High", "Part_Time_Job": "No",
        "Diet_Quality": "Good",
        "Previous_Mental_Health_History": "No"
    },
    "Medium-risk example": {
        "Age": 21, "Year_of_Study": 2, "Sleep_Hours": 5.0,
        "Study_Hours_Per_Week": 20.0, "GPA": 3.0,
        "Social_Media_Hours": 6.0, "Screen_Time_Hours": 8.5,
        "Financial_Stress": 4.0, "Family_Support": 7.0,
        "Attendance_Rate": 60.0, "Gender": "Female",
        "Physical_Activity": "Moderate", "Part_Time_Job": "Yes",
        "Diet_Quality": "Average",
        "Previous_Mental_Health_History": "No"
    },
    "High-risk example": {
        "Age": 21, "Year_of_Study": 2, "Sleep_Hours": 2.0,
        "Study_Hours_Per_Week": 70.0, "GPA": 1.5,
        "Social_Media_Hours": 6.0, "Screen_Time_Hours": 8.5,
        "Financial_Stress": 10.0, "Family_Support": 1.0,
        "Attendance_Rate": 30.0, "Gender": "Female",
        "Physical_Activity": "Low", "Part_Time_Job": "Yes",
        "Diet_Quality": "Poor",
        "Previous_Mental_Health_History": "Yes"
    }
}

profile = st.selectbox("Choose a test profile", list(profiles))
selected = profiles[profile]

values = {}

values["Age"] = st.number_input("Age", 15, 60, selected["Age"])
values["Year_of_Study"] = st.number_input("Year of Study", 1, 6, selected["Year_of_Study"])
values["Sleep_Hours"] = st.number_input("Sleep Hours", 0.0, 24.0, selected["Sleep_Hours"])
values["Study_Hours_Per_Week"] = st.number_input("Study Hours Per Week", 0.0, 168.0, selected["Study_Hours_Per_Week"])
values["GPA"] = st.number_input("GPA", 0.0, 4.0, selected["GPA"])
values["Social_Media_Hours"] = st.number_input("Social Media Hours", 0.0, 24.0, selected["Social_Media_Hours"])
values["Screen_Time_Hours"] = st.number_input("Screen Time Hours", 0.0, 24.0, selected["Screen_Time_Hours"])
values["Financial_Stress"] = st.number_input("Financial Stress", 0.0, 10.0, selected["Financial_Stress"])
values["Family_Support"] = st.number_input("Family Support", 0.0, 10.0, selected["Family_Support"])
values["Attendance_Rate"] = st.number_input("Attendance Rate", 0.0, 100.0, selected["Attendance_Rate"])

options = {
    "Gender": ["Female", "Male", "Non-binary"],
    "Physical_Activity": ["High", "Moderate", "Low"],
    "Part_Time_Job": ["Yes", "No"],
    "Diet_Quality": ["Good", "Average", "Poor"],
    "Previous_Mental_Health_History": ["Yes", "No"]
}

for col, choices in options.items():
    values[col] = st.selectbox(
        col.replace("_", " "),
        choices,
        index=choices.index(selected[col])
    )

if st.button("Predict"):

    preprocessing = joblib.load(PREPROCESSOR_PATH)

    numeric = preprocessing["numeric"]
    categorical = preprocessing["categorical"]

    sample = pd.DataFrame([values])

    numeric_cols = numeric.feature_names_in_
    categorical_cols = categorical.feature_names_in_

    X_numeric = numeric.transform(sample[numeric_cols])
    X_categorical = categorical.transform(sample[categorical_cols])

    numeric_df = pd.DataFrame(
        X_numeric,
        columns=numeric_cols
    )

    categorical_df = pd.DataFrame(
        X_categorical.toarray(),
        columns=categorical.named_steps["encoder"].get_feature_names_out(categorical_cols)
    )

    X_new = pd.concat([numeric_df, categorical_df], axis=1)
    X_new = X_new[X_test.columns]

    prediction = model.predict(X_new)[0]

    st.success(f"Predicted Risk: {prediction}")
    st.caption("Educational prediction only, not a medical diagnosis.")
