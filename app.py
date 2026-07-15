"""
app.py -- Visual UI Module
Streamlit front end. All AI logic lives in model.py; this file only
handles layout, input, and display, per the "keep code modular" requirement.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from model import (
    load_data, preprocess_data, run_model_or_algorithm,
    evaluate_model, generate_explanation, create_visuals_data, FEATURES
)

st.set_page_config(page_title="Student Performance Predictor", page_icon=":bar_chart:", layout="wide")

st.title("Student Performance Predictor")
st.write("**Built by Mehr Hussain** | AI Lab Project | Machine Learning")
st.write(
    "Predicts whether a student will pass or is at risk of failing, based on "
    "study habits and academic history. Two models are trained and compared."
)
st.divider()

# ---------------------------------------------------------------------------
# CORE LOGIC -- run once, cached so the UI stays fast
# ---------------------------------------------------------------------------
DATA_PATH = "data/student_performance.csv"

@st.cache_data
def get_data():
    return load_data(DATA_PATH)

try:
    df = get_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# ---------------------------------------------------------------------------
# A) PROBLEM SETUP MODULE
# ---------------------------------------------------------------------------
st.header("1. Problem Setup")
col1, col2 = st.columns([1, 1])

with col1:
    st.write(f"Dataset loaded: **{len(df)} students**, pass rate **{df['passed'].mean()*100:.1f}%**")
    if st.checkbox("Show raw dataset"):
        st.dataframe(df, use_container_width=True)

with col2:
    model_choice = st.selectbox(
        "Choose model to use for prediction",
        ["Random Forest", "Logistic Regression"]
    )

model = run_model_or_algorithm(X_train, y_train, model_choice)
eval_results = evaluate_model(model, X_test, y_test)

st.divider()

# ---------------------------------------------------------------------------
# INPUT FORM -- with validation (Problem Setup requirement)
# ---------------------------------------------------------------------------
st.header("2. Enter Student Details")

c1, c2, c3 = st.columns(3)
with c1:
    study_hours = st.slider("Daily study hours", 0.0, 10.0, 4.0, 0.5)
    attendance = st.slider("Attendance %", 30.0, 100.0, 78.0, 1.0)
with c2:
    previous_score = st.slider("Previous exam score", 20.0, 100.0, 65.0, 1.0)
    sleep_hours = st.slider("Sleep hours", 3.0, 10.0, 6.5, 0.5)
with c3:
    extra_classes = st.toggle("Taking extra classes?", value=False)
    run_button = st.button("Run Prediction", type="primary", use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# B) CORE LOGIC + C) RESULT PANEL + D) EXPLAINABILITY MODULE
# ---------------------------------------------------------------------------
if run_button:
    # Validation -- status messages requirement
    if study_hours < 0 or attendance < 0:
        st.error("Invalid input: values cannot be negative.")
        st.stop()

    with st.spinner("Running model..."):
        input_row = np.array([[study_hours, attendance, previous_score,
                                sleep_hours, int(extra_classes)]])
        input_scaled = scaler.transform(
            pd.DataFrame(input_row, columns=FEATURES)
        )
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

    st.success("Prediction complete")

    st.header("3. Result")
    result_col, chart_col = st.columns([1, 1.3])

    with result_col:
        if prediction == 1:
            st.markdown(f"### PASS")
            st.metric("Confidence", f"{probability*100:.1f}%")
        else:
            st.markdown(f"### AT RISK OF FAILING")
            st.metric("Confidence", f"{(1-probability)*100:.1f}%")

        explanation, importances = generate_explanation(
            model, model_choice, input_scaled, prediction
        )
        st.text(explanation)

    with chart_col:
        st.write("**Explainability: what drove this prediction**")
        labels, values = create_visuals_data(importances)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(labels, values, color="#4f8ef7")
        ax.set_xlabel("Influence on prediction")
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

# ---------------------------------------------------------------------------
# E) EVALUATION MODULE -- metrics + compare two models
# ---------------------------------------------------------------------------
st.header("4. Model Evaluation & Comparison")
st.write("Both models are trained on the same data split so results are directly comparable.")

results_table = []
for m_type in ["Logistic Regression", "Random Forest"]:
    m = run_model_or_algorithm(X_train, y_train, m_type)
    r = evaluate_model(m, X_test, y_test)
    results_table.append({
        "Model": m_type,
        "Accuracy": f"{r['accuracy']*100:.1f}%",
        "Precision": f"{r['precision']*100:.1f}%",
        "Recall": f"{r['recall']*100:.1f}%",
        "F1 Score": f"{r['f1_score']*100:.1f}%",
    })

st.table(pd.DataFrame(results_table))

st.write(f"**Confusion Matrix — currently selected model ({model_choice})**")
cm = eval_results["confusion_matrix"]
fig2, ax2 = plt.subplots(figsize=(4, 3.5))
ax2.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Fail", "Pass"])
ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Fail", "Pass"])
ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
st.pyplot(fig2)
plt.close(fig2)

st.divider()
st.caption("Built with Python, scikit-learn, Pandas, Matplotlib, and Streamlit.")
