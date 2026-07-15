# Student Performance Predictor

**Author:** Mehr Hussain
**Type:** Machine Learning AI Lab Project (Streamlit web app)

## What This App Does

Predicts whether a student will **pass** or is **at risk of failing**, based on
study hours, attendance, previous exam score, sleep hours, and whether they
take extra classes. Two ML models (Logistic Regression and Random Forest) are
trained on the same data and compared side by side, so the app also
demonstrates model evaluation, not just prediction.

## Project Structure

```
StudentPerformancePredictor_MehrHussain/
├── app.py                          # UI (Streamlit) — layout, input, display only
├── model.py                        # Core AI/ML logic — fully separate from UI
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── report.docx                     # Short report (problem, method, results)
├── data/
│   ├── generate_data.py            # Reproducible dataset generator (seeded)
│   └── student_performance.csv     # Sample dataset used by the app (400 rows)
└── screenshots/                    # Add your screenshots here before submitting
```

## How to Run

**1. Install Python** (3.9 or newer) if you don't already have it.

**2. Open a terminal in this folder and install dependencies:**
```
pip install -r requirements.txt
```

**3. (Optional) Regenerate the dataset** — not required, `student_performance.csv`
is already included:
```
cd data
python generate_data.py
cd ..
```

**4. Run the app:**
```
streamlit run app.py
```

**5. Your browser opens automatically** at `http://localhost:8501`. If it
doesn't, open that link manually.

## How to Check It's Working (step by step)

1. Run the command above. You should see terminal output ending in
   `Local URL: http://localhost:8501` with no red error text.
2. In the browser tab that opens, confirm you see the title **"Student
   Performance Predictor"** and the sentence stating the dataset was loaded
   (e.g. "Dataset loaded: 400 students, pass rate 79.5%").
3. Tick **"Show raw dataset"** — a table of student rows should appear. If
   this works, file loading is confirmed working.
4. Move the sliders under **"Enter Student Details"** to some values (e.g.
   Study hours = 6, Attendance = 90, Previous score = 80, Sleep = 7, Extra
   classes = ON).
5. Click **Run Prediction**. Within a second you should see:
   - A green "Prediction complete" message
   - A **PASS** or **AT RISK OF FAILING** result with a confidence percentage
   - A short plain-language explanation of which factors drove the result
   - A horizontal bar chart of feature importance next to it
6. Scroll down to **"Model Evaluation & Comparison"** — you should see a
   table comparing Logistic Regression vs Random Forest on Accuracy,
   Precision, Recall, and F1 Score, plus a confusion matrix for whichever
   model is currently selected.
7. Try switching the model dropdown between "Random Forest" and "Logistic
   Regression" and re-running a prediction — the result and explanation
   should update accordingly. This confirms both models are functioning
   independently.

If all 7 steps above work without a red error banner, the app is fully
functional and ready to submit.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `FileNotFoundError` for the CSV | Make sure you're running `streamlit run app.py` from inside this project folder, not from `data/` |
| Browser doesn't open | Manually visit `http://localhost:8501` |
| Port already in use | Run `streamlit run app.py --server.port 8502` instead |

## AI Components Used

- **Logistic Regression** and **Random Forest Classifier** (scikit-learn) —
  no external API calls, fully local and free.
- Feature scaling via `StandardScaler`.
- Explainability via model coefficients (Logistic Regression) and feature
  importances (Random Forest).
