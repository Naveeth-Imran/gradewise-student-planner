
from pathlib import Path
import csv
import io
import pickle
import sqlite3
import random

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import (
    Flask, render_template, request,
    redirect, url_for, send_file, flash
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)
app.secret_key = "student-project-local-key"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_model.pkl"
DATA_PATH = BASE_DIR / "students.csv"
DB_PATH = BASE_DIR / "predictions.db"
CHART_PATH = BASE_DIR / "static" / "performance.png"

# Reuse the existing trained model. Do not retrain it automatically.
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "student_model.pkl not found. Run train_model.py first."
    )

with MODEL_PATH.open("rb") as file:
    model = pickle.load(file)


def get_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                hours REAL NOT NULL,
                predicted_marks REAL NOT NULL,
                target_marks REAL NOT NULL
            )
        """)


initialize_database()


def get_history(limit=100):
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return rows


def evaluate_models():
    """Evaluate algorithms on a held-out portion of the CSV data."""
    if not DATA_PATH.exists():
        return []

    data = pd.read_csv(DATA_PATH)

    if not {"Hours", "Marks"}.issubset(data.columns):
        return []

    data = data[["Hours", "Marks"]].dropna()
    data = data[
        data["Hours"].between(0, 24)
        & data["Marks"].between(0, 100)
    ]

    if len(data) < 5 or data["Hours"].nunique() < 2:
        return []

    X = data[["Hours"]]
    y = data["Marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    algorithms = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=3, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=4, random_state=42
        )
    }

    results = []

    for name, estimator in algorithms.items():
        estimator.fit(X_train, y_train)
        predicted = estimator.predict(X_test)

        results.append({
            "name": name,
            "mae": round(mean_absolute_error(y_test, predicted), 2),
            "r2": round(r2_score(y_test, predicted), 3)
        })

    return results


def create_chart():
    if not DATA_PATH.exists():
        return False

    data = pd.read_csv(DATA_PATH)

    if not {"Hours", "Marks"}.issubset(data.columns):
        return False

    data = data[["Hours", "Marks"]].dropna()
    if len(data) < 2:
        return False

    data = data[data["Hours"].between(0, 24)]
    if data.empty:
        return False

    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)

    x_values = pd.DataFrame({
        "Hours": sorted(data["Hours"].unique())
    })

    plt.figure(figsize=(7, 4))
    plt.scatter(
        data["Hours"], data["Marks"],
        label="Dataset marks"
    )
    plt.plot(
        x_values["Hours"],
        model.predict(x_values),
        label="Existing model",
        marker="o"
    )
    plt.xlabel("Study hours per day")
    plt.ylabel("Marks")
    plt.title("Study Hours vs Predicted Marks")
    plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=140)
    plt.close()
    return True


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    hours = ""
    target = 85
    gap = None
    advice = None
    error = None

    if request.method == "POST":
        hours = request.form.get("hours", "").strip()
        target_text = request.form.get("target", "85").strip()

        try:
            study_hours = float(hours)
            target_value = float(target_text)

            if not 0 <= study_hours <= 24:
                raise ValueError("Study hours must be between 0 and 24.")

            if not 0 <= target_value <= 100:
                raise ValueError("Target marks must be between 0 and 100.")

            raw_prediction = float(
                model.predict(pd.DataFrame({"Hours": [study_hours]}))[0]
            )

            prediction = round(max(0, min(100, raw_prediction)), 2)
            target = target_value
            gap = round(target_value - prediction, 2)

            if prediction >= 80:
                advice = "Good estimated result. Keep practising and revising."
            elif prediction >= 60:
                advice = "Review difficult topics and practise more questions."
            else:
                advice = "Build a regular study routine and revise fundamentals."

            with get_db() as db:
                db.execute(
                    """INSERT INTO predictions
                    (hours, predicted_marks, target_marks)
                    VALUES (?, ?, ?)""",
                    (study_hours, prediction, target_value)
                )

        except (ValueError, TypeError, OverflowError):
            error = (
                "Enter valid numbers: study hours from 0 to 24 "
                "and target marks from 0 to 100."
            )

    create_chart()

    return render_template(
        "index.html",
        prediction=prediction,
        hours=hours,
        target=target,
        gap=gap,
        advice=advice,
        error=error,
        history=get_history(5),
        models=evaluate_models(),
        chart_exists=CHART_PATH.exists()
    )


@app.route("/planner", methods=["GET", "POST"])
def planner():
    plan = None
    available_hours = 2

    if request.method == "POST":
        try:
            available_hours = float(
                request.form.get("available_hours", "2")
            )
            subjects_text = request.form.get("subjects", "")

            subjects = [
                subject.strip()
                for subject in subjects_text.split(",")
                if subject.strip()
            ]

            if not 0.5 <= available_hours <= 12:
                raise ValueError

            if not subjects:
                raise ValueError

            if len(subjects) > 14:
                raise ValueError

            daily_slots = []
            total_slots = max(1, int(available_hours * 2))
            rotation = subjects[:]

            for index in range(total_slots):
                subject = rotation[index % len(rotation)]
                daily_slots.append({
                    "time": f"{index * 30:02d}-{(index + 1) * 30:02d} min",
                    "subject": subject
                })

            plan = daily_slots

        except (ValueError, TypeError):
            flash(
                "Enter 0.5–12 study hours and 1–14 comma-separated subjects."
            )

    return render_template(
        "planner.html",
        plan=plan,
        available_hours=available_hours
    )


@app.route("/history")
def history():
    return render_template("history.html", history=get_history(100))


@app.route("/download-history")
def download_history():
    rows = get_history(10000)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Created At", "Study Hours",
        "Predicted Marks", "Target Marks"
    ])

    for row in rows:
        writer.writerow([
            row["id"], row["created_at"], row["hours"],
            row["predicted_marks"], row["target_marks"]
        ])

    memory_file = io.BytesIO(output.getvalue().encode("utf-8-sig"))
    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype="text/csv",
        as_attachment=True,
        download_name="student_prediction_history.csv"
    )


@app.route("/performance")
def performance():
    return render_template(
        "performance.html",
        models=evaluate_models(),
        chart_exists=create_chart()
    )


if __name__ == "__main__":
    app.run(debug=True)
