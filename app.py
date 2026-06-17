from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "knn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "outputs", "model_report.txt")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class_names = ["Setosa", "Versicolor", "Virginica"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r", encoding="utf-8") as file:
            report_text = file.read()
    else:
        report_text = "Model report not found. Please run python train_model.py first."

    if request.method == "POST":
        try:
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = scaler.transform(input_data)

            result = model.predict(input_scaled)[0]
            prediction = class_names[result]

        except ValueError:
            prediction = "Please enter valid numeric values."

    return render_template(
        "index.html",
        prediction=prediction,
        report_text=report_text
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8000, use_reloader=False)