from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import numpy as np
import os
import pickle
import json

from utils.model_trainer import (
    train_all,
    load_saved,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_column_distribution,
    plot_correlation_heatmap,
    plot_class_balance,
    SCALE_COLS,
)

app = Flask(__name__)
app.secret_key = "fraud_detection_secret_2024"

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "credit_card_fraud_10k.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Global state
_models = None
_scaler = None
_feature_names = None
_results = None
_df = None


def get_df():
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)
    return _df


def models_trained():
    return os.path.exists(os.path.join(MODEL_DIR, "trained_models.pkl"))


def get_models():
    global _models, _scaler, _feature_names
    if _models is None and models_trained():
        _models, _scaler, _feature_names = load_saved()
    return _models, _scaler, _feature_names


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", trained=models_trained())


@app.route("/train", methods=["POST"])
def train():
    global _models, _scaler, _feature_names, _results
    try:
        results, models, X_test, y_test, feature_names = train_all(DATA_PATH)
        _models = models
        _feature_names = feature_names
        _results = results
        _, _scaler, _ = load_saved()

        results_serializable = {}
        for model_name, metrics in results.items():
            results_serializable[model_name] = {
                k: v for k, v in metrics.items() if k != "report"
            }

        # Persist results for dashboard
        with open(os.path.join(MODEL_DIR, "results.pkl"), "wb") as f:
            pickle.dump(results, f)

        return jsonify({"status": "success", "results": results_serializable})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/dashboard")
def dashboard():
    if not models_trained():
        return redirect(url_for("index"))

    results_path = os.path.join(MODEL_DIR, "results.pkl")
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            results = pickle.load(f)
    else:
        return redirect(url_for("index"))

    models, scaler, feature_names = get_models()

    comparison_chart = plot_model_comparison(results)
    best_model_name = max(results, key=lambda n: results[n]["f1"])
    best_cm = plot_confusion_matrix(best_model_name, results)
    feat_imp = plot_feature_importance(models, feature_names)

    df = get_df()
    class_balance = plot_class_balance(df)
    corr_heatmap = plot_correlation_heatmap(df)

    model_names = list(results.keys())

    return render_template(
        "dashboard.html",
        results=results,
        model_names=model_names,
        best_model=best_model_name,
        comparison_chart=comparison_chart,
        best_cm=best_cm,
        feat_imp=feat_imp,
        class_balance=class_balance,
        corr_heatmap=corr_heatmap,
    )


@app.route("/api/confusion_matrix/<model_name>")
def api_confusion_matrix(model_name):
    results_path = os.path.join(MODEL_DIR, "results.pkl")
    if not os.path.exists(results_path):
        return jsonify({"error": "Models not trained"}), 400
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    if model_name not in results:
        return jsonify({"error": "Model not found"}), 404
    img = plot_confusion_matrix(model_name, results)
    return jsonify({"image": img})


@app.route("/visualize")
def visualize():
    if not models_trained():
        return redirect(url_for("index"))
    df = get_df()
    columns = [c for c in df.columns if c != "transaction_id"]
    return render_template("visualize.html", columns=columns)


@app.route("/api/visualize_column", methods=["POST"])
def api_visualize_column():
    data = request.get_json()
    col = data.get("column")
    df = get_df()
    if col not in df.columns:
        return jsonify({"error": "Column not found"}), 400
    img = plot_column_distribution(df, col)
    return jsonify({"image": img})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not models_trained():
        return redirect(url_for("index"))

    models, scaler, feature_names = get_models()
    df = get_df()
    merchant_cats = sorted(df["merchant_category"].unique().tolist())

    result = None
    if request.method == "POST":
        try:
            amount = float(request.form["amount"])
            transaction_hour = int(request.form["transaction_hour"])
            merchant_category = request.form["merchant_category"]
            foreign_transaction = int(request.form["foreign_transaction"])
            location_mismatch = int(request.form["location_mismatch"])
            device_trust_score = int(request.form["device_trust_score"])
            velocity_last_24h = int(request.form["velocity_last_24h"])
            cardholder_age = int(request.form["cardholder_age"])
            selected_model = request.form.get("model_choice", "Voting Classifier")

            raw = pd.DataFrame({
                "amount": [amount],
                "transaction_hour": [transaction_hour],
                "foreign_transaction": [foreign_transaction],
                "location_mismatch": [location_mismatch],
                "device_trust_score": [device_trust_score],
                "velocity_last_24h": [velocity_last_24h],
                "cardholder_age": [cardholder_age],
            })

            # One-hot encode merchant_category to match training features
            for cat in ["Electronics", "Food", "Grocery", "Travel", "Clothing"]:
                col_name = f"merchant_category_{cat}"
                if col_name in feature_names:
                    raw[col_name] = 1 if merchant_category == cat else 0

            # Ensure all expected columns are present
            for col in feature_names:
                if col not in raw.columns:
                    raw[col] = 0
            raw = raw[feature_names]

            # Scale
            scale_present = [c for c in SCALE_COLS if c in raw.columns]
            raw[scale_present] = scaler.transform(raw[scale_present])

            model = models[selected_model]
            pred = model.predict(raw)[0]
            proba = model.predict_proba(raw)[0][1]

            result = {
                "prediction": int(pred),
                "probability": round(float(proba) * 100, 2),
                "label": "🚨 FRAUD DETECTED" if pred == 1 else "✅ LEGITIMATE",
                "model": selected_model,
            }
        except Exception as e:
            result = {"error": str(e)}

    return render_template(
        "predict.html",
        result=result,
        merchant_cats=merchant_cats,
        model_names=list(models.keys()),
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
