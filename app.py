"""
Violent Crime Rate Predictor
Flask app serving the model trained in the DATA 975 capstone notebook.

Run locally:
    python app.py
Then open http://127.0.0.1:5000
"""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = Path("model/violent_crime_model.joblib")

# ---------------------------------------------------------------
# Load the model bundle once, when the app starts, not on every
# request. Loading a model is slow, so doing it per request would
# make the app much slower.
# ---------------------------------------------------------------
BUNDLE = None
LOAD_ERROR = None

try:
    BUNDLE = joblib.load(MODEL_PATH)
except Exception as exc:                      # noqa: BLE001
    LOAD_ERROR = str(exc)


def bundle_or_none():
    return BUNDLE


def pretty(name):
    """Turn median_income into Median income for display."""
    return name.replace("_", " ").capitalize()


def field_specs():
    """
    Build the list of input fields from the model bundle.

    The form is generated from the saved feature list, so if the feature
    set changes in the notebook (for example after removing the leaky
    arrest-rate columns), the form updates automatically. Nothing here
    needs to be edited by hand.
    """
    if BUNDLE is None:
        return []

    stats = BUNDLE.get("feature_stats", {})
    specs = []
    for feat in BUNDLE["features"]:
        s = stats.get(feat, {})
        specs.append({
            "name": feat,
            "label": pretty(feat),
            "default": s.get("median", 0),
            "min": s.get("min"),
            "max": s.get("max"),
        })
    return specs


def read_inputs(source):
    """
    Pull one value per feature out of a form or JSON payload.

    Returns (dataframe, errors, warnings). The dataframe has exactly the
    columns the model was trained on, in the same order, which is what
    scikit-learn expects.
    """
    if BUNDLE is None:
        return None, ["Model is not loaded."], []

    stats = BUNDLE.get("feature_stats", {})
    values, errors, warnings = {}, [], []

    for feat in BUNDLE["features"]:
        raw = source.get(feat)

        if raw is None or str(raw).strip() == "":
            errors.append(f"Missing value for {pretty(feat)}.")
            continue

        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{pretty(feat)} must be a number.")
            continue

        # Warn when an input sits outside the range the model was trained
        # on. The model can still return a number, but predicting far
        # outside the training range is extrapolation and is unreliable.
        s = stats.get(feat, {})
        if s.get("min") is not None and s.get("max") is not None:
            if val < s["min"] or val > s["max"]:
                warnings.append(
                    f"{pretty(feat)} is outside the training range "
                    f"({s['min']:.4g} to {s['max']:.4g}). The prediction "
                    f"is an extrapolation."
                )

        values[feat] = val

    if errors:
        return None, errors, warnings

    # Column order must match the training order exactly.
    X = pd.DataFrame([[values[f] for f in BUNDLE["features"]]],
                     columns=BUNDLE["features"])
    return X, [], warnings


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        fields=field_specs(),
        bundle=bundle_or_none(),
        load_error=LOAD_ERROR,
        prediction=None,
        errors=[],
        warnings=[],
        submitted={},
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle the HTML form."""
    X, errors, warnings = read_inputs(request.form)

    prediction = None
    if not errors:
        prediction = float(BUNDLE["model"].predict(X)[0])

    return render_template(
        "index.html",
        fields=field_specs(),
        bundle=bundle_or_none(),
        load_error=LOAD_ERROR,
        prediction=prediction,
        errors=errors,
        warnings=warnings,
        submitted=request.form.to_dict(),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON endpoint, so the model can be called from other programs.

    Example:
        curl -X POST http://127.0.0.1:5000/api/predict \
             -H "Content-Type: application/json" \
             -d '{"gini_index": 0.47, "poverty_rate": 14.2, ...}'
    """
    payload = request.get_json(silent=True) or {}
    X, errors, warnings = read_inputs(payload)

    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    pred = float(BUNDLE["model"].predict(X)[0])
    return jsonify({
        "ok": True,
        "prediction": round(pred, 2),
        "units": "violent crimes per 100,000 population",
        "warnings": warnings,
        "model": BUNDLE.get("model_name", "unknown"),
    })


@app.route("/health")
def health():
    """A simple status check. Useful once the app is hosted."""
    if BUNDLE is None:
        return jsonify({"status": "error", "detail": LOAD_ERROR}), 500
    return jsonify({
        "status": "ok",
        "model": BUNDLE.get("model_name", "unknown"),
        "n_features": len(BUNDLE["features"]),
        "trained_with_sklearn": BUNDLE.get("sklearn_version", "unknown"),
    })


if __name__ == "__main__":
    # debug=True reloads the app when you edit the code. Turn it off
    # before deploying anywhere public.
    app.run(debug=True, port=5000)
