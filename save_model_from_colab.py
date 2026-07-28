"""
Add this as a NEW CELL at the end of your Colab notebook, after your models
are trained. It saves the model plus everything the web app needs to know
about it, then downloads the file to your computer.

Run it once. It produces violent_crime_model.joblib.
"""

import joblib
import sklearn
import numpy as np
from datetime import date
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------
# 1. Choose which trained model and which feature list to publish.
#
# TODO: after you remove the violent-crime arrest rates (the leakage
# fix), retrain and point these two names at the corrected model and
# corrected feature list. The web app reads the feature list from this
# file, so its input form updates automatically. You do not have to
# edit the app.
# ---------------------------------------------------------------
model_to_save = models_b["Gradient Boosting"]
features_to_save = features_b
model_name = "Gradient Boosting"

# ---------------------------------------------------------------
# 2. Record how the model scored, so the app can display it honestly.
# ---------------------------------------------------------------
pred = model_to_save.predict(X_test_b)
metrics = {
    "r2": float(r2_score(y_test_b, pred)),
    "mae": float(mean_absolute_error(y_test_b, pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_test_b, pred))),
}

# ---------------------------------------------------------------
# 3. Record the range of each predictor in the training data.
#    The app uses these to prefill the form and to warn the user when
#    an input falls outside the range the model actually learned from.
# ---------------------------------------------------------------
feature_stats = {}
for col in features_to_save:
    series = panel[col].astype(float)
    feature_stats[col] = {
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
    }

# ---------------------------------------------------------------
# 4. Bundle everything together and save.
#    Saving the feature list with the model is important. It guarantees
#    the app sends columns in the same order the model was trained on.
# ---------------------------------------------------------------
bundle = {
    "model": model_to_save,
    "model_name": model_name,
    "features": features_to_save,
    "target": "violent_crime_rate",
    "metrics": metrics,
    "feature_stats": feature_stats,
    "sklearn_version": sklearn.__version__,
    "saved_on": str(date.today()),
}

joblib.dump(bundle, "violent_crime_model.joblib")

print("Saved violent_crime_model.joblib")
print("Model:", model_name)
print("Features:", len(features_to_save))
print("Metrics:", {k: round(v, 3) for k, v in metrics.items()})
print("scikit-learn version:", sklearn.__version__)

# ---------------------------------------------------------------
# 5. Download the file from Colab to your computer.
# ---------------------------------------------------------------
from google.colab import files
files.download("violent_crime_model.joblib")
