
"""
CodeNova Task 3: House Price Prediction
Dataset: Ames Housing (OpenML)
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


# --------------------------------------------------
# 2. Load the public dataset
# --------------------------------------------------

def load_dataset():
    print("Loading Ames Housing dataset from OpenML...")

    housing = fetch_openml(
        name="house_prices",
        version=1,
        as_frame=True,
    )

    X = housing.data.copy()
    y = pd.to_numeric(housing.target, errors="coerce")

    # Identifier is not a useful house characteristic.
    X = X.drop(columns=["Id"], errors="ignore")

    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    print(f"Dataset rows: {len(X)}")
    print(f"Input features: {X.shape[1]}")

    return X, y


# --------------------------------------------------
# 3. Split data before fitting any preprocessing
# --------------------------------------------------

def split_dataset(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


# --------------------------------------------------
# 4. Preprocessing pipeline
# --------------------------------------------------

def build_preprocessor(X_train):
    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return preprocessor


# --------------------------------------------------
# 5. Evaluation metrics
# --------------------------------------------------

def calculate_metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
    }


# --------------------------------------------------
# 6. Save evaluation charts
# --------------------------------------------------

def save_charts(y_test, predictions, model):
    # Actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.65)

    minimum = min(float(y_test.min()), float(predictions.min()))
    maximum = max(float(y_test.max()), float(predictions.max()))

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )
    plt.xlabel("Actual Sale Price ($)")
    plt.ylabel("Predicted Sale Price ($)")
    plt.title("Actual vs Predicted House Prices")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Residual plot
    residuals = np.asarray(y_test) - np.asarray(predictions)

    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals, alpha=0.65)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Predicted Sale Price ($)")
    plt.ylabel("Residual ($)")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_plot.png", dpi=150)
    plt.close()

    # Top 15 Random Forest feature importances
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(15)

    importance_df = importance_df.sort_values("importance")

    plt.figure(figsize=(10, 7))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Feature Importance")
    plt.title("Top 15 House Price Predictors")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()

    importance_df.sort_values(
        "importance", ascending=False
    ).to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)


# --------------------------------------------------
# 7. Main training and evaluation workflow
# --------------------------------------------------

def main():
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # Baseline: predict the median training price for every house.
    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    # Preprocessing is part of the pipeline.
    # During cross-validation, it is fitted separately inside each fold.
    preprocessor = build_preprocessor(X_train)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    print("\nRunning 5-fold cross-validation on training data...")

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=1,
        return_train_score=False,
    )

    cv_summary = {
        "CV_MAE_mean": -cv_results["test_mae"].mean(),
        "CV_MAE_std": cv_results["test_mae"].std(),
        "CV_RMSE_mean": -cv_results["test_rmse"].mean(),
        "CV_RMSE_std": cv_results["test_rmse"].std(),
        "CV_R2_mean": cv_results["test_r2"].mean(),
        "CV_R2_std": cv_results["test_r2"].std(),
    }

    pd.DataFrame([cv_summary]).to_csv(
        OUTPUT_DIR / "cross_validation_metrics.csv",
        index=False,
    )

    print("\nTraining final Random Forest model...")
    model.fit(X_train, y_train)
    model_predictions = model.predict(X_test)

    baseline_metrics = calculate_metrics(y_test, baseline_predictions)
    model_metrics = calculate_metrics(y_test, model_predictions)

    results = pd.DataFrame([
        {"Model": "Median Baseline", **baseline_metrics},
        {"Model": "Random Forest", **model_metrics},
    ])

    results.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    pd.DataFrame({
        "Actual_SalePrice": y_test.to_numpy(),
        "Baseline_Prediction": baseline_predictions,
        "RandomForest_Prediction": model_predictions,
        "Residual": y_test.to_numpy() - model_predictions,
    }).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    joblib.dump(model, MODEL_DIR / "house_price_pipeline.joblib")

    save_charts(y_test, model_predictions, model)

    print("\n=== Test-set Model Comparison ===")
    print(results.to_string(index=False))

    print("\n=== 5-fold Cross-validation (training set only) ===")
    for metric, value in cv_summary.items():
        print(f"{metric}: {value:,.4f}")

    mae_gain = (
        (baseline_metrics["MAE"] - model_metrics["MAE"])
        / baseline_metrics["MAE"]
        * 100
        if baseline_metrics["MAE"] != 0
        else 0
    )

    print(f"\nMAE improvement over baseline: {mae_gain:.2f}%")
    if mae_gain > 0:
        print("The Random Forest reduced test-set MAE versus the baseline.")
    else:
        print("The Random Forest did not improve test-set MAE versus the baseline.")

    print("\nSaved model: models/house_price_pipeline.joblib")
    print("Saved metrics and charts in the outputs/ folder.")
    print("Training and evaluation completed.")


if __name__ == "__main__":
    main()
