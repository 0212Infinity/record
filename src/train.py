from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data_pipeline import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_training_frame,
)


def _build_preprocessor():
    numeric_columns = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_columns),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def _make_regressor(random_state: int):
    return Pipeline(
        steps=[
            ("prep", _build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=220,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _make_classifier():
    return Pipeline(
        steps=[
            ("prep", _build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=260,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _time_split(frame: pd.DataFrame):
    unique_dates = np.array(sorted(frame["target_date"].dt.strftime("%Y-%m-%d").unique()))
    split_index = max(int(len(unique_dates) * 0.8), 1)
    split_index = min(split_index, len(unique_dates) - 1)
    cutoff = unique_dates[split_index]
    train_mask = frame["target_date"].dt.strftime("%Y-%m-%d") < cutoff
    test_mask = ~train_mask
    return frame.loc[train_mask].copy(), frame.loc[test_mask].copy(), cutoff


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_and_save(data_dir: Path, models_dir: Path) -> Dict:
    frame, quality, snapshots = build_training_frame(data_dir)
    if frame.empty:
        raise RuntimeError("No aligned training rows could be built from the weather dataset.")

    train_df, test_df, cutoff = _time_split(frame)
    if train_df.empty or test_df.empty:
        raise RuntimeError("Train/test split failed because there are not enough target dates.")

    models = {
        "tempMax": _make_regressor(11),
        "tempMin": _make_regressor(12),
        "precip": _make_regressor(13),
        "textDay": _make_classifier(),
    }

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_train = {
        "tempMax": train_df["target_tempMax"],
        "tempMin": train_df["target_tempMin"],
        "precip": train_df["target_precip"],
        "textDay": train_df["target_textDay"],
    }
    y_test = {
        "tempMax": test_df["target_tempMax"],
        "tempMin": test_df["target_tempMin"],
        "precip": test_df["target_precip"],
        "textDay": test_df["target_textDay"],
    }

    for name, model in models.items():
        model.fit(X_train, y_train[name])

    predictions = {
        "tempMax": models["tempMax"].predict(X_test),
        "tempMin": models["tempMin"].predict(X_test),
        "precip": np.clip(models["precip"].predict(X_test), 0, None),
        "textDay": models["textDay"].predict(X_test),
    }

    backtest = {
        "split_cutoff": cutoff,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "metrics": {
            "tempMax_mae": float(mean_absolute_error(y_test["tempMax"], predictions["tempMax"])),
            "tempMax_rmse": _rmse(y_test["tempMax"], predictions["tempMax"]),
            "tempMin_mae": float(mean_absolute_error(y_test["tempMin"], predictions["tempMin"])),
            "tempMin_rmse": _rmse(y_test["tempMin"], predictions["tempMin"]),
            "precip_mae": float(mean_absolute_error(y_test["precip"], predictions["precip"])),
            "precip_rmse": _rmse(y_test["precip"], predictions["precip"]),
            "textDay_accuracy": float(accuracy_score(y_test["textDay"], predictions["textDay"])),
        },
        "series": [
            {
                "target_date": row["target_date"].strftime("%Y-%m-%d"),
                "lead_day": int(row["lead_day"]),
                "actual_tempMax": float(row["target_tempMax"]),
                "pred_tempMax": float(predictions["tempMax"][index]),
                "actual_tempMin": float(row["target_tempMin"]),
                "pred_tempMin": float(predictions["tempMin"][index]),
                "actual_precip": float(row["target_precip"]),
                "pred_precip": float(predictions["precip"][index]),
                "actual_textDay": str(row["target_textDay"]),
                "pred_textDay": str(predictions["textDay"][index]),
            }
            for index, (_, row) in enumerate(test_df.iterrows())
        ],
    }

    X_full = frame[FEATURE_COLUMNS]
    full_targets = {
        "tempMax": frame["target_tempMax"],
        "tempMin": frame["target_tempMin"],
        "precip": frame["target_precip"],
        "textDay": frame["target_textDay"],
    }
    final_models = {
        "tempMax": _make_regressor(21),
        "tempMin": _make_regressor(22),
        "precip": _make_regressor(23),
        "textDay": _make_classifier(),
    }
    for name, model in final_models.items():
        model.fit(X_full, full_targets[name])

    artifact = {
        "models": final_models,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "backtest": backtest,
        "quality": quality,
        "train_summary": {
            "train_rows": int(len(frame)),
            "latest_target_date": frame["target_date"].max().strftime("%Y-%m-%d"),
            "first_target_date": frame["target_date"].min().strftime("%Y-%m-%d"),
            "snapshot_count": len(snapshots),
        },
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / "weather_ai.joblib"
    joblib.dump(artifact, artifact_path)
    return artifact


def main():
    project_root = Path(__file__).resolve().parent.parent
    artifact = train_and_save(project_root / "data", project_root / "models")
    metrics = artifact["backtest"]["metrics"]
    print("Saved models to", project_root / "models" / "weather_ai.joblib")
    print(
        "Metrics:",
        {
            "tempMax_mae": round(metrics["tempMax_mae"], 3),
            "tempMin_mae": round(metrics["tempMin_mae"], 3),
            "precip_mae": round(metrics["precip_mae"], 3),
            "textDay_accuracy": round(metrics["textDay_accuracy"], 3),
        },
    )


if __name__ == "__main__":
    main()
