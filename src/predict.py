from pathlib import Path

import joblib

from .data_pipeline import build_latest_forecast_frame, build_training_frame
from .train import train_and_save


def load_or_train_artifacts(data_dir: Path, models_dir: Path):
    artifact_path = models_dir / "weather_ai.joblib"
    if artifact_path.exists():
        return joblib.load(artifact_path)
    return train_and_save(data_dir, models_dir)


def generate_dashboard_payload(data_dir: Path, artifacts: dict):
    frame, quality, snapshots = build_training_frame(data_dir)
    latest_frame, latest_snapshot = build_latest_forecast_frame(snapshots)
    models = artifacts["models"]

    pred_temp_max = models["tempMax"].predict(latest_frame).tolist()
    pred_temp_min = models["tempMin"].predict(latest_frame).tolist()
    pred_precip = [max(0.0, value) for value in models["precip"].predict(latest_frame).tolist()]
    pred_text = models["textDay"].predict(latest_frame).tolist()

    prediction_rows = []
    for index, row in latest_frame.iterrows():
        raw_day = latest_snapshot.daily[index]
        prediction_rows.append(
            {
                "forecast_date": row["forecast_date"].strftime("%Y-%m-%d"),
                "lead_day": int(row["lead_day"]),
                "raw": {
                    "tempMax": float(row["forecast_tempMax"]),
                    "tempMin": float(row["forecast_tempMin"]),
                    "precip": float(row["forecast_precip"]),
                    "textDay": raw_day.get("textDay", "未知"),
                    "humidity": float(row["forecast_humidity"]),
                },
                "predicted": {
                    "tempMax": round(float(pred_temp_max[index]), 1),
                    "tempMin": round(float(pred_temp_min[index]), 1),
                    "precip": round(float(pred_precip[index]), 1),
                    "textDay": str(pred_text[index]),
                    "humidity": float(row["forecast_humidity"]),
                },
            }
        )

    summary = {
        "city": "广州",
        "source": "QWeather snapshot history",
        "data_start": artifacts["train_summary"]["first_target_date"],
        "data_end": artifacts["train_summary"]["latest_target_date"],
        "latest_forecast_date": latest_snapshot.fetch_date,
        "latest_update_time": latest_snapshot.update_time,
        "valid_snapshots": quality["valid_snapshots"],
        "invalid_snapshots": quality["invalid_snapshots"],
        "train_rows": artifacts["train_summary"]["train_rows"],
        "fx_link": latest_snapshot.fx_link,
    }

    quality_payload = {
        "total_files": quality["total_files"],
        "valid_snapshots": quality["valid_snapshots"],
        "invalid_snapshots": quality["invalid_snapshots"],
        "invalid_examples": quality["issues"][:25],
        "unresolved_targets": len(quality["unresolved_targets"]),
    }

    return {
        "summary": summary,
        "predictions": {
            "latest_fetch_date": latest_snapshot.fetch_date,
            "latest_update_time": latest_snapshot.update_time,
            "days": prediction_rows,
        },
        "backtest": artifacts["backtest"],
        "quality": quality_payload,
    }
