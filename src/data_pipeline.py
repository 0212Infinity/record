import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ENCODINGS = ("utf-8", "utf-8-sig", "gb18030")


NUMERIC_FIELDS = [
    "tempMax",
    "tempMin",
    "precip",
    "humidity",
    "pressure",
    "vis",
    "cloud",
    "uvIndex",
    "iconDay",
    "wind360Day",
    "wind360Night",
    "windSpeedDay",
    "windSpeedNight",
]

FEATURE_COLUMNS = [
    "lead_day",
    "forecast_tempMax",
    "forecast_tempMin",
    "forecast_precip",
    "forecast_humidity",
    "forecast_pressure",
    "forecast_vis",
    "forecast_cloud",
    "forecast_uvIndex",
    "forecast_iconDay",
    "forecast_wind360Day",
    "forecast_wind360Night",
    "forecast_windSpeedDay",
    "forecast_windSpeedNight",
    "forecast_month",
    "forecast_dayofyear",
    "forecast_is_weekend",
    "forecast_textDay",
    "forecast_windDirDay",
    "forecast_windScaleDay",
]

TARGET_COLUMNS = ["target_tempMax", "target_tempMin", "target_precip", "target_textDay"]
CATEGORICAL_COLUMNS = ["forecast_textDay", "forecast_windDirDay", "forecast_windScaleDay"]


@dataclass
class Snapshot:
    fetch_date: str
    update_time: str
    fx_link: str
    daily: List[dict]
    path: str


def _read_json_with_fallback(path: Path):
    raw = path.read_bytes()
    if not raw:
        return None, "empty_file"
    for encoding in ENCODINGS:
        try:
            return json.loads(raw.decode(encoding)), None
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError:
            return None, "invalid_json"
    return None, "decode_error"


def load_snapshots(data_dir: Path) -> Tuple[List[Snapshot], List[dict], Dict[str, Snapshot]]:
    valid: List[Snapshot] = []
    issues: List[dict] = []

    for path in sorted(data_dir.rglob("*.json")):
        obj, error = _read_json_with_fallback(path)
        if error:
            issues.append(
                {"fetch_date": path.stem, "path": str(path), "reason": error, "daily_count": 0}
            )
            continue

        daily = obj.get("daily") or []
        if obj.get("code") != "200" or len(daily) != 7:
            reason = "non_200_code" if obj.get("code") != "200" else "unexpected_daily_count"
            issues.append(
                {
                    "fetch_date": path.stem,
                    "path": str(path),
                    "reason": reason,
                    "daily_count": len(daily),
                }
            )
            continue

        valid.append(
            Snapshot(
                fetch_date=path.stem,
                update_time=obj.get("updateTime", ""),
                fx_link=obj.get("fxLink", ""),
                daily=daily,
                path=str(path),
            )
        )

    snapshot_map = {snapshot.fetch_date: snapshot for snapshot in valid}
    return valid, issues, snapshot_map


def _to_number(value, default=0.0):
    if value in ("", None):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_feature_row(snapshot: Snapshot, lead_day: int, item: dict) -> dict:
    forecast_date = pd.to_datetime(item["fxDate"])
    row = {
        "fetch_date": snapshot.fetch_date,
        "forecast_date": item["fxDate"],
        "lead_day": int(lead_day),
        "forecast_textDay": item.get("textDay", "未知"),
        "forecast_windDirDay": item.get("windDirDay", "未知"),
        "forecast_windScaleDay": item.get("windScaleDay", "未知"),
        "forecast_month": int(forecast_date.month),
        "forecast_dayofyear": int(forecast_date.dayofyear),
        "forecast_is_weekend": int(forecast_date.dayofweek >= 5),
    }
    for field in NUMERIC_FIELDS:
        row[f"forecast_{field}"] = _to_number(item.get(field))
    return row


def build_training_frame(data_dir: Path):
    snapshots, issues, snapshot_map = load_snapshots(data_dir)
    rows = []
    unresolved_targets = []

    for snapshot in snapshots:
        for lead_day, item in enumerate(snapshot.daily):
            feature_row = _extract_feature_row(snapshot, lead_day, item)
            target_key = item["fxDate"].replace("-", "")
            target_snapshot = snapshot_map.get(target_key)
            if target_snapshot is None:
                unresolved_targets.append(
                    {
                        "fetch_date": snapshot.fetch_date,
                        "forecast_date": item["fxDate"],
                        "reason": "missing_target_snapshot",
                    }
                )
                continue

            target_day = target_snapshot.daily[0]
            feature_row.update(
                {
                    "target_date": target_day["fxDate"],
                    "target_tempMax": _to_number(target_day.get("tempMax")),
                    "target_tempMin": _to_number(target_day.get("tempMin")),
                    "target_precip": _to_number(target_day.get("precip")),
                    "target_textDay": target_day.get("textDay", "未知"),
                }
            )
            rows.append(feature_row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["fetch_date"] = pd.to_datetime(frame["fetch_date"], format="%Y%m%d")
        frame["target_date"] = pd.to_datetime(frame["target_date"])
        frame["forecast_date"] = pd.to_datetime(frame["forecast_date"])
        frame = frame.sort_values(["target_date", "lead_day", "fetch_date"]).reset_index(drop=True)

    quality = {
        "total_files": len(list(data_dir.rglob("*.json"))),
        "valid_snapshots": len(snapshots),
        "invalid_snapshots": len(issues),
        "issues": issues,
        "unresolved_targets": unresolved_targets,
    }
    return frame, quality, snapshots


def build_latest_forecast_frame(snapshots: List[Snapshot]) -> Tuple[pd.DataFrame, Snapshot]:
    latest_snapshot = max(snapshots, key=lambda item: item.fetch_date)
    rows = [
        _extract_feature_row(latest_snapshot, lead_day, item)
        for lead_day, item in enumerate(latest_snapshot.daily)
    ]
    frame = pd.DataFrame(rows)
    frame["fetch_date"] = pd.to_datetime(frame["fetch_date"], format="%Y%m%d")
    frame["forecast_date"] = pd.to_datetime(frame["forecast_date"])
    return frame, latest_snapshot
