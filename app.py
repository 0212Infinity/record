import socket
from pathlib import Path

from flask import Flask, jsonify, render_template

from src.predict import generate_dashboard_payload, load_or_train_artifacts


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)
_cached_payload = None


def get_payload(force_refresh: bool = False):
    global _cached_payload
    if force_refresh or _cached_payload is None:
        artifacts = load_or_train_artifacts(DATA_DIR, MODELS_DIR)
        _cached_payload = generate_dashboard_payload(DATA_DIR, artifacts)
    return _cached_payload


def pick_port(host: str, start_port: int = 3000, attempts: int = 20) -> int:
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + attempts - 1}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/summary")
def summary():
    return jsonify(get_payload()["summary"])


@app.route("/api/predictions")
def predictions():
    return jsonify(get_payload()["predictions"])


@app.route("/api/backtest")
def backtest():
    return jsonify(get_payload()["backtest"])


@app.route("/api/quality")
def quality():
    return jsonify(get_payload()["quality"])


@app.route("/api/refresh")
def refresh():
    payload = get_payload(force_refresh=True)
    return jsonify(
        {
            "status": "ok",
            "latestForecastDate": payload["summary"]["latest_forecast_date"],
            "trainRows": payload["summary"]["train_rows"],
        }
    )


if __name__ == "__main__":
    get_payload()
    host = "127.0.0.1"
    port = pick_port(host)
    print(f"Serving Guangzhou Weather AI at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
