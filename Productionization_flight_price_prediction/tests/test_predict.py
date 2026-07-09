import math
import os

import pandas as pd

from app import app
from predict import predict

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sample_combo():
    combos = pd.read_csv(os.path.join(CURRENT_DIR, "load_distance_time.csv"))
    return combos.iloc[0]


def test_predict_returns_finite_positive_price():
    combo = _sample_combo()
    input_data = pd.DataFrame([{
        "date": "2024-08-20",
        "from": combo["from"],
        "to": combo["to"],
        "flighttype": combo["flighttype"],
        "agency": combo["agency"],
    }])

    result = predict(input_data)
    price = float(result[0])

    assert math.isfinite(price)
    assert price > 0


def test_flask_root_returns_200():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
