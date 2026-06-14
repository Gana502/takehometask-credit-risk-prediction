from fastapi.testclient import TestClient
from api.app import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_predict():
    payload = {
        "customer_id": "CUST_0001",
        "txn_count": 3,
        "total_debit": -65.98,
        "total_credit": 2500.0,
        "avg_amount": 811.34,
        "kw_rent": 0,
        "kw_netflix": 1,
        "kw_tesco": 1,
        "kw_payroll": 1,
        "kw_bonus": 0,
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

        assert response.status_code == 200

        body = response.json()

        assert body["customer_id"] == "CUST_0001"
        assert "probability" in body
        assert "prediction" in body
        assert isinstance(body["probability"], float)
        assert body["prediction"] in [0, 1]