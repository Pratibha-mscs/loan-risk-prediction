from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_predict_endpoint_no_model():
    payload = {
        "annual_income": 65000,
        "loan_amount": 15000,
        "dti": 18.0,
        "int_rate": 12.0,
        "installment": 450.0,
        "emp_length_years": 5,
        "revol_util": 45.0,
        "revol_bal": 12000,
        "open_acc": 8,
        "total_acc": 20,
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
        "pub_rec": 0,
        "grade_numeric": 2,
        "sub_grade_numeric": 10,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 503]


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code in [200, 503]
