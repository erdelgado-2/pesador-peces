# -*- coding: utf-8 -*-
def test_health(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.get_json().get("status") == "OK"

def test_predict_single_instance(client):
    payload = [{"Species": "Bream","Length1": 38,"Length2": 35.0,"Length3": 40.5,"Height": 10,"Width": 5.589}]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "results" in data and isinstance(data["results"], list) and len(data["results"]) == 1
    item = data["results"][0]
    assert "prediction" in item

