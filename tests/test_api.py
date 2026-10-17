from fastapi.testclient import TestClient

from api.main import app


def test_api_key_required_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY", "secret-test-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.get("/v1/workspaces/demo1/runs").status_code == 401
    assert c.get("/v1/workspaces/demo1/runs", headers={"X-API-Key": "wrong"}).status_code == 401
    ok = c.get("/v1/workspaces/demo1/runs", headers={"X-API-Key": "secret-test-key"})
    assert ok.status_code == 200 and ok.json() == []


def test_analyze_detects_cost_file_in_single_upload(monkeypatch, tmp_path):
    from pathlib import Path
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    samples = Path(__file__).resolve().parent.parent / "data" / "samples"
    c = TestClient(app)
    files = [("files", (n, (samples / n).read_bytes(), "text/csv"))
             for n in ("shopify_orders.csv", "product_costs.csv")]
    res = c.post("/v1/analyze", files=files, data={"narrative": "false"})
    assert res.status_code == 200
    body = res.json()
    assert body["meta"]["quality"]["checks"][5]["check"] == "Product costs"
    assert body["meta"]["quality"]["checks"][5]["score"] == 100  # costs joined -> margin available
    assert len(body["meta"]["sources"]) == 1  # the cost file is not treated as an order export
    only_costs = c.post("/v1/analyze", files=[files[1]], data={"narrative": "false"})
    assert only_costs.status_code == 422
