from fastapi.testclient import TestClient

from app import api


def test_start_returns_immediately_and_validates_weights(monkeypatch):
    key = "00000000-0000-0000-0000-000000000001"
    requests = []
    monkeypatch.setattr(api, "create_screening", lambda request: requests.append(request) or key)
    client = TestClient(api.app)
    response = client.post("/screenings", json={"jd_text": "Python developer required"})
    assert response.status_code == 202 and response.json()["jd_id"] == key
    assert len(requests) == 1
    scoped = client.post("/screenings", json={"jd_text": "Python developer required", "resume_ids": [key]})
    assert scoped.status_code == 202
    assert requests[-1].model_dump(mode="json")["resume_ids"] == [key]
    assert client.post("/screenings", json={"jd_text": "Python developer required", "resume_ids": []}).status_code == 422
    assert client.post("/screenings", json={"jd_text": "Python developer required", "resume_ids": ["invalid"]}).status_code == 422
    assert client.post("/screenings", json={"jd_text": "Python developer", "weights": {"proficiency": 1, "domain": 1}}).status_code == 422


def test_upload_isolates_bad_files(monkeypatch):
    def upload(content, name):
        if name.endswith(".txt"):
            raise ValueError("Only PDF and DOCX files are supported")
        return {"id": "test", "status": "uploaded"}

    monkeypatch.setattr(api, "process_upload", upload)
    response = TestClient(api.app).post("/resumes", files=[("files", ("bad.txt", b"bad")), ("files", ("good.pdf", b"test"))])
    assert [row["status"] for row in response.json()] == ["error", "uploaded"]
