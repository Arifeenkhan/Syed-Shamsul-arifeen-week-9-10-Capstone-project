from fastapi.testclient import TestClient
from app.main import app
def test_health():
 with TestClient(app) as c: response=c.get('/health')
 assert response.status_code==200 and 'model_loaded' in response.json()
def test_blank_comment():
 with TestClient(app) as c: response=c.post('/predict',json={'text':''})
 assert response.status_code==422
