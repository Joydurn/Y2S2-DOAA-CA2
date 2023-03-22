#conftest.py 
#create fixtures for pytest, used in test_app.py
import pytest
from app  import app as flask_app

@pytest.fixture
def app():
   yield flask_app

@pytest.fixture
def client(app):
   print(app.static_url_path)
   return app.test_client()
