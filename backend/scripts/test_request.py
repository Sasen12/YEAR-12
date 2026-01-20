"""Run a quick test against the app.

Tries to use FastAPI's TestClient if available; otherwise calls the
`health()` controller directly as a fallback (no httpx required).
"""

import sys
import os

# Ensure backend folder is on sys.path so `app` package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from fastapi.testclient import TestClient
    from app.main import app
    def run_testclient():
        client = TestClient(app)
        resp = client.get('/health')
        print('STATUS:', resp.status_code)
        try:
            print('JSON:', resp.json())
        except Exception:
            print('CONTENT:', resp.text)
    run_testclient()
except Exception:
    # Fallback: call the controller function directly (works for simple endpoints)
    try:
        from app.main import health
        print('Direct call to health():', health())
    except Exception as e:
        print('Failed to run TestClient and direct call fallback also failed:', e)
