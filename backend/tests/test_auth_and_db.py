import pytest
from fastapi.testclient import TestClient
from app.main import app
from pathlib import Path
import os

client = TestClient(app)

def test_register_login_and_fetch_questions():
    # register
    r = client.post('/auth/register', json={'username':'testuser','password':'pass123'})
    assert r.status_code == 200
    # login
    r2 = client.post('/auth/login', json={'username':'testuser','password':'pass123'})
    assert r2.status_code == 200
    assert 'access_token' in r2.json()
    token = r2.json()['access_token']
    # fetch questions (public)
    r3 = client.get('/softwaredev/questions')
    assert r3.status_code == 200
    # test protected import endpoint rejects missing token
    files = {'file': ('q.txt', b'Q?\nA1\nA2')}
    r4 = client.post('/softwaredev/import', files=files)
    assert r4.status_code == 403 or r4.status_code == 401
    # test import with token
    headers = {'Authorization': f'Bearer {token}'}
    r5 = client.post('/softwaredev/import', files=files, headers=headers)
    assert r5.status_code == 200
