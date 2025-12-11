from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_quiz_and_goals_flow():
    # ensure demo user exists from seed migration
    login = client.post('/auth/login', json={'username':'demo','password':'demo'})
    # login may fail if seed password not known; register a user
    if login.status_code != 200:
        client.post('/auth/register', json={'username':'demo','password':'demo'})
        login = client.post('/auth/login', json={'username':'demo','password':'demo'})
    assert login.status_code == 200
    token = login.json()['access_token']
    # fetch quiz
    headers = {'Authorization': f'Bearer {token}'}
    r = client.get('/softwaredev/quiz', headers=headers)
    assert r.status_code == 200
    data = r.json()
    # if there are no questions, ensure endpoint handles zero
    if not data:
        # import a simple question via file upload
        files = {'file': ('q.txt', b'What is A?\nAnswer1\nAnswer2')}
        client.post('/softwaredev/import', files=files, headers=headers)
        r = client.get('/softwaredev/quiz', headers=headers)
        data = r.json()
    # submit grading
    answers = []
    for q in data[:2]:
        # pick first answer as given and include answer_id for stricter matching
        if q['answers']:
            first = q['answers'][0]
            answers.append({'question_id': q['id'], 'given_answer': first['answer_text'], 'answer_id': first['id']})
        else:
            answers.append({'question_id': q['id'], 'given_answer': 'NA'})
    payload = {'answers': answers}
    g = client.post('/softwaredev/grade', json=payload, headers=headers)
    assert g.status_code == 200
    # set goal
    s = client.post('/goals/set', params={'week_start':'2025-12-01', 'goal_type':'quizzes', 'goal_value':5}, headers=headers)
    assert s.status_code == 200
    p = client.get('/goals/progress', params={'week_start':'2025-12-01'}, headers=headers)
    assert p.status_code == 200
