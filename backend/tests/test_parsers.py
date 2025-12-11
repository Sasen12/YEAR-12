import io
from app.utils.parsers import parse_file_to_questions

def test_parse_json():
    data = b'[{"question_text":"Q1","possible_answers":[{"answer_text":"A","is_correct":true}]}]'
    res = parse_file_to_questions(data, 'questions.json')
    assert isinstance(res, list)
    assert res[0]['question_text'] == 'Q1'

def test_parse_csv():
    csv = b'question,answers,correct\nWhat is X?,A|B|C,A\n'
    res = parse_file_to_questions(csv, 'q.csv')
    assert res[0]['question_text'].startswith('What')

def test_parse_txt_empty_sections():
    txt = b'\n\n'
    res = parse_file_to_questions(txt, 'q.txt')
    assert res == []
