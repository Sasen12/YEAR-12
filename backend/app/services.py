"""Business logic services used by HTTP controllers.

This module holds small service classes that coordinate repositories,
parsers and auxiliary logic. Services are intentionally thin: they
perform validation, execute domain logic and persist aggregates via
repositories.
"""

from datetime import date, datetime, timedelta
from passlib.context import CryptContext
import jwt
import os
from typing import List, Optional
from . import models, repositories
from sqlmodel import Session
from .utils.parsers import parse_file_to_questions
from .utils.report_parser import find_report_explanation
from .utils.answer_keys import get_correct_answers_from_key
from pathlib import Path

PWD_CTX = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
JWT_SECRET = os.getenv('JWT_SECRET', 'change_me_for_prod')
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
JWT_EXPIRE_HOURS = int(os.getenv('JWT_EXPIRE_HOURS', '24'))


class AuthService:
    """Authentication related operations (register + authenticate)."""
    def __init__(self, session: Session):
        self.session = session
        self.user_repo = repositories.UserRepository(session)

    def register(self, username: str, password: str) -> models.User:
        """Create a new user with a hashed password.

        Returns the persisted `User` instance.
        """
        hashed = PWD_CTX.hash(password)
        u = models.User(username=username, password_hash=hashed)
        return self.user_repo.create(u)

    def authenticate(self, username: str, password: str):
        """Verify credentials and return a signed JWT token on success.

        Returns `None` if authentication fails.
        """
        # One lookup by username then verify the supplied password hash.
        user = self.user_repo.get_by_username(username)
        if not user:
            return None
        if not PWD_CTX.verify(password, user.password_hash):
            return None
        expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
        payload = {"user_id": user.id, "username": user.username, "exp": int(expire.timestamp())}
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        return token


class ImportService:
    """Import questions from files and persist them to the DB."""
    def __init__(self, session: Session):
        self.session = session
        self.q_repo = repositories.QuestionRepository(session)
        self.a_repo = repositories.AnswerRepository(session)

    def import_file(self, file_bytes: bytes, filename: str, subject: str = "Software Development", deduplicate: bool = True, default_exam_year: Optional[int] = None, dry_run: bool = False):
        """Parse `filename` contents and create `Question` and `Answer` rows.

        Returns a dictionary with the number of created questions and any
        validation `errors` encountered per item. When `deduplicate` is True,
        questions with identical subject/text are skipped. When both exam_year
        and question_number are provided, dedupe also checks that pair.
        """
        # parse into a list of question dicts
        parsed = parse_file_to_questions(file_bytes, filename)
        created = []
        errors = []
        warnings = []
        skipped = 0
        for idx, p in enumerate(parsed):
            try:
                self._validate_parsed_question(p)
            except ValueError as e:
                errors.append({'index': idx, 'error': str(e), 'item': p})
                continue
            question_text = p.get('question_text')
            # Prevent dupes by subject/text when requested.
            if deduplicate and self.q_repo.exists_by_subject_and_text(subject, question_text):
                skipped += 1
                continue
            if deduplicate and p.get('exam_year') and p.get('question_number'):
                if self.q_repo.exists_by_year_and_number(p.get('exam_year'), p.get('question_number')):
                    skipped += 1
                    continue
            q = models.Question(
                subject=subject,
                question_text=p.get('question_text'),
                explanation=p.get('explanation'),
                exam_year=p.get('exam_year') or default_exam_year,
                question_number=p.get('question_number'),
                difficulty=p.get('difficulty'),
                question_type=p.get('question_type')
            )
            answers = []
            possible_answers = p.get('possible_answers', [])
            # If no answers are explicitly marked correct, mark the first.
            has_marked_correct = any(bool(a.get('is_correct')) for a in possible_answers)
            for i, a in enumerate(possible_answers):
                is_correct = bool(a.get('is_correct')) or (not has_marked_correct and i == 0)
                answers.append(models.Answer(answer_text=a.get('answer_text'), is_correct=is_correct))
            if not dry_run:
                self.q_repo.create(q, answers)
                created.append(q)
        return {'created': len(created), 'skipped': skipped, 'errors': errors, 'warnings': warnings}

    def _validate_parsed_question(self, p: dict):
        """Validate a parsed question dictionary and raise ValueError on error."""
        if not isinstance(p, dict):
            raise ValueError('question item must be an object')
        qt = p.get('question_text') or p.get('question')
        if not qt or not isinstance(qt, str) or not qt.strip():
            raise ValueError('missing or empty question_text')
        pas = p.get('possible_answers')
        if not pas or not isinstance(pas, list) or len(pas) == 0:
            raise ValueError('possible_answers missing or empty')
        # Ensure each answer has text
        for a in pas:
            if not isinstance(a, dict):
                raise ValueError('each possible_answer must be an object')
            if not a.get('answer_text'):
                raise ValueError('answer missing answer_text')


class GradingService:
    """Grade submitted quizzes and persist results."""
    def __init__(self, session: Session):
        self.session = session
        self.q_repo = repositories.QuestionRepository(session)
        self.a_repo = repositories.AnswerRepository(session)
        self.quiz_repo = repositories.QuizRepository(session)

    def grade(self, user_id: int, answers: List[dict]):
        """Grade a list of `{question_id, given_answer}` dicts.

        The method determines correctness by comparing the submitted
        answer text to stored correct answers. If a question has no
        explicit correct answer stored, the first stored answer is
        treated as correct (a pragmatic fallback for unstructured imports).
        The quiz result and detailed items are persisted and a summary
        payload is returned.
        """
        total = len(answers)
        correct = 0
        items = []
        payload_items = []
        # Locate Exams root to optionally extract report explanations per question.
        exams_root = Path(__file__).resolve().parents[2] / 'Exams'
        for a in answers:
            q = self.q_repo.get(a['question_id'])
            if not q:
                raise ValueError(f"question not found: {a['question_id']}")
            # fetch correct from answers table
            db_answers = self.a_repo.list_for_question(q.id)
            correct_texts = [an.answer_text for an in db_answers if an.is_correct]
            # fallback: if no answer marked correct, assume first stored answer is correct
            if not correct_texts and db_answers:
                correct_texts = [db_answers[0].answer_text]
            # fallback to answer key mapping if still empty
            if not correct_texts:
                key_answers = get_correct_answers_from_key(exams_root, q.exam_year, q.question_number)
                if key_answers:
                    correct_texts = key_answers
            # Determine submitted answer text: prefer answer_id if provided/valid, otherwise given_answer text.
            submitted_text: Optional[str] = None
            answer_id = a.get('answer_id')
            if answer_id is not None:
                db_answer = self.a_repo.get(answer_id)
                if not db_answer or db_answer.question_id != q.id:
                    raise ValueError(f"answer not found for question: {answer_id}")
                submitted_text = db_answer.answer_text
            else:
                submitted_text = a.get('given_answer')
            if submitted_text is None:
                raise ValueError("given_answer or answer_id required")
            norm = lambda s: s.strip().lower() if isinstance(s, str) else s
            is_correct = norm(submitted_text) in {norm(c) for c in correct_texts}
            if is_correct:
                correct += 1
            item = models.QuizResultItem(question_id=q.id, given_answer=submitted_text, correct=is_correct)
            items.append(item)
            # Prefer explicit explanation on the question; otherwise try to pull from examiner report by year/qnum.
            explanation = q.explanation
            if not explanation and q.exam_year and q.question_number and exams_root.exists():
                explanation = find_report_explanation(exams_root, q.exam_year, q.question_number)
            payload_items.append({
                'question_id': q.id,
                'given': submitted_text,
                'correct': bool(is_correct),
                'correct_answers': correct_texts,
                'explanation': explanation
            })
        score = correct
        percentage = (correct / total) * 100 if total > 0 else 0.0
        result = models.QuizResult(user_id=user_id, score=score, percentage=percentage, total_questions=total)
        created = self.quiz_repo.create_result(result, items)
        # build return payload
        return {
            'result_id': created.id,
            'score': created.score,
            'percentage': created.percentage,
            'total': created.total_questions,
            'items': payload_items
        }


class GoalService:
    """Manage weekly goals and compute progress summaries."""
    def __init__(self, session: Session):
        self.session = session
        self.goal_repo = repositories.GoalRepository(session)
        self.quiz_repo = repositories.QuizRepository(session)

    def set_weekly_goal(self, user_id: int, week_start: date, goal_type: str, goal_value: int):
        """Create or update a weekly goal record."""
        if goal_value < 0:
            raise ValueError("goal_value must be >= 0")
        g = models.WeeklyGoal(user_id=user_id, week_start=week_start, goal_type=goal_type, goal_value=goal_value)
        return self.goal_repo.set_goal(g)

    def get_progress(self, user_id: int, week_start: date):
        """Return a small progress summary for the given week.

        The current implementation counts quizzes and questions completed
        during the week and returns any configured goals.
        """
        # naive progress calculation: count quizzes in quiz_results within week_start..week_start+6
        import datetime
        start = datetime.datetime.combine(week_start, datetime.time())
        end = start + datetime.timedelta(days=7)
        from sqlmodel import select
        stmt = select(models.QuizResult).where(models.QuizResult.user_id == user_id, models.QuizResult.timestamp >= start, models.QuizResult.timestamp < end)
        results = self.session.exec(stmt).all()
        quizzes_completed = len(results)
        # count total questions completed in those quizzes
        total_questions = sum(r.total_questions or 0 for r in results)
        goals = self.goal_repo.get_goals_for_user_week(user_id, week_start)
        goal_map = {g.goal_type: g.goal_value for g in goals}
        return {
            'quizzes_completed': quizzes_completed,
            'questions_completed': total_questions,
            'goals': goal_map
        }
