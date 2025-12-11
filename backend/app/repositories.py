"""Repository classes encapsulating database operations.

Each repository is small and focused on a single aggregate (users,
questions, answers, quizzes, goals). Repositories return SQLModel
objects and perform commits/refreshes where appropriate.
"""

from typing import List, Optional
from sqlmodel import Session, select
from sqlalchemy import func
from . import models


class UserRepository:
    """CRUD operations for `User` objects."""
    def __init__(self, session: Session):
        self.session = session

    def create(self, user: models.User) -> models.User:
        """Persist a new user and return the managed instance."""
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    def get_by_username(self, username: str) -> Optional[models.User]:
        """Return a `User` by username or `None` if not found."""
        stmt = select(models.User).where(models.User.username == username)
        return self.session.exec(stmt).first()

    def get(self, user_id: int) -> Optional[models.User]:
        """Get a `User` by primary key."""
        return self.session.get(models.User, user_id)


class QuestionRepository:
    """CRUD operations for `Question` and related `Answer` records."""
    def __init__(self, session: Session):
        self.session = session

    def create(self, question: models.Question, answers: List[models.Answer]) -> models.Question:
        """Create a question and attach provided answers.

        The function commits the question first to obtain an id, then
        assigns that id to answers before committing them.
        """
        self.session.add(question)
        self.session.commit()
        self.session.refresh(question)
        for a in answers:
            a.question_id = question.id
            self.session.add(a)
        self.session.commit()
        return question

    def list_by_subject(self, subject: str) -> List[models.Question]:
        """Return all questions for a given subject."""
        stmt = select(models.Question).where(models.Question.subject == subject)
        return self.session.exec(stmt).all()

    def exists_by_subject_and_text(self, subject: str, question_text: str) -> bool:
        """Return True if a question with the same subject/text already exists."""
        stmt = select(models.Question.id).where(
            models.Question.subject == subject,
            models.Question.question_text == question_text
        )
        return self.session.exec(stmt).first() is not None

    def list_by_year_and_number(self, year: int, qnum: int) -> List[models.Question]:
        """Return questions for a given exam year and question number."""
        stmt = select(models.Question).where(
            models.Question.exam_year == year,
            models.Question.question_number == qnum
        )
        return self.session.exec(stmt).all()

    def exists_by_year_and_number(self, year: int, qnum: int) -> bool:
        """Return True if a question exists for the given exam year/number."""
        stmt = select(models.Question.id).where(
            models.Question.exam_year == year,
            models.Question.question_number == qnum
        )
        return self.session.exec(stmt).first() is not None

    def get_random(self, subject: str, limit: int = 10) -> List[models.Question]:
        """Return up to `limit` random questions for `subject`.

        The implementation first attempts to use the database function
        `random()` (SQLite). If that fails for any reason, it falls back
        to loading all questions into memory and shuffling them.
        """
        stmt = select(models.Question).where(models.Question.subject == subject).order_by(func.random()).limit(limit)
        # fallback: simple select then shuffle if random not available
        try:
            return self.session.exec(stmt).all()
        except Exception:
            q = self.list_by_subject(subject)
            import random
            random.shuffle(q)
            return q[:limit]

    def get(self, question_id: int) -> Optional[models.Question]:
        """Fetch a question by id."""
        return self.session.get(models.Question, question_id)


class AnswerRepository:
    """Query helpers for `Answer` records."""
    def __init__(self, session: Session):
        self.session = session

    def get(self, answer_id: int) -> Optional[models.Answer]:
        """Fetch a single answer by id."""
        return self.session.get(models.Answer, answer_id)

    def list_for_question(self, question_id: int) -> List[models.Answer]:
        """List all answer rows for the provided `question_id`."""
        stmt = select(models.Answer).where(models.Answer.question_id == question_id)
        return self.session.exec(stmt).all()


class QuizRepository:
    """Persist quiz result aggregates and their items."""
    def __init__(self, session: Session):
        self.session = session

    def create_result(self, result: models.QuizResult, items: List[models.QuizResultItem]) -> models.QuizResult:
        """Store a `QuizResult` and attach its `QuizResultItem`s."""
        self.session.add(result)
        self.session.commit()
        self.session.refresh(result)
        for it in items:
            it.quiz_result_id = result.id
            self.session.add(it)
        self.session.commit()
        return result


class GoalRepository:
    """Repository for weekly goal upserts and queries."""
    def __init__(self, session: Session):
        self.session = session

    def set_goal(self, goal: models.WeeklyGoal) -> models.WeeklyGoal:
        """Upsert a weekly goal for a user/week/type combination."""
        # upsert by user_id/week_start/goal_type
        existing = self.session.exec(
            select(models.WeeklyGoal).where(
                models.WeeklyGoal.user_id == goal.user_id,
                models.WeeklyGoal.week_start == goal.week_start,
                models.WeeklyGoal.goal_type == goal.goal_type
            )
        ).first()
        if existing:
            existing.goal_value = goal.goal_value
            self.session.add(existing)
            self.session.commit()
            return existing
        self.session.add(goal)
        self.session.commit()
        self.session.refresh(goal)
        return goal

    def get_goals_for_user_week(self, user_id: int, week_start) -> List[models.WeeklyGoal]:
        """Return all goals for `user_id` in the specified `week_start`."""
        stmt = select(models.WeeklyGoal).where(models.WeeklyGoal.user_id == user_id, models.WeeklyGoal.week_start == week_start)
        return self.session.exec(stmt).all()
