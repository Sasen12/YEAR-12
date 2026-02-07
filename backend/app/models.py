"""SQLModel data models.

This module defines the application's database tables using SQLModel.
Each class maps to a table and uses relationships where appropriate.
"""

from typing import Optional
from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime, date, timezone
from typing import List


class User(SQLModel, table=True):
    """A registered user.

    Fields:
    - `username`: unique login name
    - `password_hash`: hashed password string (never store plaintext)
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, nullable=False, unique=True)
    password_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Question(SQLModel, table=True):
    """A multiple-choice question belonging to a subject."""
    id: Optional[int] = Field(default=None, primary_key=True)
    subject: str = Field(index=True)
    question_text: str
    explanation: Optional[str] = None
    exam_year: Optional[int] = Field(default=None, index=True)
    question_number: Optional[int] = Field(default=None, index=True)
    difficulty: Optional[str] = None
    question_type: Optional[str] = None
    answers: List['Answer'] = Relationship(back_populates='question')


class Answer(SQLModel, table=True):
    """Possible answer for a `Question`.

    `is_correct` marks whether this answer is considered correct.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    question_id: int = Field(foreign_key='question.id')
    answer_text: str
    is_correct: bool = False
    question: Optional[Question] = Relationship(back_populates='answers')


class QuizResult(SQLModel, table=True):
    """A stored quiz result for a user with aggregated score."""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key='user.id')
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    score: Optional[int] = None
    percentage: Optional[float] = None
    total_questions: Optional[int] = None
    items: List['QuizResultItem'] = Relationship(back_populates='quiz_result')


class QuizResultItem(SQLModel, table=True):
    """A single question outcome inside a `QuizResult`."""
    id: Optional[int] = Field(default=None, primary_key=True)
    quiz_result_id: int = Field(foreign_key='quizresult.id')
    question_id: int = Field(foreign_key='question.id')
    given_answer: Optional[str]
    correct: bool = False
    quiz_result: Optional[QuizResult] = Relationship(back_populates='items')


class WeeklyGoal(SQLModel, table=True):
    """A simple per-user weekly goal record.

    `week_start` should be a date representing the first day of the
    tracking week.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key='user.id')
    week_start: date
    goal_type: str
    goal_value: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
