"""Pydantic request/response schemas used by the API.

Schemas keep API input/output shapes stable and provide validation for
controller handlers and tests.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class RegisterIn(BaseModel):
    """Payload for user registration/login endpoints."""
    username: str
    password: str


class TokenOut(BaseModel):
    """Authentication response containing an access token."""
    access_token: str


class AnswerIn(BaseModel):
    """Representation of a possible answer in requests."""
    answer_text: str
    is_correct: bool = False


class QuestionIn(BaseModel):
    """Request format for importing a single question."""
    question_text: str
    possible_answers: List[AnswerIn]
    correct_answer: Optional[str]
    difficulty: Optional[str]
    question_type: Optional[str]


class QuizSubmissionItem(BaseModel):
    """Single submitted answer item used when grading a quiz."""
    question_id: int
    given_answer: Optional[str] = None
    answer_id: Optional[int] = None


class QuizSubmission(BaseModel):
    """Request model for grading containing a list of answers."""
    answers: List[QuizSubmissionItem]
