"""Authentication helpers and FastAPI security dependency.

This module provides utilities to decode JWT tokens and a FastAPI
dependency `get_current_user` that validates the bearer token and
returns the corresponding `User` model instance from the database.

The implementation is intentionally small: token verification raises
HTTPExceptions on failure so it can be used directly inside route
dependencies.
"""

from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from .services import JWT_SECRET, JWT_ALGORITHM
from sqlmodel import Session
from .database import engine
from . import repositories

bearer_scheme = HTTPBearer()

def decode_token(token: str):
    """Decode and verify a JWT token.

    Returns the decoded payload on success or raises an HTTPException
    with status 401 on failure.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail='token expired')
    except Exception:
        raise HTTPException(status_code=401, detail='invalid token')


def get_current_user(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    """FastAPI dependency that returns the authenticated user.

    The function extracts the bearer token from the request, decodes it
    and performs a database lookup to return the `User` object. It raises
    an HTTPException(401) for any authentication issue.
    """
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('user_id')
    if not user_id:
        raise HTTPException(status_code=401, detail='invalid token payload')
    # simple DB lookup
    with Session(engine) as session:
        user = repositories.UserRepository(session).get(user_id)
        if not user:
            raise HTTPException(status_code=401, detail='user not found')
        return user
