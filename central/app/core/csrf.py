"""CSRF token utilities"""

import secrets
from fastapi import Request

CSRF_COOKIE = "csrf_token"

def get_csrf_token(request: Request) -> str | None:
    """Get the CSRF token from the cookie"""
    return request.cookies.get(CSRF_COOKIE)

def gen_csrf_token() -> str:
    """Generate a new CSRF token"""
    return secrets.token_urlsafe(32)
