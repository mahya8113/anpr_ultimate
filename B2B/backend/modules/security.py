"""
security.py - امنیت و احراز هویت (JWT، رمزنگاری، blacklist)
"""

import jwt
import hashlib
import secrets
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict
import os
import logging

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("JWT_SECRET", "super-secret-key-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))


class SecurityManager:
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        return pwd_context.verify(plain, hashed)

    @staticmethod
    def create_tokens(user_id: int, role: str) -> Dict[str, str]:
        now = datetime.utcnow()
        access_payload = {
            "sub": str(user_id), "role": role, "type": "access",
            "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        }
        refresh_payload = {
            "sub": str(user_id), "type": "refresh",
            "exp": now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        }
        access = jwt.encode(access_payload, SECRET_KEY, algorithm=ALGORITHM)
        refresh = jwt.encode(refresh_payload, SECRET_KEY, algorithm=ALGORITHM)
        return {"access_token": access, "refresh_token": refresh}

    @staticmethod
    def decode_token(token: str, expected_type: str = "access") -> Optional[Dict]:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("type") != expected_type:
                return None
            return payload
        except jwt.PyJWTError:
            return None

    @staticmethod
    def blacklist_token(token: str, ttl: int):
        from core.redis_client import redis_client
        import asyncio
        asyncio.create_task(redis_client.setex(f"bl_{token}", ttl, "1"))

    @staticmethod
    def generate_api_key() -> str:
        return secrets.token_urlsafe(32)