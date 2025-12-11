"""Application settings and validation."""

import os


class Settings:
    ENV: str
    JWT_SECRET: str
    MAX_UPLOAD_BYTES: int
    ALLOW_INSECURE_JWT: bool
    ALLOW_DEV_CORS: bool

    def __init__(self):
        self.ENV = os.getenv("ENV", "dev").lower()
        self.JWT_SECRET = os.getenv("JWT_SECRET", "change_me_for_prod")
        self.MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB default
        self.ALLOW_INSECURE_JWT = os.getenv("ALLOW_INSECURE_JWT", "false").lower() == "true"
        self.ALLOW_DEV_CORS = os.getenv("ALLOW_DEV_CORS", "true").lower() == "true"
        self._validate()

    def _validate(self):
        if self.ENV != "dev" and not self.ALLOW_INSECURE_JWT and self.JWT_SECRET == "change_me_for_prod":
            raise RuntimeError("JWT_SECRET must be set to a non-default value in non-dev environments")


settings = Settings()
