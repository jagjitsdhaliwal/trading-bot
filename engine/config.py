"""App-wide configuration. Broker credentials load from .env (never commit
that file -- .gitignore already excludes it). Risk limits are NOT here --
see engine/risk/engine.py's RiskLimits, which is intentionally kept separate
and code-level so it can't be casually reconfigured at runtime."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tradovate_username: str = ""
    tradovate_password: str = ""
    tradovate_app_id: str = "trading-bot"
    tradovate_app_version: str = "1.0"
    tradovate_cid: str = ""
    tradovate_sec: str = ""
    tradovate_demo: bool = True

    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 1

    db_path: str = "data/trading.db"


settings = Settings()
