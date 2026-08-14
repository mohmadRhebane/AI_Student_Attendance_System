
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONNECTION: str = os.getenv("DB_CONNECTION")
DB_HOST: str = os.getenv("DB_HOST")
DB_PORT: int = int(os.getenv("DB_PORT"))

DB_DATABASE: str = os.getenv("DB_DATABASE")
DB_USERNAME: str = os.getenv("DB_USERNAME")
DB_PASSWORD: str = os.getenv("DB_PASSWORD")
DATABASE_URL = ""
if DB_CONNECTION == "mysql":
    DATABASE_URL = f"mysql+aiomysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"

if DB_CONNECTION == "postgresql":
    DATABASE_URL = f"postgresql+asyncpg://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"

SECRET_KEY: str = os.getenv("SECRET_KEY")
ALGORITHM: str = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES:int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))


AI_PROJECT_ROOT: str = os.getenv("AI_PROJECT_ROOT")


def _env_bool(
    name: str,
    default: str = "false",
) -> bool:
    return (
        os.getenv(name, default)
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
    )


SCHEDULER_ENABLED: bool = _env_bool(
    "SCHEDULER_ENABLED",
    "false",
)

SCAN_PLANNER_INTERVAL_MINUTES: int = int(
    os.getenv(
        "SCAN_PLANNER_INTERVAL_MINUTES",
        "5",
    )
)

SCAN_WORKER_POLL_SECONDS: float = float(
    os.getenv(
        "SCAN_WORKER_POLL_SECONDS",
        "2",
    )
)

SCAN_JOB_MAX_ATTEMPTS: int = int(
    os.getenv(
        "SCAN_JOB_MAX_ATTEMPTS",
        "3",
    )
)