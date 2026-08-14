
import logging
import os

os.makedirs("logs", exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/fastapi.log", encoding="utf-8"),

    ]
)

logger = logging.getLogger("SchoolAPI")