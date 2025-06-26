import logging
from pathlib import Path

logs_dir = Path(__file__).parent.parent / "logging_info" / "LinReg"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

for level in levels:
    handler = logging.FileHandler(f"{logs_dir}/{level}.log")
    handler.setLevel(getattr(logging, level))
    logger.addHandler(handler)

def add_module_handler(logger, level = logging.DEBUG):
    handler = logging.FileHandler(f"{logs_dir}/{logger.name.replace('.', '-')}.log")
    handler.setLevel(level)
    logger.addHandler(handler)