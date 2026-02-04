import logging
from pathlib import Path

LOGGER = logging.getLogger("cali_logger")
LOGGER.setLevel(logging.DEBUG)
LOGGER.propagate = False  # Prevent duplicate logs from propagating to root logger

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# File handler
log_file = Path(__file__).parent / "cali_logger.log"
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Default: show INFO and above in console
console_handler.setFormatter(formatter)
# Set encoding to UTF-8 to support emoji characters on Windows
if hasattr(console_handler.stream, "reconfigure"):
    console_handler.stream.reconfigure(encoding="utf-8")
LOGGER.addHandler(console_handler)


def set_console_level(level: int | str) -> None:
    """Set the console handler level dynamically.

    This allows the console output level to be changed at runtime,
    e.g., via CLI arguments.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO, or "DEBUG", "INFO")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    console_handler.setLevel(level)
