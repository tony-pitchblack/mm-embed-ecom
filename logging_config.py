import logging
import os
import sys


def configure_logging(level: int | None = None) -> None:
    if level is None:
        name = os.environ.get("LOGLEVEL", "INFO").upper()
        try:
            level = getattr(logging, name)
        except AttributeError:
            level = logging.INFO
    kwargs: dict = {
        "level": level,
        "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "datefmt": "%H:%M:%S",
    }
    if sys.version_info >= (3, 8):
        kwargs["force"] = True
    logging.basicConfig(**kwargs)
    for name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "urllib3",
    ):
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
