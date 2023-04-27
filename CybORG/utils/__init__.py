from pathlib import Path
import sys
import logging
from logging import handlers


def prepare_log(base_dir, filename="info"):
    fmt = '%(asctime)s [%(levelname)s: %(filename)s %(lineno)d] %(message)s'
    log_dir = Path(base_dir) / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{filename}.log"
    return fmt, log_path


def set_log(level=logging.INFO, filename="info", base_dir=r".."):
    fmt, log_path = prepare_log(base_dir, filename)
    logging.basicConfig(level=level, format=fmt,
                        handlers=[handlers.RotatingFileHandler(log_path, maxBytes=100000000, backupCount=5),
                                  logging.StreamHandler(stream=sys.stdout)]
                        )
    logging.info("path: {}".format(log_path))
