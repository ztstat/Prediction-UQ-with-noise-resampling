# src/utils.py

"""
Lightweight utilities for experiment logging and file management.
"""

from dataclasses import dataclass
from typing import TextIO
import os


@dataclass
class Logger:
    file: TextIO

    def log(self, msg: str) -> None:
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def open_logger(results_dir: str, filename: str = "run.log") -> Logger:
    ensure_dir(results_dir)
    f = open(os.path.join(results_dir, filename), "w", encoding="utf-8")
    return Logger(file=f)