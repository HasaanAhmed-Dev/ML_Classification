"""Small helper functions shared by all modules."""
import re
from pathlib import Path

import numpy as np


class ProjectError(Exception):
    """Raised for problems the user can fix (missing CSV, bad target name, ...)."""


class DecisionLog:
    """Collects every automatic decision so it can be printed and saved."""

    def __init__(self, echo=True):
        self.entries = []
        self.echo = echo

    def add(self, stage, message):
        line = f"[{stage}] {message}"
        self.entries.append(line)
        if self.echo:
            print("  " + line)

    def to_text(self):
        return "\n".join(self.entries)


def ensure_directories(*paths):
    """Create folders if they do not exist yet."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_text(path, text):
    """Write text to a UTF-8 file."""
    Path(path).write_text(text, encoding="utf-8")


def banner(title, char="=", width=60):
    """Return a simple banner string."""
    return f"{char * width}\n{title}\n{char * width}"


def split_name_tokens(name):
    """Split a column name into lowercase words (handles snake_case and camelCase)."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def shorten(label, max_len=22):
    """Shorten long labels so plots stay readable."""
    label = str(label)
    return label if len(label) <= max_len else label[: max_len - 3] + "..."


def fmt(value, digits=4):
    """Format a number for text output ('n/a' for missing values)."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        return f"{value:.{digits}f}"
    except (TypeError, ValueError):
        return str(value)
