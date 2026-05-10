"""Cascade pipeline — the actual classify_task business logic.

The package's ``__init__`` re-exports from here. Putting the cascade in its
own module keeps ``classifier/__init__.py`` to ~100 lines of re-exports and
makes the data flow easier to navigate:

    classifier/pipeline/
    └── classify_task.py    classify_task() + _classify_inner + tier adjustments
"""

from .classify_task import (
    MAX_TASK_CHARS,
    classify_task,
    reset_last_decision,
)

__all__ = [
    "MAX_TASK_CHARS",
    "classify_task",
    "reset_last_decision",
]
