"""Reversible per-instance method replacement shared by the collective wrappers."""

from __future__ import annotations

from typing import Any

MISSING = object()

# (target, attribute, previous instance value or MISSING, installed wrapper)
InstalledPatch = tuple[Any, str, Any, Any]


def replace_method(
    installed: list[InstalledPatch], target: Any, name: str, wrapper: Any
) -> None:
    """Install ``wrapper`` on the instance and remember how to undo it."""
    previous = vars(target).get(name, MISSING)
    setattr(target, name, wrapper)
    installed.append((target, name, previous, wrapper))


def restore_methods(installed: list[InstalledPatch], kind: str) -> None:
    """Undo ``installed`` in reverse order, never overwriting a foreign value.

    Successfully restored entries are removed from ``installed``; entries that
    an external actor replaced stay recorded, and the first error is raised
    after every other entry has been attempted.
    """
    errors = []
    for index in reversed(range(len(installed))):
        target, name, previous, wrapper = installed[index]
        try:
            if getattr(target, name, None) is not wrapper:
                raise RuntimeError(f"MuxPD {kind} methods changed externally: {name}")
            if previous is MISSING:
                delattr(target, name)
            else:
                setattr(target, name, previous)
        except BaseException as error:
            errors.append(error)
        else:
            installed.pop(index)
    if errors:
        raise errors[0]


def note_rollback_failure(
    primary: BaseException, cleanup: BaseException, kind: str
) -> None:
    if hasattr(primary, "add_note"):
        primary.add_note(f"MuxPD {kind} rollback also failed: {cleanup!r}")
