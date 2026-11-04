"""ECPA output receipts are opt-in, lazy, and emitted only once after success."""

import builtins
import os
import sys
from types import ModuleType

import pytest

from vllm_muxpd import evidence


class Owner:
    pass


@pytest.fixture(autouse=True)
def clear_state(monkeypatch):
    for key in evidence._BINDING_ENV:
        monkeypatch.delenv(key, raising=False)
    evidence._seen.clear()
    evidence._owner_pid = os.getpid()
    yield
    evidence._seen.clear()


def managed(monkeypatch, callback):
    for key in evidence._BINDING_ENV:
        monkeypatch.setenv(key, "test-binding")
    parent = ModuleType("vllm_hust_ext")
    child = ModuleType("vllm_hust_ext.runtime_evidence")
    child.emit_module_effective = callback
    monkeypatch.setitem(sys.modules, "vllm_hust_ext", parent)
    monkeypatch.setitem(sys.modules, "vllm_hust_ext.runtime_evidence", child)


def emit(owner=None, **kwargs):
    return evidence.emit_runtime_effective(
        owner or Owner(),
        role="worker",
        request_count=1,
        output_token_count=1,
        **kwargs,
    )


def test_ordinary_legacy_call_does_not_import_manager(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith("vllm_hust_ext")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert (
        evidence.emit_runtime_effective(
            Owner(), role="worker", request_count=0, output_token_count=0
        )
        is False
    )
    assert not evidence._seen


def test_success_deduplicates_but_new_launch_or_fork_reemits(monkeypatch):
    calls = []
    managed(
        monkeypatch,
        lambda implementation, **kwargs: calls.append((implementation, kwargs)) or True,
    )
    assert emit() is True
    assert emit() is False
    assert len(calls) == 1 and calls[0][0] is Owner
    assert calls[0][1] == {
        "role": "worker",
        "request_count": 1,
        "output_token_count": 1,
    }
    monkeypatch.setenv("VLLM_ECPA_LAUNCH_ID", "next-launch")
    assert emit() is True
    evidence._owner_pid = -1  # Inherited state belongs to another process.
    assert emit() is True
    assert len(calls) == 3


def test_incomplete_binding_never_silently_uses_legacy(monkeypatch):
    monkeypatch.setenv("VLLM_ECPA_PLAN_ID", "partial")
    with pytest.raises(ValueError, match="incomplete"):
        emit()
    assert not evidence._seen


@pytest.mark.parametrize("failure", [False, ValueError("rejected symbol")])
def test_failed_receipt_is_not_deduplicated(monkeypatch, failure):
    calls = []

    def callback(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            if isinstance(failure, Exception):
                raise failure
            return failure
        return True

    managed(monkeypatch, callback)
    with pytest.raises((RuntimeError, ValueError)):
        emit()
    assert not evidence._seen
    assert emit() is True and len(calls) == 2


def test_empty_or_invalid_output_counts_cannot_become_effective(monkeypatch):
    managed(monkeypatch, lambda *args, **kwargs: pytest.fail("must not emit"))
    for count in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive"):
            evidence.emit_runtime_effective(
                Owner(), role="scheduler", request_count=1, output_token_count=count
            )
    with pytest.raises(ValueError, match="unknown"):
        evidence.emit_runtime_effective(
            Owner(), role="startup", request_count=1, output_token_count=1
        )


def test_changed_environment_is_not_hidden_by_prior_deduplication(monkeypatch):
    calls = []

    def callback(*args, **kwargs):
        calls.append(kwargs)
        if os.environ["VLLM_ECPA_PLAN_ID"] == "forged":
            raise ValueError("wrong binding")
        return True

    managed(monkeypatch, callback)
    assert emit() is True
    monkeypatch.setenv("VLLM_ECPA_PLAN_ID", "forged")
    with pytest.raises(ValueError, match="wrong binding"):
        emit()
    assert len(calls) == 2
