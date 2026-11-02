"""Isolate native registry state across optional platform imports in tests."""

import sys

import pytest


@pytest.fixture(autouse=True)
def isolate_kv_spec_registry(monkeypatch):
    # Importing the native Ascend worker replaces the Mamba manager class.
    # A prior scheduler fixture may have registered the unpatched class.
    # Each test owns its registry rather than depending on collection order.
    registry = sys.modules.get("vllm.v1.kv_cache_spec_registry")
    if registry is not None:
        monkeypatch.setattr(registry, "_REGISTRY_KVCACHESPEC_LIST", {})
