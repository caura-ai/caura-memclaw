"""Unit tests for AuthContext enforcement methods."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from core_api.auth import AuthContext


def test_enforce_read_only_allows_non_demo():
    ctx = AuthContext(tenant_id="t1", is_demo=False)
    ctx.enforce_read_only()  # no raise


def test_enforce_read_only_blocks_demo():
    ctx = AuthContext(tenant_id="t1", is_demo=True)
    with pytest.raises(HTTPException) as exc_info:
        ctx.enforce_read_only()
    assert exc_info.value.status_code == 403
    assert "demo" in exc_info.value.detail.lower()


def test_enforce_usage_limits_allows_normal_org():
    ctx = AuthContext(tenant_id="t1", is_read_only=False)
    ctx.enforce_usage_limits()  # no raise


def test_enforce_usage_limits_blocks_read_only_org():
    ctx = AuthContext(tenant_id="t1", is_read_only=True)
    with pytest.raises(HTTPException) as exc_info:
        ctx.enforce_usage_limits()
    assert exc_info.value.status_code == 403
    assert "read-only" in exc_info.value.detail.lower()
    assert "upgrade" in exc_info.value.detail.lower()


def test_read_only_is_independent_of_demo():
    """is_demo and is_read_only are separate flags enforced by separate methods."""
    # Demo but not read-only
    ctx = AuthContext(tenant_id="t1", is_demo=True, is_read_only=False)
    with pytest.raises(HTTPException):
        ctx.enforce_read_only()
    ctx.enforce_usage_limits()  # not read-only → no raise

    # Read-only but not demo (post-cancellation over limits)
    ctx = AuthContext(tenant_id="t1", is_demo=False, is_read_only=True)
    ctx.enforce_read_only()  # not demo → no raise
    with pytest.raises(HTTPException):
        ctx.enforce_usage_limits()
