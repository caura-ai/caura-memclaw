"""Unit tests for common.structlog_config — GCP-compatible structlog processors."""

from __future__ import annotations

import logging

from common.structlog_config import (
    _THIRD_PARTY_LOGGERS_TO_REROUTE,
    _map_to_gcp_severity,
    _rename_event_to_message,
    _route_third_party_to_root,
    _third_party_logger_original_state,
)


def test_map_to_gcp_severity_adds_matching_label() -> None:
    for method, expected in [
        ("info", "INFO"),
        ("warning", "WARNING"),
        ("warn", "WARNING"),
        ("error", "ERROR"),
        ("critical", "CRITICAL"),
        ("debug", "DEBUG"),
    ]:
        event_dict = {"event": "hello"}
        result = _map_to_gcp_severity(None, method, event_dict)
        assert result["severity"] == expected


def test_map_to_gcp_severity_unknown_method_falls_back_to_default() -> None:
    result = _map_to_gcp_severity(None, "notice", {"event": "hi"})
    assert result["severity"] == "DEFAULT"


def test_map_to_gcp_severity_preserves_explicit_override() -> None:
    # Callers can pass severity= directly to log a higher level than the
    # method name implies (e.g. logger.info("x", severity="NOTICE")).
    result = _map_to_gcp_severity(None, "info", {"event": "x", "severity": "NOTICE"})
    assert result["severity"] == "NOTICE"


def test_map_to_gcp_severity_treats_none_override_as_absent() -> None:
    # `logger.info("x", severity=None)` must not leak "severity": null to
    # Cloud Logging (which treats it as DEFAULT).
    result = _map_to_gcp_severity(None, "info", {"event": "x", "severity": None})
    assert result["severity"] == "INFO"


def test_map_to_gcp_severity_treats_empty_string_as_absent() -> None:
    # GCP maps severity="" to DEFAULT too — replace with the method-derived
    # label the same way we do for None.
    result = _map_to_gcp_severity(None, "info", {"event": "x", "severity": ""})
    assert result["severity"] == "INFO"


def test_map_to_gcp_severity_preserves_falsy_non_none_non_empty() -> None:
    # Don't silently rewrite 0/False/[] — those are caller-bound values, not
    # an "absent" signal. Contract is: None and "" are absent; anything else
    # is intentional.
    values: list[object] = [0, False, []]
    for value in values:
        result = _map_to_gcp_severity(None, "info", {"event": "x", "severity": value})
        assert result["severity"] == value


def test_rename_event_to_message_moves_field() -> None:
    result = _rename_event_to_message(
        None, "info", {"event": "hello world", "extra": 1}
    )
    assert result == {"message": "hello world", "extra": 1}


def test_rename_event_to_message_preserves_existing_message() -> None:
    # If someone explicitly set `message`, don't overwrite it with `event`.
    # But still remove `event` so Cloud Logging JSON doesn't carry both.
    result = _rename_event_to_message(
        None, "info", {"event": "e", "message": "explicit"}
    )
    assert result == {"message": "explicit"}


def test_rename_event_to_message_noop_without_event() -> None:
    result = _rename_event_to_message(None, "info", {"other": "x"})
    assert result == {"other": "x"}


def test_rename_event_to_message_event_none_produces_empty_message() -> None:
    # logger.info(None) reaches here with `event` explicitly set to None.
    # Emit an empty string so the GCP entry still has a `message` summary.
    result = _rename_event_to_message(None, "info", {"event": None})
    assert result == {"message": ""}


# ─── _route_third_party_to_root ─────────────────────────────────────────


def test_route_third_party_to_root_clears_handlers_and_enables_propagation() -> None:
    """Each rerouted logger ends with handlers=[] and propagate=True so its
    lines flow through the root ProcessorFormatter."""
    # Pre-populate one of the listed loggers with a fake handler + propagate=False
    # to simulate uvicorn / fastmcp's shipped state.
    target = logging.getLogger(_THIRD_PARTY_LOGGERS_TO_REROUTE[0])
    fake_handler = logging.NullHandler()
    target.addHandler(fake_handler)
    target.propagate = False
    try:
        _route_third_party_to_root()
        assert fake_handler not in target.handlers
        assert target.propagate is True
    finally:
        # Restore for any subsequent test that touches this logger.
        target.propagate = True


def test_route_third_party_to_root_is_idempotent() -> None:
    """Calling twice doesn't change state on the second call (already-rerouted
    loggers stay handler-less, propagate stays True)."""
    _route_third_party_to_root()
    snapshot = {
        name: (
            list(logging.getLogger(name).handlers),
            logging.getLogger(name).propagate,
        )
        for name in _THIRD_PARTY_LOGGERS_TO_REROUTE
    }
    _route_third_party_to_root()
    for name in _THIRD_PARTY_LOGGERS_TO_REROUTE:
        lg = logging.getLogger(name)
        assert list(lg.handlers) == snapshot[name][0]
        assert lg.propagate == snapshot[name][1]


def test_route_third_party_to_root_preserves_operator_set_level_when_no_handlers() -> (
    None
):
    """If a logger has no own handlers, its level was set by the operator
    relative to root (e.g. ``uvicorn --log-level warning``). Don't silently
    reset to NOTSET — that would flood Cloud Logging with previously
    suppressed access logs."""
    target = logging.getLogger(_THIRD_PARTY_LOGGERS_TO_REROUTE[0])
    # Clear leaked snapshot state from earlier tests so this test sees a
    # genuinely-empty handler list at snapshot time. Without this, a prior
    # test's snapshot of ``[NullHandler]`` would make this run go through
    # the rerouting path that resets the level — masking the regression
    # this test is supposed to catch.
    _third_party_logger_original_state.clear()
    for h in list(target.handlers):
        target.removeHandler(h)
    target.propagate = True
    target.setLevel(logging.WARNING)
    try:
        _route_third_party_to_root()
        assert target.level == logging.WARNING, (
            "operator-set level should not be silently reset to NOTSET when "
            "the library hadn't installed its own handlers"
        )
    finally:
        target.setLevel(logging.NOTSET)
        _third_party_logger_original_state.clear()
