"""Unit tests for common.structlog_config — GCP-compatible structlog processors."""

from __future__ import annotations

from common.structlog_config import _map_to_gcp_severity, _rename_event_to_message


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
