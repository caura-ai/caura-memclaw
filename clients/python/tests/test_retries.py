"""Retry public reads through a real client with an offline HTTP transport."""

import time

import httpx
import pytest

from caura_client import Caura, CauraAPIError


@pytest.mark.parametrize("operation", ["search", "recall", "health", "get_document"])
@pytest.mark.parametrize("failure", [429, 502, 503, 504, "transport"])
def test_read_retries(operation, failure, monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    def handler(request):
        calls.append((request.method, str(request.url), request.content))
        if len(calls) < 3:
            if failure == "transport":
                raise httpx.ReadTimeout("transient", request=request)
            return httpx.Response(failure)
        return httpx.Response(200, json={"items": [], "summary": "ok"})

    with Caura("key", tenant_id="tenant", retries=2, transport=httpx.MockTransport(handler)) as client:
        if operation == "get_document":
            client.get_document("a/b", collection="docs")
        elif operation == "health":
            client.health()
        else:
            getattr(client, operation)("query")
    assert calls[0] == calls[1] == calls[2]
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize("operation", ["health", "write", "submit_interview"])
def test_default_and_writes_do_not_retry(operation, monkeypatch):
    calls = []
    monkeypatch.setattr(time, "sleep", lambda _: pytest.fail("unexpected retry"))

    def handler(request):
        calls.append(request)
        return httpx.Response(503)

    kwargs = {} if operation == "health" else {"retries": 3}
    with Caura("key", tenant_id="tenant", transport=httpx.MockTransport(handler), **kwargs) as client:
        with pytest.raises(CauraAPIError):
            if operation == "health":
                client.health()
            elif operation == "write":
                client.write("memory")
            else:
                client.submit_interview(node_id="n", agent_id="a", cursor_from=0, cursor_to=1, events=[])
    assert len(calls) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422, 500])
def test_permanent_errors_are_not_retried(status, monkeypatch):
    calls = []
    monkeypatch.setattr(time, "sleep", lambda _: pytest.fail("unexpected retry"))

    def handler(request):
        calls.append(request)
        return httpx.Response(status)

    with Caura("key", tenant_id="tenant", retries=3, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(CauraAPIError):
            client.health()
    assert len(calls) == 1


@pytest.mark.parametrize(
    "header,expected",
    [
        ("3", 3.0),
        ("bad", 0.5),
        ("-1", 0.5),
        ("inf", 0.5),
        ("nan", 0.5),
        ("Thu, 01 Jan 1970 00:00:03 GMT", 3.0),
    ],
)
def test_retry_after_and_exhaustion(header, expected, monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(time, "time", lambda: 0)

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": header})

    with Caura("key", tenant_id="tenant", retries=1, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(CauraAPIError) as exc:
            client.health()
    assert exc.value.status_code == 429
    assert len(calls) == 2
    assert sleeps == [expected]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"retries": -1},
        {"retries": 1.5},
        {"retry_backoff": -1},
        {"retry_backoff": float("nan")},
        {"retry_backoff": float("inf")},
    ],
)
def test_invalid_retry_configuration(kwargs):
    with pytest.raises(ValueError):
        Caura("key", tenant_id="tenant", **kwargs)


def test_transport_exhaustion_preserves_exception(monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("offline", request=request)

    with Caura("key", tenant_id="tenant", retries=1, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.ConnectError, match="offline"):
            client.health()
    assert len(calls) == 2
    assert sleeps == [0.5]
