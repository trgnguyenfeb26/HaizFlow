import json

import pytest

from haizflow.services import zernio


class _Response:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, _size):
        return self._body


def _patch(monkeypatch, payload, seen):
    monkeypatch.setenv("ZERNIO_API_KEY", "test-key")
    monkeypatch.setattr(zernio.secure_credentials, "read_secret", lambda target: "")

    def fake(req, timeout):
        seen.append(req)
        return _Response(payload)

    monkeypatch.setattr(zernio.request, "urlopen", fake)


def test_create_post_defaults_to_draft(monkeypatch):
    seen = []
    _patch(monkeypatch, {"post": {"_id": "p1", "status": "draft"}}, seen)
    post = zernio.create_post("hi", [{"platform": "x", "accountId": "a1"}])
    body = json.loads(seen[0].data)
    assert post["_id"] == "p1"
    assert "publishNow" not in body and "scheduledFor" not in body
    assert seen[0].get_header("Authorization") == "Bearer test-key"


def test_list_accounts(monkeypatch):
    seen = []
    _patch(monkeypatch, {"accounts": [{"_id": "a1", "platform": "x"}]}, seen)
    assert zernio.list_accounts()[0]["platform"] == "x"


def test_missing_key(monkeypatch):
    monkeypatch.delenv("ZERNIO_API_KEY", raising=False)
    monkeypatch.setattr(zernio.secure_credentials, "read_secret", lambda target: "")
    with pytest.raises(zernio.ZernioError):
        zernio.list_accounts()
