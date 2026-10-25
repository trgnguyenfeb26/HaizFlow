"""Minimal Zernio REST client (https://zernio.com/agent-quickstart.md).

The API key comes from the OS credential store, falling back to the
ZERNIO_API_KEY environment variable. It is never written to project files.
"""

from __future__ import annotations

import json
import os
from urllib import error, request

from haizflow.services import secure_credentials

CREDENTIAL_TARGET = "HaizFlow/Zernio/APIKey"
API_BASE = "https://zernio.com/api/v1"
TIMEOUT_SECONDS = 30
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class ZernioError(RuntimeError):
    pass


def api_key() -> str:
    try:
        stored = secure_credentials.read_secret(CREDENTIAL_TARGET)
    except Exception:
        stored = ""
    return (stored or os.environ.get("ZERNIO_API_KEY", "")).strip()


def _call(method: str, path: str, payload: dict | None = None) -> dict:
    key = api_key()
    if not key:
        raise ZernioError("Zernio API key is not configured.")
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(API_BASE + path, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except error.HTTPError as exc:
        raise ZernioError(f"Zernio request rejected (HTTP {exc.code}).") from exc
    except (error.URLError, TimeoutError) as exc:
        raise ZernioError("Zernio connection failed; check the network.") from exc
    if len(body) > MAX_RESPONSE_BYTES:
        raise ZernioError("Zernio response too large.")
    try:
        result = json.loads(body.decode("utf-8"))
    except ValueError as exc:
        raise ZernioError("Zernio returned an invalid response.") from exc
    return result if isinstance(result, dict) else {"data": result}


def list_accounts() -> list[dict]:
    """Connected social accounts; each has an ``_id`` (accountId) and ``platform``."""
    result = _call("GET", "/accounts")
    accounts = result.get("accounts", result.get("data", []))
    return accounts if isinstance(accounts, list) else []


def create_post(content: str, targets: list[dict], *, scheduled_for: str | None = None,
                publish_now: bool = False) -> dict:
    """Create a post. With neither ``scheduled_for`` nor ``publish_now`` it is a draft.

    ``targets`` is a list of ``{"platform": ..., "accountId": ...}``.
    """
    payload: dict = {"content": content, "platforms": targets}
    if scheduled_for:
        payload["scheduledFor"] = scheduled_for
    elif publish_now:
        payload["publishNow"] = True
    result = _call("POST", "/posts", payload)
    post = result.get("post")
    return post if isinstance(post, dict) else result
