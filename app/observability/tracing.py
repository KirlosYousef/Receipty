from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def current_request_id() -> str | None:
    return _request_id.get()


@contextmanager
def bind_request_id(request_id: str | None) -> Iterator[None]:
    if request_id is None:
        yield
        return
    token = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(token)


class SpanRecorder:
    """Append one JSON object per finished step. A missing path records nothing."""

    def __init__(self, path: Path | None):
        self._path = path

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any]) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        error_type: str | None = None
        try:
            yield attributes
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            self._write(name, attributes, started, error_type)

    def _write(
        self,
        name: str,
        attributes: dict[str, Any],
        started: float,
        error_type: str | None,
    ) -> None:
        if self._path is None:
            return
        fields = dict(attributes)
        record: dict[str, Any] = {
            "name": name,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "request_id": fields.pop("request_id", current_request_id()),
        }
        record.update(
            {key: value for key, value in fields.items() if value is not None}
        )
        if error_type is not None:
            record["error.type"] = error_type
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
