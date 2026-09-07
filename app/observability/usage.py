import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class UsageLogger:
    def __init__(self, path: Path):
        self._path = path

    def log(self, completion: Any, kind: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        u = completion.usage
        extra = getattr(u, "model_extra", None) or {}
        usd = getattr(u, "cost", None)
        if usd is None:
            usd = extra.get("cost")
        line = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "model": completion.model,
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
            "usd": usd,
        }
        with self._path.open("a") as f:
            f.write(json.dumps(line) + "\n")
