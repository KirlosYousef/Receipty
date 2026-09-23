from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from evals.agent_scoring import CASES_PATH, evaluate_cases, load_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Score scripted agent tool calls")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    cases = load_cases(args.cases)
    with tempfile.TemporaryDirectory() as tmp:
        rows = evaluate_cases(cases, db_path=Path(tmp) / "receipts.db")
    passed = sum(1 for row in rows if row["ok"])
    summary = {"cases": len(rows), "passed": passed, "rows": rows}
    text = json.dumps(summary, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    print(text)
    if passed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
