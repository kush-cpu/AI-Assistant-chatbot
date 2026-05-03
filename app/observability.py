import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "data" / "logs"
OBSERVABILITY_LOG_PATH = LOG_DIR / "observability.jsonl"
EVALUATION_LOG_PATH = LOG_DIR / "evaluations.jsonl"
MAX_RETURNED_EVENTS = 200

_log_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_trace_id() -> str:
    return str(uuid4())


def _json_default(value):
    return str(value)


def append_jsonl(path: Path, event: dict) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "id": event.get("id") or new_trace_id(),
        "timestamp": event.get("timestamp") or utc_now(),
        **event,
    }

    with _log_lock:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=_json_default))
            file.write("\n")

    return record


def log_observability(event_type: str, payload: dict) -> dict:
    return append_jsonl(
        OBSERVABILITY_LOG_PATH,
        {
            "event_type": event_type,
            **payload,
        },
    )


def log_evaluation(payload: dict) -> dict:
    return append_jsonl(EVALUATION_LOG_PATH, payload)


def read_jsonl(path: Path, limit: int = 50) -> list[dict]:
    if not path.exists():
        return []

    limit = max(1, min(limit, MAX_RETURNED_EVENTS))

    with _log_lock:
        lines = path.read_text(encoding="utf-8").splitlines()

    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return records


def get_observability_events(limit: int = 50, event_type: str | None = None) -> list[dict]:
    events = read_jsonl(OBSERVABILITY_LOG_PATH, limit=MAX_RETURNED_EVENTS)
    if event_type:
        events = [event for event in events if event.get("event_type") == event_type]

    return events[-max(1, min(limit, MAX_RETURNED_EVENTS)):]


def get_evaluation_events(limit: int = 20) -> list[dict]:
    return read_jsonl(EVALUATION_LOG_PATH, limit=limit)


def get_metrics_summary(limit: int = 50) -> dict:
    eval_events = get_evaluation_events(limit=limit)
    observability_events = get_observability_events(limit=limit)

    completed_eval_runs = [
        event for event in eval_events
        if event.get("event_type") == "prompt_comparison"
    ]
    latency_values = [
        event.get("latency_ms", {}).get("total")
        for event in observability_events
        if isinstance(event.get("latency_ms"), dict)
        and isinstance(event.get("latency_ms", {}).get("total"), (int, float))
    ]

    avg_latency = None
    if latency_values:
        avg_latency = round(sum(latency_values) / len(latency_values), 2)

    latest_eval = completed_eval_runs[-1] if completed_eval_runs else None

    return {
        "evaluation_runs": len(completed_eval_runs),
        "observability_events": len(observability_events),
        "average_latency_ms": avg_latency,
        "latest_evaluation": latest_eval,
        "recent_evaluations": completed_eval_runs[-10:],
    }
