"""Structured JSON logging: every question, tool call and latency is one JSON line."""

import json
import logging
import sys
from datetime import UTC, datetime

_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Format records as one JSON object per line; `extra=` fields become top-level keys."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the JSON line for one record."""
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload |= {k: v for k, v in vars(record).items() if k not in _RESERVED}
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Send JSON logs to stderr and quiet noisy libraries."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for noisy in (
        "httpx",
        "httpx2",
        "httpcore",
        "LiteLLM",
        "litellm",
        "urllib3",
        "sentence_transformers",
        "huggingface_hub",
        "transformers",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
