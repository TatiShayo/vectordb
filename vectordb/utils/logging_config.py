"""
Structured JSON logging — G02.
Emit newline-delimited JSON for every log record.
Human-readable in dev (set VDB_LOG_FORMAT=text), JSON in prod.
"""
from __future__ import annotations
import json
import logging
import os
import sys
import time
from typing import Optional


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc = {
            "ts":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        # Extra fields set via logger.info("msg", extra={"trace_id": "..."})
        for key in ("trace_id", "collection", "op", "duration_ms"):
            if hasattr(record, key):
                doc[key] = getattr(record, key)
        return json.dumps(doc, ensure_ascii=False)


def setup(level: str = "INFO") -> None:
    fmt = os.environ.get("VDB_LOG_FORMAT", "json").lower()
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s  %(message)s")
        )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Silence noisy third-party loggers
    for name in ("uvicorn.access", "httpx", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)
