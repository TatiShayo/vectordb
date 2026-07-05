"""
Background task tracker — G09.
Lets long operations (rebuild, snapshot) run async and be polled.
"""
from __future__ import annotations
import threading, time, uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


class TaskStore:
    def __init__(self):
        self._tasks: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def submit(self, name: str, fn: Callable, *args, **kwargs) -> str:
        task_id = str(uuid.uuid4())[:8]
        task = {
            "task_id": task_id, "name": name, "status": "running",
            "message": "Started", "result": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
        }
        with self._lock:
            self._tasks[task_id] = task

        def _run():
            try:
                result = fn(*args, **kwargs)
                with self._lock:
                    self._tasks[task_id].update({
                        "status": "done", "message": "Completed",
                        "result": result,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    })
            except Exception as exc:
                with self._lock:
                    self._tasks[task_id].update({
                        "status": "error", "message": str(exc),
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    })

        t = threading.Thread(target=_run, daemon=True, name=f"task-{task_id}")
        t.start()
        return task_id

    def get(self, task_id: str) -> Optional[Dict]:
        with self._lock:
            return dict(self._tasks.get(task_id, {})) or None

    def list(self) -> list:
        with self._lock:
            return [dict(t) for t in self._tasks.values()]


tasks = TaskStore()
