"""
Singleton task manager: queue, single execution, pause/resume/cancel, progress tracking.
"""
import asyncio
import time
import uuid
import traceback
from typing import Any, Optional
from collections import deque


class TaskProgress:
    """Mutable progress state shared between the runner and the task coroutine."""
    __slots__ = (
        "task_id", "task_type", "label", "status",
        "current", "total", "phase",
        "started_at", "elapsed_sec", "eta_sec",
        "extra", "result",
        "_pause_event", "_cancel_flag",
    )

    def __init__(self, task_id: str, task_type: str, label: str, total: int = 0):
        self.task_id = task_id
        self.task_type = task_type          # "backtest" | "bruteforce" | "compare"
        self.label = label
        self.status = "queued"              # queued | running | paused | done | cancelled | error
        self.current = 0
        self.total = total
        self.phase = ""                     # e.g. "Loading data", "Horizon 1: 1200/5000"
        self.started_at: float = 0
        self.elapsed_sec: float = 0
        self.eta_sec: float = 0
        self.extra: dict = {}               # any extra info (best_accuracy, etc.)
        self.result: Any = None

        self._pause_event = asyncio.Event()
        self._pause_event.set()             # not paused initially
        self._cancel_flag = False

    def to_dict(self) -> dict:
        now = time.time()
        elapsed = (now - self.started_at) if self.started_at else 0
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "label": self.label,
            "status": self.status,
            "current": self.current,
            "total": self.total,
            "phase": self.phase,
            "started_at": self.started_at,
            "elapsed_sec": round(elapsed, 1),
            "eta_sec": round(self.eta_sec, 1),
            "extra": self.extra,
        }

    def update(self, current: int, total: int = 0, phase: str = ""):
        self.current = current
        if total > 0:
            self.total = total
        if phase:
            self.phase = phase
        now = time.time()
        self.elapsed_sec = now - self.started_at if self.started_at else 0
        if self.current > 0 and self.total > 0:
            rate = self.elapsed_sec / self.current
            remaining = self.total - self.current
            self.eta_sec = rate * remaining
        else:
            self.eta_sec = 0

    async def check_pause_cancel(self):
        """Await this in the inner loop. Blocks while paused, raises on cancel."""
        if self._cancel_flag:
            raise CancelledError(self.task_id)
        await self._pause_event.wait()
        if self._cancel_flag:
            raise CancelledError(self.task_id)

    def pause(self):
        self._pause_event.clear()
        self.status = "paused"

    def resume(self):
        self._pause_event.set()
        self.status = "running"

    def cancel(self):
        self._cancel_flag = True
        self._pause_event.set()  # unblock if paused so it can exit


class CancelledError(Exception):
    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"Task {task_id} cancelled")


class TaskManager:
    """Singleton: one running task at a time, queue for the rest."""

    _instance: Optional["TaskManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._queue: deque[tuple[str, Any]] = deque()   # (task_id, coro_factory)
        self._progress: dict[str, TaskProgress] = {}    # task_id -> progress
        self._history: list[dict] = []                   # finished tasks (last 100)
        self._current_task_id: Optional[str] = None
        self._runner_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    # ---------- public API ----------

    def enqueue(self, task_type: str, label: str, coro_factory, total: int = 0) -> str:
        """
        Add a task to the queue. coro_factory is an async callable that takes
        (TaskProgress) and returns the result dict.
        Returns task_id.
        """
        task_id = uuid.uuid4().hex[:12]
        progress = TaskProgress(task_id, task_type, label, total)
        self._progress[task_id] = progress
        self._queue.append((task_id, coro_factory))
        self._ensure_runner()
        return task_id

    def get_progress(self, task_id: str) -> Optional[dict]:
        p = self._progress.get(task_id)
        return p.to_dict() if p else None

    def get_status(self) -> dict:
        """Return full status: current task, queue, recent history."""
        current = None
        if self._current_task_id and self._current_task_id in self._progress:
            current = self._progress[self._current_task_id].to_dict()

        queue_items = []
        for tid, _ in self._queue:
            p = self._progress.get(tid)
            if p:
                queue_items.append(p.to_dict())

        return {
            "current": current,
            "queue": queue_items,
            "queue_length": len(self._queue),
            "history": self._history[-20:],
        }

    def pause(self, task_id: str) -> bool:
        p = self._progress.get(task_id)
        if p and p.status == "running":
            p.pause()
            return True
        return False

    def resume(self, task_id: str) -> bool:
        p = self._progress.get(task_id)
        if p and p.status == "paused":
            p.resume()
            return True
        return False

    def cancel(self, task_id: str) -> bool:
        p = self._progress.get(task_id)
        if not p:
            return False
        if p.status in ("queued",):
            # Remove from queue
            self._queue = deque(
                (tid, cf) for tid, cf in self._queue if tid != task_id
            )
            p.status = "cancelled"
            self._add_history(p)
            return True
        if p.status in ("running", "paused"):
            p.cancel()
            return True
        return False

    def remove_from_queue(self, task_id: str) -> bool:
        """Remove a queued (not running) task."""
        p = self._progress.get(task_id)
        if not p or p.status != "queued":
            return False
        self._queue = deque(
            (tid, cf) for tid, cf in self._queue if tid != task_id
        )
        p.status = "cancelled"
        self._add_history(p)
        return True

    def clear_queue(self) -> int:
        """Cancel all queued tasks (not the running one)."""
        count = 0
        new_q = deque()
        for tid, cf in self._queue:
            p = self._progress.get(tid)
            if p:
                p.status = "cancelled"
                self._add_history(p)
            count += 1
        self._queue = new_q
        return count

    def get_result(self, task_id: str) -> Optional[Any]:
        p = self._progress.get(task_id)
        if p:
            return p.result
        return None

    # ---------- internal ----------

    def _ensure_runner(self):
        if self._runner_task is None or self._runner_task.done():
            self._runner_task = asyncio.ensure_future(self._run_loop())

    async def _run_loop(self):
        while self._queue:
            task_id, coro_factory = self._queue.popleft()
            progress = self._progress.get(task_id)
            if not progress or progress.status == "cancelled":
                continue

            self._current_task_id = task_id
            progress.status = "running"
            progress.started_at = time.time()

            try:
                result = await coro_factory(progress)
                progress.elapsed_sec = time.time() - progress.started_at
                progress.status = "done"
                progress.result = result
            except CancelledError:
                progress.elapsed_sec = time.time() - progress.started_at
                progress.status = "cancelled"
            except Exception as e:
                traceback.print_exc()
                progress.elapsed_sec = time.time() - progress.started_at
                progress.status = "error"
                progress.extra["error"] = str(e)
            finally:
                self._add_history(progress)
                self._current_task_id = None

    def _add_history(self, p: TaskProgress):
        self._history.append(p.to_dict())
        if len(self._history) > 100:
            self._history = self._history[-100:]


# Global singleton
task_mgr = TaskManager()
