from __future__ import annotations

import os
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class PerfMark:
    name: str
    elapsed_sec: float


class PerfTimer:
    def __init__(self, *, enabled: bool = False):
        self.enabled = bool(enabled)
        self._start = time.perf_counter()
        self._marks: list[PerfMark] = []

    def mark(self, name: str):
        if not self.enabled:
            return
        elapsed = float(time.perf_counter() - self._start)
        self._marks.append(PerfMark(name=str(name), elapsed_sec=elapsed))

    def rows(self) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        prev = 0.0
        for mark in self._marks:
            out.append(
                {
                    "step": mark.name,
                    "elapsed_sec": round(mark.elapsed_sec, 3),
                    "delta_sec": round(mark.elapsed_sec - prev, 3),
                }
            )
            prev = mark.elapsed_sec
        return out

    def summary_text(self, *, prefix: str = "perf") -> str:
        if not self.enabled:
            return ""
        rows = self.rows()
        if not rows:
            return f"{prefix}: no marks"
        chunks = [f"{item['step']}={item['delta_sec']:.3f}s" for item in rows]
        total = rows[-1]["elapsed_sec"]
        return f"{prefix}: " + " | ".join(chunks) + f" | total={total:.3f}s"


def perf_debug_enabled() -> bool:
    token = str(os.getenv("REALTIME0052_PERF_DEBUG", "")).strip().lower()
    return token in {"1", "true", "yes", "on"}


__all__ = ["PerfMark", "PerfTimer", "perf_debug_enabled"]
