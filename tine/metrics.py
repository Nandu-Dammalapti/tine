"""In-process counters for the proxy, written in the Prometheus text format.

No dependency and no background thread: counters live in one process and start at
zero when it starts, as Prometheus counters do. Labels hold tenant names, providers,
routes and status classes, never request text, keys or tokens. ``GET /metrics`` serves
them only to a caller holding ``TINE_METRICS_TOKEN`` (see ``docs/PROXY.md``).
"""

from __future__ import annotations

import math
import threading

# Upper bounds of the request latency histogram, in seconds.
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(labels: tuple[tuple[str, str], ...]) -> str:
    return "{" + ",".join(f'{name}="{_escape(value)}"' for name, value in labels) + "}" if labels else ""


class Metrics:
    """Counters, gauges and one histogram, keyed by sorted label pairs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[tuple, float]] = {}
        self._gauges: dict[str, dict[tuple, float]] = {}
        self._histograms: dict[str, dict[tuple, list]] = {}
        self._help: dict[str, tuple[str, str]] = {}

    def describe(self, name: str, kind: str, text: str) -> None:
        self._help[name] = (kind, text)

    def inc(self, name: str, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            series = self._counters.setdefault(name, {})
            series[key] = series.get(key, 0.0) + amount

    def set(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._gauges.setdefault(name, {})[tuple(sorted(labels.items()))] = value

    def observe(self, name: str, value: float, **labels: str) -> None:
        if not math.isfinite(value):
            return
        key = tuple(sorted(labels.items()))
        with self._lock:
            state = self._histograms.setdefault(name, {}).setdefault(key, [[0] * len(LATENCY_BUCKETS), 0, 0.0])
            for index, bound in enumerate(LATENCY_BUCKETS):
                if value <= bound:
                    state[0][index] += 1
            state[1] += 1
            state[2] += value

    def value(self, name: str, **labels: str) -> float:
        """A counter's or gauge's current value, 0 when it has none: for tests."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            return self._counters.get(name, {}).get(key, self._gauges.get(name, {}).get(key, 0.0))

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for kind, store in (("counter", self._counters), ("gauge", self._gauges)):
                for name in sorted(store):
                    _, text = self._help.get(name, (kind, name))
                    lines += [f"# HELP {name} {text}", f"# TYPE {name} {kind}"]
                    lines += [f"{name}{_labels(key)} {value:g}" for key, value in sorted(store[name].items())]
            for name in sorted(self._histograms):
                _, text = self._help.get(name, ("histogram", name))
                lines += [f"# HELP {name} {text}", f"# TYPE {name} histogram"]
                for key, (buckets, count, total) in sorted(self._histograms[name].items()):
                    for bound, seen in zip(LATENCY_BUCKETS, buckets):
                        lines.append(f"{name}_bucket{_labels(key + (('le', f'{bound:g}'),))} {seen}")
                    lines.append(f"{name}_bucket{_labels(key + (('le', '+Inf'),))} {count}")
                    lines.append(f"{name}_sum{_labels(key)} {total:g}")
                    lines.append(f"{name}_count{_labels(key)} {count}")
        return "\n".join(lines) + "\n"
