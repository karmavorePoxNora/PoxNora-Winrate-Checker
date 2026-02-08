from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any


@dataclass(frozen=True)
class WindowStats:
    n: int
    wins: int
    losses: int
    winrate: float  # 0..1


def compute_window_stats(results: Iterable[Any], n: int) -> WindowStats:
    """
    results: iterable of match results ordered newest->oldest (or oldest->newest but be consistent).
    Each item can be bool (True=win), or "W"/"L", or "Win"/"Loss".
    Unknowns are ignored safely.
    """
    window = list(results)[: int(n)]
    wins = 0
    losses = 0
    for r in window:
        if r is True or r == "W" or r == "Win":
            wins += 1
        elif r is False or r == "L" or r == "Loss":
            losses += 1

    total = wins + losses
    winrate = (wins / total) if total else 0.0
    return WindowStats(n=int(n), wins=wins, losses=losses, winrate=winrate)
