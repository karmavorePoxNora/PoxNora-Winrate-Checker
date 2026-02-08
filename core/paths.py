# core/paths.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

# Project root = folder containing this file's parent ("core")
ROOT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default "canonical" outputs (UI may also create per-player datasets)
OUT_CSV = OUTPUT_DIR / "poxnora_matches.csv"
OUT_DASHBOARD_PNG = OUTPUT_DIR / "dashboard.png"
OUT_REPORT_HTML = OUTPUT_DIR / "report.html"

# Optional: keep datasets together
DATASETS_DIR = OUTPUT_DIR  # keep in output for now (backwards compatible)


def slug_player(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "player"


def make_dataset_filename(
    player_name: str,
    scrape_type: str,          # "all" or "recent"
    target_new: int | None,    # e.g. 50/100/200 for recent, None for all
    ranked_only: bool,
    *,
    ext: str = ".csv",
) -> str:
    """Human-readable, sortable dataset filename."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = slug_player(player_name)
    rank_tag = "ranked" if ranked_only else "all"
    if scrape_type == "recent":
        n = int(target_new or 0)
        return f"poxnora_{p}_{rank_tag}_recent_{n}_{ts}{ext}"
    return f"poxnora_{p}_{rank_tag}_full_{ts}{ext}"
