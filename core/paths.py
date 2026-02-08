# core/paths.py
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


# Project root = EXE folder when frozen, project root when running from source
ROOT_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent.parent
)

OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTPUT_DIR / "poxnora_matches.csv"
OUT_DASHBOARD_PNG = OUTPUT_DIR / "dashboard.png"
OUT_REPORT_HTML = OUTPUT_DIR / "report.html"

DATASETS_DIR = OUTPUT_DIR


def slug_player(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "player"


def make_dataset_filename(
    player_name: str,
    scrape_type: str,
    target_new: int | None,
    ranked_only: bool,
    *,
    ext: str = ".csv",
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = slug_player(player_name)
    rank_tag = "ranked" if ranked_only else "all"
    if scrape_type == "recent":
        n = int(target_new or 0)
        return f"poxnora_{p}_{rank_tag}_recent_{n}_{ts}{ext}"
    return f"poxnora_{p}_{rank_tag}_full_{ts}{ext}"
