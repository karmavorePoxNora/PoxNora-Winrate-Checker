# core/dataset.py
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd

# =====================================================
# Column normalization / validation
# =====================================================

_CANONICAL_ALIASES = {
    "Win": {"win", "winner", "player_win"},
    "Loss": {"loss", "loser", "player_loss"},
    "Map": {"map", "mapname", "arena"},
    "Type": {"type", "mode"},
    "Ranked": {"ranked", "is_ranked", "rank"},
    "Date": {"date", "played", "timestamp"},
    "Duration": {"duration", "time"},
    "Rating": {"rating", "mmr"},
}

_REQUIRED_FOR_OPP_MAP = ("Win", "Loss", "Map")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    lower_to_actual = {str(c).strip().lower(): c for c in df.columns}
    rename: dict[str, str] = {}

    for canonical, aliases in _CANONICAL_ALIASES.items():
        if canonical in df.columns:
            continue
        for a in aliases:
            if a in lower_to_actual:
                rename[lower_to_actual[a]] = canonical
                break

    if rename:
        df = df.rename(columns=rename)

    # normalize common string columns (keeps downstream logic stable)
    for c in ["Win", "Loss", "Map", "Type", "Ranked", "Date", "Duration", "Rating"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Maps should never be empty/NaN for aggregation
    if "Map" in df.columns:
        df["Map"] = df["Map"].replace({"": "Unknown", "nan": "Unknown"}).fillna("Unknown")

    return df


def validate_for_opp_map(df: pd.DataFrame) -> Tuple[bool, str]:
    if df is None or len(df) == 0:
        return False, "Dataset has 0 rows."
    missing = [c for c in _REQUIRED_FOR_OPP_MAP if c not in df.columns]
    if missing:
        return False, "Missing columns for Opponents/Maps: " + ", ".join(missing)
    return True, "OK"


# =====================================================
# CSV loading
# =====================================================

def load_csv(path: Path) -> pd.DataFrame:
    """Load the tracker CSV safely and normalize columns.

    Returns empty DataFrame if missing or unreadable.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    return _normalize_columns(df)


# =====================================================
# Import (copy) CSV into output folder
# =====================================================

def import_csv_copy(src_path: Path, dst_path: Path) -> Path:
    """Copies a CSV into the tool's output location."""
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return dst_path
