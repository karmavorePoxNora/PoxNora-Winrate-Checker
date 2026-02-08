# ui/app_qt_v22.py
from __future__ import annotations

import os
import shutil
import sys
import traceback
import re
from dataclasses import dataclass
from core.paths import ROOT_DIR, OUTPUT_DIR, OUT_DASHBOARD_PNG, OUT_REPORT_HTML
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from ui.pg_dashboard_v34 import LiveDashboard  # type: ignore
except Exception:
    try:
        from ui.pg_dashboard_v34 import LiveDashboard  # type: ignore
    except Exception:
        try:
            from ui.pg_dashboard_v34 import LiveDashboard  # type: ignore
        except Exception:
            try:
                from ui.pg_dashboard_v34 import LiveDashboard  # type: ignore
            except Exception:
                try:
                    from ui.pg_dashboard_v34 import LiveDashboard  # type: ignore
                except Exception:
                    LiveDashboard = None  # type: ignore


def _style_table_widget(tbl: QtWidgets.QTableWidget) -> None:
    tbl.setAlternatingRowColors(True)
    tbl.setShowGrid(False)
    tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    tbl.verticalHeader().setVisible(False)
    hdr = tbl.horizontalHeader()
    hdr.setStretchLastSection(True)
    hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

    # subtle row separators + spacing (Opponents / Maps tables)
    tbl.setStyleSheet(
        "QTableWidget { background: transparent; }"
        "QHeaderView::section { padding: 10px 8px; }"
        "QTableWidget::item { padding: 10px 8px; border-bottom: 1px solid rgba(255,255,255,0.06); }"
        "QTableWidget::item:selected { background: rgba(120,170,255,0.30); }"
    )

from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import QMessageBox

APP_TITLE = "PoxNora Tracker v34"


# -----------------------------
# Scraper integration (poxnora.com via Playwright) - safe import
# -----------------------------
try:
    from core.poxnora_scraper import ScrapeConfig, run_scrape, ScrapeCancelled  # type: ignore

    _SCRAPER_AVAILABLE = True
except Exception:
    ScrapeConfig = None  # type: ignore
    run_scrape = None  # type: ignore
    ScrapeCancelled = Exception  # type: ignore
    _SCRAPER_AVAILABLE = False


# -----------------------------
# Paths / helpers
# -----------------------------
@dataclass(frozen=True)
class AppPaths:
    root_dir: Path
    output_dir: Path
    png_path: Path
    report_path: Path


def _project_root() -> Path:
    return ROOT_DIR


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "player"


def _resolve_paths() -> AppPaths:
    """Centralized paths. Uses core/paths.py as defaults."""
    root = ROOT_DIR

    # Optional override for portable installs
    env_raw = os.environ.get("POXNORA_TRACKER_OUTPUT_DIR", "").strip()
    output_dir = Path(env_raw).expanduser() if env_raw else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / OUT_DASHBOARD_PNG.name
    report_path = output_dir / OUT_REPORT_HTML.name

    return AppPaths(root_dir=root, output_dir=output_dir, png_path=png_path, report_path=report_path)


def _to_file_url(path: Path) -> QtCore.QUrl:
    return QtCore.QUrl.fromLocalFile(str(path.resolve()))


def _safe_open_path(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return QDesktopServices.openUrl(_to_file_url(path))
    except Exception:
        return False


def _try_import_callable(module_name: str, candidates: list[str]) -> Optional[Callable[..., Any]]:
    try:
        mod = __import__(module_name, fromlist=["*"])
    except Exception:
        return None
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None


def _auto_find_csv(root_dir: Path, output_dir: Path) -> Optional[Path]:
    def newest_csv_in(folder: Path) -> Optional[Path]:
        if not folder.exists() or not folder.is_dir():
            return None
        csvs = [p for p in folder.glob("*.csv") if p.is_file()]
        if not csvs:
            return None
        csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return csvs[0]

    preferred = [
        "poxnora_matches_clean.csv",
        "poxnora_matches.csv",
        "matches.csv",
        "games.csv",
    ]
    for name in preferred:
        p = output_dir / name
        if p.exists() and p.is_file():
            return p
    for name in preferred:
        p = root_dir / name
        if p.exists() and p.is_file():
            return p

    p = newest_csv_in(output_dir)
    if p:
        return p
    p = newest_csv_in(root_dir)
    if p:
        return p

    for sub in ["data", "datasets", "exports", "export", "csv", "input", "inputs"]:
        p = newest_csv_in(root_dir / sub)
        if p:
            return p

    return None


# -----------------------------
# KPI parsing (supports multiple CSV formats)
# -----------------------------
def _format_percent(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        return f"{x * 100:.1f}%"
    except Exception:
        return "—"


def _truthy(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        try:
            return bool(int(v))
        except Exception:
            return False
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "ranked", "on"}


def _effective_player_for_df(df: Any, typed_player: str) -> str:
    """Use typed player only if present in df; otherwise infer from Win/Loss."""
    try:
        if df is None:
            return (typed_player or '').strip()
        cols = {str(c).strip().lower(): c for c in getattr(df, 'columns', [])}
        c_win = cols.get('win')
        c_loss = cols.get('loss')
        if not (c_win and c_loss):
            return (typed_player or '').strip()
        tp = (typed_player or '').strip()
        if tp:
            p = tp.lower()
            w = df[c_win].astype(str).str.strip().str.lower()
            l = df[c_loss].astype(str).str.strip().str.lower()
            if ((w == p) | (l == p)).any():
                return tp
        inferred = _infer_player_from_winloss(df) or ''
        return inferred.strip() or tp
    except Exception:
        return (typed_player or '').strip()


def _infer_player_from_winloss(df: Any) -> Optional[str]:
    try:
        if "Win" not in df.columns or "Loss" not in df.columns:
            return None
        import pandas as pd  # type: ignore

        names = pd.concat([df["Win"], df["Loss"]], ignore_index=True)
        names = names.dropna().astype(str).str.strip()
        names = names[names != ""]
        if len(names) == 0:
            return None
        return names.value_counts().idxmax()
    except Exception:
        return None


def _sorted_matches(df: Any) -> Any:
    """Sort matches oldest->newest when Date exists, so "Last N" windows are truly MOST RECENT.

    View-layer only (does not alter scraping or on-disk CSV).
    """
    try:
        if df is None or len(df) == 0:
            return df
        cols = {str(c).strip().lower(): c for c in getattr(df, "columns", [])}
        c_date = cols.get("date")
        if not c_date:
            return df

        import pandas as pd  # type: ignore
        dt = pd.to_datetime(df[c_date], format="%b %d, %Y %I:%M:%S %p", errors="coerce")
        try:
            if getattr(dt, "notna", None) is not None and dt.notna().sum() < max(3, int(0.10 * len(dt))):
                dt = pd.to_datetime(df[c_date], errors="coerce")
        except Exception:
            pass

        try:
            if getattr(dt, "notna", None) is not None and dt.notna().sum() < max(3, int(0.10 * len(dt))):
                return df
        except Exception:
            pass

        work = df.copy()
        work["_dt_sort"] = dt
        work = work.sort_values("_dt_sort", kind="mergesort")  # stable
        work = work.drop(columns=["_dt_sort"], errors="ignore")
        return work
    except Exception:
        return df



def _compute_kpis(df: Any, player_name: str, ranked_only: bool) -> dict[str, str]:
    """
    Supports:
      A) Match-history: columns Win/Loss (names), optional Ranked/Type/Date
      B) Aggregated: columns wins/losses
    """
    if df is None:
        return {"Games": "—", "Winrate": "—", "Last 50": "—", "Last 100": "—", "Last 150": "—", "Last 200": "—"}

    try:
        cols_lower = {str(c).strip().lower() for c in df.columns}
    except Exception:
        return {"Games": "—", "Winrate": "—", "Last 50": "—", "Last 100": "—", "Last 150": "—", "Last 200": "—"}

    # Format B: aggregated
    if {"wins", "losses"}.issubset(cols_lower):
        try:
            wins_col = next(c for c in df.columns if str(c).strip().lower() == "wins")
            losses_col = next(c for c in df.columns if str(c).strip().lower() == "losses")

            wins = float(df[wins_col].fillna(0).sum())
            losses = float(df[losses_col].fillna(0).sum())
            games = int(wins + losses)
            wr = (wins / (wins + losses)) if (wins + losses) > 0 else None

            return {
                "Games": str(games),
                "Winrate": _format_percent(wr),
                "Last 50": "—",
                "Last 100": "—",
                "Last 150": "—",
                "Last 200": "—",
            }
        except Exception:
            return {"Games": "—", "Winrate": "—", "Last 50": "—", "Last 100": "—", "Last 150": "—", "Last 200": "—"}

    # Format A: match history
    if {"win", "loss"}.issubset(cols_lower):
        try:
            import pandas as pd  # type: ignore

            win_col = next(c for c in df.columns if str(c).strip().lower() == "win")
            loss_col = next(c for c in df.columns if str(c).strip().lower() == "loss")

            work = df.copy()

            if ranked_only and "ranked" in cols_lower:
                ranked_col = next(c for c in work.columns if str(c).strip().lower() == "ranked")
                work = work[work[ranked_col].apply(_truthy)]

            if "type" in cols_lower:
                type_col = next(c for c in work.columns if str(c).strip().lower() == "type")
                work = work[work[type_col].astype(str).str.strip().str.lower().isin({"1v1", "1 v 1"})]

            player = player_name.strip()
            if not player:
                player = _infer_player_from_winloss(work) or ""

            if not player:
                return {
                    "Games": str(len(work)),
                    "Winrate": "—",
                    "Last 50": "—",
                    "Last 200": "—",
                }

            w = work[win_col].astype(str).str.strip()
            l = work[loss_col].astype(str).str.strip()
            involved = (w == player) | (l == player)
            work = work[involved]

            if len(work) == 0:
                return {"Games": "0", "Winrate": "—", "Last 50": "—", "Last 200": "—"}

            work = _sorted_matches(work)
            is_win = (work[win_col].astype(str).str.strip() == player).astype(int)

            def wr_last(n: int) -> Optional[float]:
                if len(is_win) == 0:
                    return None
                tail = is_win.tail(n) if len(is_win) > n else is_win
                return float(tail.sum()) / float(len(tail)) if len(tail) else None

            return {
                "Games": str(int(len(is_win))),
                "Winrate": _format_percent(wr_last(len(is_win))),
                "Last 50": _format_percent(wr_last(50)),
                "Last 100": _format_percent(wr_last(100)),
                "Last 150": _format_percent(wr_last(150)),
                "Last 200": _format_percent(wr_last(200)),
            }
        except Exception:
            return {"Games": "—", "Winrate": "—", "Last 50": "—", "Last 100": "—", "Last 150": "—", "Last 200": "—"}

    try:
        return {"Games": str(len(df)), "Winrate": "—", "Last 50": "—", "Last 200": "—"}
    except Exception:
        return {"Games": "—", "Winrate": "—", "Last 50": "—", "Last 100": "—", "Last 150": "—", "Last 200": "—"}


def _compute_opponents_maps(df: Any, player_name: str, ranked_only: bool, top_n: int = 15) -> tuple[list[tuple], list[tuple]]:
    """Compute Opponents + Maps tables from the active dataset.

    Returns:
      opp_rows: (Opponent, Games, Wins, Losses, Winrate%)
      map_rows: (Map, Games, Wins, Losses, Winrate%)
    """
    if df is None:
        return [], []

    try:
        import pandas as pd  # type: ignore
    except Exception:
        return [], []

    d = df.copy()

    # normalize column access
    cols = {str(c).strip().lower(): c for c in getattr(d, "columns", [])}
    c_win = cols.get("win")
    c_loss = cols.get("loss")
    c_map = cols.get("map")
    c_ranked = cols.get("ranked")
    c_type = cols.get("type")

    if not (c_win and c_loss):
        return [], []

    # ranked-only filter (match dashboard style)
    if ranked_only and c_ranked:
        try:
            d = d[d[c_ranked].apply(_truthy)]
        except Exception:
            pass

    # 1v1 filter if Type exists
    if c_type:
        try:
            d = d[d[c_type].astype(str).str.strip().str.lower().isin(["1v1", "1 v 1"])]
        except Exception:
            pass

    # infer player if not provided (match KPI behavior)
    pname = (player_name or "").strip()
    if not pname:
        try:
            pname = _infer_player_from_winloss(d) or ""
        except Exception:
            pname = ""
    pname_norm = pname.strip().lower()

    if not pname_norm:
        return [], []

    # build row-aligned opponent + win flag
    opp_list: list[str] = []
    win_flag: list[int] = []
    map_list: list[str] = []

    for w, l, m in zip(
        d[c_win].astype(str).tolist(),
        d[c_loss].astype(str).tolist(),
        (d[c_map].astype(str).tolist() if c_map else ["Unknown"] * len(d)),
    ):
        w0 = str(w).strip().lower()
        l0 = str(l).strip().lower()
        if w0 == pname_norm:
            opp_list.append(str(l).strip() or "Unknown")
            win_flag.append(1)
            map_list.append(str(m).strip() or "Unknown")
        elif l0 == pname_norm:
            opp_list.append(str(w).strip() or "Unknown")
            win_flag.append(0)
            map_list.append(str(m).strip() or "Unknown")
        else:
            continue

    if not opp_list:
        return [], []

    # Opponents table
    opp_rows: list[tuple] = []
    try:
        g = (
            pd.DataFrame({"Opponent": opp_list, "Win": win_flag})
            .groupby("Opponent", dropna=False)["Win"]
            .agg(["count", "sum"])
            .rename(columns={"count": "Games", "sum": "Wins"})
        )
        g["Losses"] = g["Games"] - g["Wins"]
        g["Winrate"] = (g["Wins"] / g["Games"]) * 100.0
        g = g.sort_values(["Games", "Winrate"], ascending=[False, False]).head(int(top_n))
        for name, row in g.iterrows():
            opp_rows.append((str(name), int(row["Games"]), int(row["Wins"]), int(row["Losses"]), f'{float(row["Winrate"]):.1f}%'))
    except Exception:
        opp_rows = []

    # Maps table
    map_rows: list[tuple] = []
    try:
        gm = (
            pd.DataFrame({"Map": pd.Series(map_list).astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"}).fillna("Unknown"), "Win": win_flag})
            .groupby("Map")["Win"]
            .agg(["count", "sum"])
            .rename(columns={"count": "Games", "sum": "Wins"})
        )
        gm["Losses"] = gm["Games"] - gm["Wins"]
        gm["Winrate"] = (gm["Wins"] / gm["Games"]) * 100.0
        gm = gm.sort_values(["Games", "Winrate"], ascending=[False, False]).head(int(top_n))
        for name, row in gm.iterrows():
            map_rows.append((str(name), int(row["Games"]), int(row["Wins"]), int(row["Losses"]), f'{float(row["Winrate"]):.1f}%'))
    except Exception:
        map_rows = []

    return opp_rows, map_rows
# -----------------------------
# UI primitives
# -----------------------------
class Card(QtWidgets.QFrame):
    def __init__(self, title: str, subtitle: Optional[str] = None, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("CardTitle")
        layout.addWidget(self.title)

        self.subtitle = QtWidgets.QLabel(subtitle or "")
        self.subtitle.setObjectName("CardSub")
        self.subtitle.setVisible(bool(subtitle))
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.subtitle)

        self.body = QtWidgets.QVBoxLayout()
        self.body.setContentsMargins(0, 6, 0, 0)
        self.body.setSpacing(10)
        layout.addLayout(self.body, 1)


class KPIBadge(QtWidgets.QFrame):
    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("KPIBadge")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("KPIBadgeTitle")
        self.value_label = QtWidgets.QLabel("—")
        self.value_label.setObjectName("KPIBadgeValue")

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


# -----------------------------
# Pages
# -----------------------------
class HomePage(QtWidgets.QWidget):
    build_requested = QtCore.Signal()
    import_copy_requested = QtCore.Signal()
    open_output_requested = QtCore.Signal()
    open_png_requested = QtCore.Signal()
    open_report_requested = QtCore.Signal()
    player_name_changed = QtCore.Signal(str)
    ranked_only_changed = QtCore.Signal(bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        header = QtWidgets.QFrame()
        header.setObjectName("HeaderCard")
        hl = QtWidgets.QVBoxLayout(header)
        hl.setContentsMargins(14, 14, 14, 14)
        hl.setSpacing(10)
        outer.addWidget(header)

        top = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(APP_TITLE)
        title.setObjectName("HeaderTitle")
        top.addWidget(title)
        top.addStretch(1)
        hl.addLayout(top)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)

        self.player_edit = QtWidgets.QLineEdit()
        self.player_edit.setObjectName("PlayerEditWide")
        self.player_edit.setPlaceholderText("Player name (e.g. karmavore)")
        self.player_edit.setFixedHeight(38)
        self.player_edit.setReadOnly(False)
        self.player_edit.setEnabled(True)
        self.player_edit.textChanged.connect(self.player_name_changed.emit)
        row.addWidget(self.player_edit, 1)

        self.ranked_only = QtWidgets.QCheckBox("Ranked 1v1 only")
        self.ranked_only.setObjectName("RankedOnly")
        self.ranked_only.stateChanged.connect(lambda _: self.ranked_only_changed.emit(self.ranked_only.isChecked()))
        row.addWidget(self.ranked_only, 0, Qt.AlignRight)
        hl.addLayout(row)


        # Dataset picker (CSV in output folder)
        ds = QtWidgets.QHBoxLayout()
        ds.setSpacing(10)
        ds_lbl = QtWidgets.QLabel("Dataset:")
        ds_lbl.setStyleSheet("color:#9fb0c3;")
        self.combo_dataset = QtWidgets.QComboBox()
        self.combo_dataset.setMinimumWidth(520)
        self.btn_refresh_datasets = QtWidgets.QPushButton("Refresh list")
        self.btn_refresh_datasets.setObjectName("TertiaryButton")
        self.btn_refresh_datasets.setFixedHeight(30)
        self.btn_load_dataset = QtWidgets.QPushButton("Load selected")
        self.btn_load_dataset.setObjectName("SecondaryButton")
        self.btn_load_dataset.setFixedHeight(30)
        ds.addWidget(ds_lbl)
        ds.addWidget(self.combo_dataset, 1)
        ds.addWidget(self.btn_refresh_datasets)
        ds.addWidget(self.btn_load_dataset)
        hl.addLayout(ds)

        btns = QtWidgets.QHBoxLayout()
        btns.setSpacing(10)

        self.btn_build = QtWidgets.QPushButton("Build Only (from existing CSV)")
        self.btn_build.setObjectName("PrimaryButton")
        self.btn_build.setFixedHeight(38)
        self.btn_build.clicked.connect(self.build_requested.emit)

        self.btn_import_copy = QtWidgets.QPushButton("Import CSV (copy)")
        self.btn_import_copy.setObjectName("SecondaryButton")
        self.btn_import_copy.setFixedHeight(38)
        self.btn_import_copy.clicked.connect(self.import_copy_requested.emit)

        btns.addWidget(self.btn_build)
        btns.addWidget(self.btn_import_copy)
        btns.addStretch(1)

        self.btn_open_output = QtWidgets.QPushButton("Open Output Folder")
        self.btn_open_output.setObjectName("TertiaryButton")
        self.btn_open_output.setFixedHeight(38)
        self.btn_open_output.clicked.connect(self.open_output_requested.emit)

        self.btn_open_png = QtWidgets.QPushButton("Open Dashboard PNG")
        self.btn_open_png.setObjectName("TertiaryButton")
        self.btn_open_png.setFixedHeight(38)
        self.btn_open_png.clicked.connect(self.open_png_requested.emit)

        self.btn_open_report = QtWidgets.QPushButton("Open Report HTML")
        self.btn_open_report.setObjectName("TertiaryButton")
        self.btn_open_report.setFixedHeight(38)
        self.btn_open_report.clicked.connect(self.open_report_requested.emit)

        btns.addWidget(self.btn_open_output)
        btns.addWidget(self.btn_open_png)
        btns.addWidget(self.btn_open_report)
        hl.addLayout(btns)

        kpis = QtWidgets.QHBoxLayout()
        kpis.setSpacing(10)
        self.kpi_games = KPIBadge("Games")
        self.kpi_wr = KPIBadge("Winrate")
        self.kpi_50 = KPIBadge("Last 50")
        self.kpi_100 = KPIBadge("Last 100")
        self.kpi_150 = KPIBadge("Last 150")
        self.kpi_200 = KPIBadge("Last 200")
        kpis.addWidget(self.kpi_games)
        kpis.addWidget(self.kpi_wr)
        kpis.addWidget(self.kpi_50)
        kpis.addWidget(self.kpi_100)
        kpis.addWidget(self.kpi_150)
        kpis.addWidget(self.kpi_200)
        outer.addLayout(kpis)

        content = QtWidgets.QGridLayout()
        content.setHorizontalSpacing(12)
        content.setVerticalSpacing(12)
        outer.addLayout(content, 1)

        self.card_activity = Card("Activity", "Scrape/build messages live here.")
        self.activity_log = QtWidgets.QTextEdit()
        self.activity_log.setObjectName("LogBox")
        self.activity_log.setReadOnly(True)
        self.card_activity.body.addWidget(self.activity_log, 1)
        content.addWidget(self.card_activity, 0, 0, 2, 1)

        self.card_status = Card("Status", "Quick view of current data + environment.")
        self.status_text = QtWidgets.QLabel("Ready.")
        self.status_text.setObjectName("StatusText")
        self.status_text.setWordWrap(True)
        self.status_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.card_status.body.addWidget(self.status_text)
        content.addWidget(self.card_status, 0, 1, 1, 1)

        self.dashboard_wrap = QtWidgets.QWidget()
        self.dashboard_wrap.setObjectName("DashboardWrap")
        dash_lay = QtWidgets.QVBoxLayout(self.dashboard_wrap)
        dash_lay.setContentsMargins(0, 0, 0, 0)
        dash_lay.setSpacing(10)

        dash_title = QtWidgets.QLabel("Dashboard")
        dash_title.setObjectName("SectionTitle")
        dash_sub = QtWidgets.QLabel("Live analytics (pyqtgraph). PNG export still available.")
        dash_sub.setObjectName("SectionSub")
        dash_sub.setWordWrap(True)

        dash_lay.addWidget(dash_title)
        dash_lay.addWidget(dash_sub)

        self.preview_label = QtWidgets.QLabel("Live dashboard unavailable — install pyqtgraph + numpy to enable.")
        self.preview_label.setObjectName("PreviewImage")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(320)
        self.preview_label.setWordWrap(True)

        # Live pyqtgraph dashboard (preferred). If unavailable, we fall back to PNG preview label.
        self.live_dashboard = None
        try:
            if LiveDashboard is not None:
                self.live_dashboard = LiveDashboard()
        except Exception:
            self.live_dashboard = None

        if self.live_dashboard is not None:
            dash_lay.addWidget(self.live_dashboard, 1)
            self.preview_label.hide()
            dash_lay.addWidget(self.preview_label)
        else:
            dash_lay.addWidget(self.preview_label, 1)

        content.addWidget(self.dashboard_wrap, 1, 1, 1, 1)

        content.setColumnStretch(0, 3)
        content.setColumnStretch(1, 5)

    def append_log(self, text: str) -> None:
        self.activity_log.append(text)

    def set_kpis(self, k: dict[str, str]) -> None:
        self.kpi_games.set_value(k.get("Games", "—"))
        self.kpi_wr.set_value(k.get("Winrate", "—"))
        self.kpi_50.set_value(k.get("Last 50", "—"))
        self.kpi_100.set_value(k.get("Last 100", "—"))
        self.kpi_150.set_value(k.get("Last 150", "—"))
        self.kpi_200.set_value(k.get("Last 200", "—"))

    def set_status(self, text: str) -> None:
        self.status_text.setText(text)

    def set_preview_png(self, png_path: Path) -> None:
        if not png_path.exists():
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("dashboard.png not found yet")
            return

        try:
            self.preview_label.setPixmap(QPixmap())
            data = png_path.read_bytes()
            pix = QPixmap()
            ok = pix.loadFromData(data)
            if not ok or pix.isNull():
                raise RuntimeError("loadFromData failed")
        except Exception:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Failed to load dashboard.png")
            return

        target = self.preview_label.size() - QtCore.QSize(18, 18)
        if target.width() > 0 and target.height() > 0:
            pix = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.preview_label.setPixmap(pix)
        self.preview_label.setText("")


    def set_live_dashboard(self, df, player_name: str, ranked_only: bool) -> None:
        if getattr(self, "live_dashboard", None) is None:
            return
        try:
            self.live_dashboard.update_from_dataframe(df, player_name=player_name, ranked_only=ranked_only)
        except Exception:
            # Live preview is optional; never crash the app.
            return

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        pix = self.preview_label.pixmap()
        if pix is None or pix.isNull():
            return
        target = self.preview_label.size() - QtCore.QSize(18, 18)
        if target.width() <= 0 or target.height() <= 0:
            return
        self.preview_label.setPixmap(pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation))



class OpponentsPage(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        title = QtWidgets.QLabel("Opponents")
        title.setObjectName("PageTitle")
        outer.addWidget(title)

        self.card = Card("Opponent Summary", "Top opponents from the active CSV.")
        self.search = QtWidgets.QLineEdit()
        self.search.setObjectName("OpponentsPageSearch")
        self.search.setPlaceholderText("Search opponents…")
        self.search.setFixedHeight(34)
        self.search.setStyleSheet("padding: 6px 10px; border-radius: 16px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); color: rgba(236,242,255,0.95);")
        self.search.textChanged.connect(self._apply_filter)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setObjectName("Table")
        _style_table_widget(self.table)
        self.table.setHorizontalHeaderLabels(["Opponent", "Games", "Wins", "Losses", "Winrate"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.card.body.addWidget(self.search)
        self.card.body.addWidget(self.table, 1)
        outer.addWidget(self.card, 1)

    def set_rows(self, rows: list[tuple[str, int, int, int, str]]) -> None:
        self.table.setRowCount(0)
        for opp, games, wins, losses, wr in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            values = [opp, str(games), str(wins), str(losses), wr]
            for c, v in enumerate(values):
                it = QtWidgets.QTableWidgetItem(v)
                it.setFlags(it.flags() ^ Qt.ItemIsEditable)
                # alignment: name left, numbers centered
                if c == 0:
                    it.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                else:
                    it.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.table.setItem(r, c, it)




    def _apply_filter(self) -> None:
        q = (self.search.text() or "").strip().lower()
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            txt = (item.text() if item else "").strip().lower()
            self.table.setRowHidden(r, bool(q) and (q not in txt))

class MapsPage(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        title = QtWidgets.QLabel("Maps")
        title.setObjectName("PageTitle")
        outer.addWidget(title)

        self.card = Card("Map Summary", "Top maps from the active CSV.")
        self.search = QtWidgets.QLineEdit()
        self.search.setObjectName("MapsPageSearch")
        self.search.setPlaceholderText("Search maps…")
        self.search.setFixedHeight(34)
        self.search.setStyleSheet("padding: 6px 10px; border-radius: 16px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); color: rgba(236,242,255,0.95);")
        self.search.textChanged.connect(self._apply_filter)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setObjectName("Table")
        _style_table_widget(self.table)
        self.table.setHorizontalHeaderLabels(["Map", "Games", "Wins", "Losses", "Winrate"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.card.body.addWidget(self.search)
        self.card.body.addWidget(self.table, 1)
        outer.addWidget(self.card, 1)

    def set_rows(self, rows: list[tuple[str, int, int, int, str]]) -> None:
        self.table.setRowCount(0)
        for name, games, wins, losses, wr in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            values = [name, str(games), str(wins), str(losses), wr]
            for c, v in enumerate(values):
                it = QtWidgets.QTableWidgetItem(v)
                it.setFlags(it.flags() ^ Qt.ItemIsEditable)
                # alignment: name left, numbers centered
                if c == 0:
                    it.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                else:
                    it.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.table.setItem(r, c, it)



    def _apply_filter(self) -> None:
        q = (self.search.text() or "").strip().lower()
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            txt = (item.text() if item else "").strip().lower()
            self.table.setRowHidden(r, bool(q) and (q not in txt))

class ScrapeWorker(QtCore.QObject):
    sig_log = QtCore.Signal(str)
    sig_status = QtCore.Signal(str)
    sig_progress = QtCore.Signal(int, int)
    sig_need_continue = QtCore.Signal()
    sig_done = QtCore.Signal(str)  # csv path
    sig_error = QtCore.Signal(str)

    def __init__(self, cfg: Any, mode: str, target_new: int) -> None:
        super().__init__()
        self._cfg = cfg
        self._mode = mode
        self._target_new = target_new
        self._continue_event = QtCore.QWaitCondition()
        self._mutex = QtCore.QMutex()
        self._cancelled = False

    @QtCore.Slot()
    def cancel(self) -> None:
        self._mutex.lock()
        self._cancelled = True
        self._mutex.unlock()
        self._continue_event.wakeAll()

    def _is_cancelled(self) -> bool:
        self._mutex.lock()
        v = self._cancelled
        self._mutex.unlock()
        return v

    @QtCore.Slot()
    def continue_after_login(self) -> None:
        self._continue_event.wakeAll()

    def _wait_for_continue(self) -> None:
        self.sig_need_continue.emit()
        self._mutex.lock()
        try:
            self._continue_event.wait(self._mutex)
        finally:
            self._mutex.unlock()

    @QtCore.Slot()
    def run(self) -> None:
        if not _SCRAPER_AVAILABLE or run_scrape is None:
            self.sig_error.emit("Scraper module not available. Ensure core/poxnora_scraper.py exists.")
            return

        try:
            self.sig_status.emit("Opening browser for login...")
            csv_path, _df = run_scrape(
                cfg=self._cfg,
                mode=self._mode,
                target_new=self._target_new,
                log=lambda m: self.sig_log.emit(str(m)),
                progress_cb=lambda cur, total: self.sig_progress.emit(int(cur), int(total)),
                wait_for_continue=self._wait_for_continue,
                cancelled=self._is_cancelled,
            )
            self.sig_done.emit(str(csv_path))
            self.sig_status.emit("Ready.")
        except ScrapeCancelled:
            self.sig_status.emit("Cancelled.")
            self.sig_log.emit("Scrape cancelled.")
        except Exception as e:
            self.sig_status.emit("Error.")
            self.sig_error.emit(str(e))


class ScrapePage(QtWidgets.QWidget):
    sig_scrape_all = QtCore.Signal()
    sig_scrape_recent = QtCore.Signal(int)
    sig_continue = QtCore.Signal()
    sig_cancel = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        title = QtWidgets.QLabel("Scrape")
        title.setObjectName("PageTitle")
        outer.addWidget(title)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)

        self.btn_all = QtWidgets.QPushButton("Scrape All")
        self.btn_50 = QtWidgets.QPushButton("Scrape +50")
        self.btn_100 = QtWidgets.QPushButton("Scrape +100")
        self.btn_150 = QtWidgets.QPushButton("Scrape +150")
        self.btn_200 = QtWidgets.QPushButton("Scrape +200")

        self.btn_continue = QtWidgets.QPushButton("Continue after login")
        self.btn_continue.setEnabled(False)

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)

        row.addWidget(self.btn_all)
        row.addWidget(self.btn_50)
        row.addWidget(self.btn_100)
        row.addWidget(self.btn_150)
        row.addWidget(self.btn_200)
        row.addStretch(1)
        row.addWidget(self.btn_continue)
        row.addWidget(self.btn_cancel)

        outer.addLayout(row)

        hint = QtWidgets.QLabel(
            "Flow: click a Scrape button → browser opens → log in on poxnora.com → open Match History page → click Continue."
        )
        hint.setWordWrap(True)
        hint.setObjectName("HelperText")
        outer.addWidget(hint)

        info = QtWidgets.QLabel("Tip: Use +50/+100/+150/+200 to append without re-scraping everything.")
        info.setWordWrap(True)
        info.setObjectName("HelperText")
        outer.addWidget(info)

        outer.addStretch(1)

        self.btn_all.clicked.connect(self.sig_scrape_all.emit)
        self.btn_50.clicked.connect(lambda: self.sig_scrape_recent.emit(50))
        self.btn_100.clicked.connect(lambda: self.sig_scrape_recent.emit(100))
        self.btn_150.clicked.connect(lambda: self.sig_scrape_recent.emit(150))
        self.btn_200.clicked.connect(lambda: self.sig_scrape_recent.emit(200))
        self.btn_continue.clicked.connect(self.sig_continue.emit)
        self.btn_cancel.clicked.connect(self.sig_cancel.emit)

    def set_waiting_for_continue(self, waiting: bool) -> None:
        self.btn_continue.setEnabled(waiting)
        self.btn_cancel.setEnabled(True)

    def set_running(self, running: bool) -> None:
        for b in (self.btn_all, self.btn_50, self.btn_100, self.btn_150, self.btn_200):
            b.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        if not running:
            self.btn_continue.setEnabled(False)


class ExportsPage(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        title = QtWidgets.QLabel("Exports")
        title.setObjectName("PageTitle")
        outer.addWidget(title)

        card = Card("Export Tools", "Ready for wiring (PNG/HTML subsets, filtered exports, etc).")
        outer.addWidget(card)
        outer.addStretch(1)


# -----------------------------
# Main Window
# -----------------------------
class AppWindow(QtWidgets.QMainWindow):
    def _ensure_player_edit_editable(self) -> None:
        try:
            self.page_home.player_edit.setEnabled(True)
            self.page_home.player_edit.setReadOnly(False)
        except Exception:
            pass


    def __init__(self, start_page: int = 0) -> None:
        super().__init__()
        self.paths = _resolve_paths()
        self.current_csv_path: Optional[Path] = _auto_find_csv(self.paths.root_dir, self.paths.output_dir)
        self.active_df = None  # loaded DataFrame for current_csv_path
        self._active_csv_loaded: Optional[Path] = None
        self._dataset_version: int = 0

        self.player_name: str = ""
        self.ranked_only: bool = False

        self._scrape_thread: Optional[QtCore.QThread] = None
        self._scrape_worker: Optional[ScrapeWorker] = None

        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1200, 760)

        root = QtWidgets.QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)

        layout = QtWidgets.QHBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        layout.addWidget(self._build_sidebar())

        self.stack = QtWidgets.QStackedWidget()
        self.stack.setObjectName("Stack")

        self.page_home = HomePage()
        self.page_opponents = OpponentsPage()
        self.page_maps = MapsPage()
        self.page_scrape = ScrapePage()
        self.page_exports = ExportsPage()

        self.stack.addWidget(self.page_home)
        self.stack.addWidget(self.page_opponents)
        self.stack.addWidget(self.page_maps)
        self.stack.addWidget(self.page_scrape)
        self.stack.addWidget(self.page_exports)

        layout.addWidget(self.stack, 1)

        self._wire_signals()
        self._apply_dark_theme()

        self._set_page(start_page)
        self._log_startup()
        self._refresh_home()

    # ---------- Sidebar / nav ----------
    def _build_sidebar(self) -> QtWidgets.QWidget:
        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(250)

        sb = QtWidgets.QVBoxLayout(sidebar)
        sb.setContentsMargins(12, 12, 12, 12)
        sb.setSpacing(10)

        brand = QtWidgets.QLabel("PoxNora\nTracker")
        brand.setObjectName("Brand")
        sb.addWidget(brand)

        sub = QtWidgets.QLabel("Modern dashboard build")
        sub.setObjectName("BrandSub")
        sb.addWidget(sub)

        sb.addSpacing(8)

        self.nav_home = self._nav_button("🏠  Home")
        self.nav_opponents = self._nav_button("⚔️  Opponents")
        self.nav_maps = self._nav_button("🗺️  Maps")
        self.nav_scrape = self._nav_button("🧲  Scrape")
        self.nav_exports = self._nav_button("📦  Exports")

        sb.addWidget(self.nav_home)
        sb.addWidget(self.nav_opponents)
        sb.addWidget(self.nav_maps)
        sb.addWidget(self.nav_scrape)
        sb.addWidget(self.nav_exports)

        sb.addStretch(1)

        footer = QtWidgets.QFrame()
        footer.setObjectName("SidebarFooter")
        fl = QtWidgets.QVBoxLayout(footer)
        fl.setContentsMargins(10, 10, 10, 10)
        fl.setSpacing(8)

        self.btn_popout = QtWidgets.QPushButton("Open current page in new window")
        self.btn_popout.setObjectName("SidebarFooterButton")
        self.btn_popout.setFixedHeight(36)
        fl.addWidget(self.btn_popout)

        self.btn_output = QtWidgets.QPushButton("↗  Output folder")
        self.btn_output.setObjectName("SidebarFooterButton")
        self.btn_output.setFixedHeight(36)
        fl.addWidget(self.btn_output)

        sb.addWidget(footer)

        return sidebar

    def _nav_button(self, text: str) -> QtWidgets.QToolButton:
        b = QtWidgets.QToolButton()
        b.setObjectName("NavButton")
        b.setText(text)
        b.setCheckable(True)
        b.setToolButtonStyle(Qt.ToolButtonTextOnly)
        return b

    def _wire_signals(self) -> None:
        self.nav_home.clicked.connect(lambda: self._set_page(0))
        self.nav_opponents.clicked.connect(lambda: self._set_page(1))
        self.nav_maps.clicked.connect(lambda: self._set_page(2))
        self.nav_scrape.clicked.connect(lambda: self._set_page(3))
        self.nav_exports.clicked.connect(lambda: self._set_page(4))

        self.btn_popout.clicked.connect(self._open_popout)
        self.btn_output.clicked.connect(self.on_open_output_folder)

        self.page_home.build_requested.connect(self.on_build_only)
        self.page_home.import_copy_requested.connect(self.on_import_csv_copy)
        # dataset picker
        self.page_home.btn_refresh_datasets.clicked.connect(lambda: self.refresh_datasets(self.current_csv_path))
        self.page_home.btn_load_dataset.clicked.connect(self.load_selected_dataset)
        self.refresh_datasets(self.current_csv_path)
        self.page_home.open_output_requested.connect(self.on_open_output_folder)
        self.page_home.open_png_requested.connect(self.on_open_png)
        self.page_home.open_report_requested.connect(self.on_open_report)
        self.page_home.player_name_changed.connect(self._on_player_changed)
        self.page_home.ranked_only_changed.connect(self._on_ranked_only_changed)

        self.page_scrape.sig_scrape_all.connect(self._scrape_all)
        self.page_scrape.sig_scrape_recent.connect(self._scrape_recent)
        self.page_scrape.sig_continue.connect(self._scrape_continue)
        self.page_scrape.sig_cancel.connect(self._scrape_cancel)

    def _set_page(self, idx: int) -> None:
        idx = max(0, min(idx, self.stack.count() - 1))
        self.stack.setCurrentIndex(idx)
        self._sync_nav_checks(idx)

    def _sync_nav_checks(self, idx: int) -> None:
        buttons = [self.nav_home, self.nav_opponents, self.nav_maps, self.nav_scrape, self.nav_exports]
        for i, b in enumerate(buttons):
            b.setChecked(i == idx)

    def _open_popout(self) -> None:
        idx = self.stack.currentIndex()
        w = AppWindow(start_page=idx)

        w.current_csv_path = self.current_csv_path
        w.player_name = self.player_name
        w.ranked_only = self.ranked_only

        w.page_home.player_edit.setText(w.player_name)
        w.page_home.ranked_only.setChecked(w.ranked_only)

        w._refresh_home()
        w.show()

    # ---------- Theme ----------

    
    # ---------- Theme ----------
    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            /* =========================
               ECLIPSE v5 – closer to render
               Deep blue primary, purple/pink secondary
            ========================= */

            QWidget { font-family: "Segoe UI", Arial; font-size: 13px; }

            #Root {
                background: qradialgradient(cx:0.25, cy:0.0, radius: 1.2,
                    fx:0.25, fy:0.0,
                    stop:0 rgba(65, 82, 255, 0.14),
                    stop:0.40 rgba(11, 18, 32, 1.0),
                    stop:1 rgba(7, 11, 18, 1.0));
            }

            #Sidebar {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 rgba(10,16,28,0.96),
                    stop:1 rgba(7,11,18,0.96));
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 22px;
            }

            #Brand { color:#EAF0FF; font-size:22px; font-weight:900; }
            #BrandSub { color: rgba(185,200,230,0.70); font-size:12px; margin-top:-6px; }

            #NavButton {
                color: rgba(232,240,255,0.92);
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 14px;
                padding: 11px 12px;
                text-align: left;
                font-weight: 650;
            }
            #NavButton:hover {
                background: rgba(99,102,241,0.12);
                border-color: rgba(99,102,241,0.40);
            }
            #NavButton:checked {
                background: rgba(37,99,235,0.20);
                border-color: rgba(37,99,235,0.85);
                color: #D6E4FF;
            }

            #HeaderCard, #Card, #KPIBadge {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 rgba(17,27,45,0.92),
                    stop:1 rgba(10,16,28,0.92));
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 22px;
            }

            #HeaderTitle { color:#F5F8FF; font-size:20px; font-weight:900; }
            #CardTitle { color:#F1F5FF; font-size:16px; font-weight:900; }
            #CardSub { color: rgba(170,185,215,0.72); font-size:12px; }

            QLineEdit {
                color:#EAF0FF;
                background: rgba(7,11,18,0.55);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 16px;
                padding: 8px 14px;
            }
            QLineEdit:focus { border-color: rgba(37,99,235,0.90); }

            QComboBox {
                color:#EAF0FF;
                background: rgba(7,11,18,0.55);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 14px;
                padding: 6px 10px;
            }
            QComboBox QAbstractItemView {
                background: #0B1220;
                color: #EAF0FF;
                border: 1px solid rgba(255,255,255,0.10);
                selection-background-color: rgba(37,99,235,0.25);
            }

            QPushButton { border-radius: 14px; padding: 10px 14px; font-weight: 750; }

            #PrimaryButton {
                color:#EAF0FF;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #1D4ED8,
                    stop:1 #3B82F6);
                border: 1px solid rgba(59,130,246,0.75);
            }
            #PrimaryButton:hover { border-color: rgba(214,228,255,0.92); }

            #SecondaryButton {
                color: rgba(232,240,255,0.92);
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
            }
            #SecondaryButton:hover {
                background: rgba(217,70,239,0.08);
                border-color: rgba(217,70,239,0.75);
                color: #F0ABFC;
            }

            #TertiaryButton {
                color: rgba(200,210,230,0.92);
                background: rgba(255,255,255,0.02);
                border: 1px solid rgba(255,255,255,0.08);
            }
            #TertiaryButton:hover { border-color: rgba(99,102,241,0.60); color: #D6E4FF; }

            #KPIBadgeTitle { color: rgba(170,185,215,0.78); font-size:12px; }
            #KPIBadgeValue { color: #D6E4FF; font-size:22px; font-weight:950; }

            #LogBox {
                color: rgba(210,225,255,0.92);
                background: rgba(7,11,18,0.70);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 18px;
                padding: 10px;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                font-size: 12px;
            }

            #PreviewImage {
                color: rgba(170,185,215,0.72);
                background: rgba(7,11,18,0.55);
                border: 1px dashed rgba(59,130,246,0.38);
                border-radius: 18px;
                padding: 12px;
            }

            QTableWidget {
                background: rgba(7,11,18,0.72);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 18px;
                color: rgba(232,240,255,0.92);
                gridline-color: rgba(255,255,255,0.06);
            }
            QTableWidget::item { padding: 6px 8px; }
            QTableWidget::item:selected { background: rgba(59,130,246,0.22); }

            QHeaderView::section {
                background: rgba(17,27,45,0.92);
                color: rgba(220,230,255,0.92);
                border: 1px solid rgba(255,255,255,0.06);
                padding: 8px 10px;
                font-weight: 850;
            }
            """
        )


    # ---------- Logging / status ----------
    def log(self, text: str) -> None:
        self.page_home.append_log(text)

    def _log_startup(self) -> None:
        self.log(f"Started. Project root: {self.paths.root_dir}")
        self.log(f"Output dir: {self.paths.output_dir}")
        self.log(f"PNG: {self.paths.png_path}")
        self.log(f"Report: {self.paths.report_path}")

        if self.current_csv_path and self.current_csv_path.exists():
            self.log(f"✅ Found existing CSV: {self.current_csv_path}")
        else:
            self.log("⚠️ No CSV found yet. Use Import CSV (copy).")

        if not _SCRAPER_AVAILABLE:
            self.log("⚠️ Scraper core not loaded. Ensure core/poxnora_scraper.py exists and Playwright is installed.")

        self._update_status()

    def _on_player_changed(self, name: str) -> None:
        self.player_name = name.strip()
        if self.player_name:
            self.log(f"Player set: {self.player_name}")
        self._update_status()
        self._refresh_home()
        self._refresh_opponents()
        self._refresh_maps()

    def _on_ranked_only_changed(self, enabled: bool) -> None:
        self.ranked_only = bool(enabled)
        self.log(f"Ranked 1v1 only: {'ON' if self.ranked_only else 'OFF'}")
        self._update_status()
        self._refresh_home()
        self._refresh_opponents()
        self._refresh_maps()

    def _update_status(self) -> None:
        csv_line = str(self.current_csv_path) if (self.current_csv_path and self.current_csv_path.exists()) else "—"
        ready = "Ready." if csv_line != "—" else "Waiting for CSV."
        ranked = "ON" if self.ranked_only else "OFF"
        player = self.player_name if self.player_name else "—"
        pw = "playwright ✅" if _SCRAPER_AVAILABLE else "playwright ❌"

        self.page_home.set_status(
            f"{ready}\n\n"
            f"Player: {player}\n"
            f"Ranked 1v1 only: {ranked}\n\n"
            f"CSV: {csv_line}\n"
            f"Scraper: {pw}\n"
        )

    # ---------- Data load / refresh ----------

    def _load_df_if_possible(self, csv_path: Optional[Path]) -> Any:
        """Load CSV deterministically via core.dataset.load_csv.

        This is the ONLY place the UI should read a CSV from disk.
        Everything else should use self.active_df.
        """
        if csv_path is None or not csv_path.exists():
            return None
        try:
            from core.dataset import load_csv  # type: ignore
            return load_csv(csv_path)
        except Exception as e:
            self.log(f"Dataset load failed: {e}")
            self.log(traceback.format_exc())
            return None

    def _ensure_active_dataset_loaded(self) -> None:
        """Ensure self.active_df matches self.current_csv_path."""
        if not (self.current_csv_path and self.current_csv_path.exists()):
            self.active_df = None
            self._active_csv_loaded = None
            return

        if self._active_csv_loaded == self.current_csv_path and self.active_df is not None:
            return

        df = self._load_df_if_possible(self.current_csv_path)
        self.active_df = df
        # v22: dataset owner drives the player context (prevents mismatched top-bar name)
        try:
            inferred_owner = _infer_player_from_winloss(df) or ''
            if inferred_owner:
                self.player_name = inferred_owner
                try:
                    self.page_home.player_edit.blockSignals(True)
                    self.page_home.player_edit.setText(inferred_owner)
                finally:
                    self.page_home.player_edit.blockSignals(False)
                self._ensure_player_edit_editable()
        except Exception:
            pass
        self._active_csv_loaded = self.current_csv_path
        self._dataset_version += 1

    def _set_active_dataset(self, csv_path: Path, *, trigger_build: bool = False) -> None:
        self.current_csv_path = Path(csv_path)
        self._active_csv_loaded = None
        self.active_df = None
        self._ensure_active_dataset_loaded()
        self.log(f"✅ Active dataset set: {self.current_csv_path}")
        self.refresh_datasets(prefer=self.current_csv_path)
        self._refresh_home()
        self._refresh_opponents()
        self._refresh_maps()
        if trigger_build:
            self.on_build_only()

    def refresh_datasets(self, prefer: Optional[Path] = None) -> None:
        if not hasattr(self.page_home, "combo_dataset"):
            return
        combo = self.page_home.combo_dataset
        out_dir = self.paths.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        csvs = list(out_dir.glob("*.csv"))
        csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        combo.blockSignals(True)
        combo.clear()
        for p in csvs:
            combo.addItem(p.name, str(p))
        combo.blockSignals(False)
        cur = str((prefer or self.current_csv_path) or "")
        if cur:
            idx = combo.findData(cur)
            if idx >= 0:
                combo.setCurrentIndex(idx)


    def load_selected_dataset(self) -> None:
        if not hasattr(self.page_home, "combo_dataset"):
            return
        path_str = self.page_home.combo_dataset.currentData()
        if not path_str:
            return
        p = Path(str(path_str))
        if not p.exists():
            self.log(f"Dataset not found: {p}")
            return
        # Build deterministically from the selected dataset
        self._set_active_dataset(p, trigger_build=False)


    def _refresh_home(self) -> None:
        self._ensure_active_dataset_loaded()
        df = self.active_df
        self.page_home.set_kpis(_compute_kpis(df, self.player_name, self.ranked_only))
        self.page_home.set_live_dashboard(df, self.player_name, self.ranked_only)
        self.page_home.set_preview_png(self.paths.png_path)
        self._update_status()

    def _refresh_opponents(self) -> None:
        self._ensure_active_dataset_loaded()
        df = self.active_df
        try:
            opp_rows, _ = _compute_opponents_maps(df, self.player_name, self.ranked_only, top_n=999999)
            self.page_opponents.set_rows(opp_rows)
        except Exception:
            pass

    def _refresh_maps(self) -> None:
        self._ensure_active_dataset_loaded()
        df = self.active_df
        try:
            _, map_rows = _compute_opponents_maps(df, self.player_name, self.ranked_only, top_n=999999)
            self.page_maps.set_rows(map_rows)
        except Exception:
            pass

    def on_import_csv_copy(self) -> None:
        start_dir = str(self.paths.output_dir if self.paths.output_dir.exists() else self.paths.root_dir)
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import CSV (copy into output folder)",
            start_dir,
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not file_path:
            return

        src = Path(file_path)
        if not src.exists():
            self.log("Import cancelled: selected file does not exist.")
            return

        self.paths.output_dir.mkdir(parents=True, exist_ok=True)

        dest = self.paths.output_dir / src.name
        if dest.exists():
            base = src.stem
            ext = src.suffix
            i = 2
            while True:
                cand = self.paths.output_dir / f"{base}_{i}{ext}"
                if not cand.exists():
                    dest = cand
                    break
                i += 1

        try:
            shutil.copy2(src, dest)
            self._set_active_dataset(dest, trigger_build=False)
        except Exception as e:
            self.log(f"Import CSV failed: {e}")
            self.log(traceback.format_exc())
            QMessageBox.warning(self, "Import CSV failed", str(e))


    def on_build_only(self) -> None:
        # --- AUTO LOAD SELECTED DATASET FIRST ---
        if self.page_home.combo_dataset.currentIndex() >= 0:
            self.load_selected_dataset()

        # Ensure we have an active dataset
        if not (self.current_csv_path and self.current_csv_path.exists()):
            self.current_csv_path = _auto_find_csv(self.paths.root_dir, self.paths.output_dir)
            self._active_csv_loaded = None

        if self.current_csv_path is None or not self.current_csv_path.exists():
            self.log("Build Only: no CSV found. Use Import CSV (copy) or scrape first.")
            self._update_status()
            return

        self._ensure_active_dataset_loaded()
        df = self.active_df
        if df is None:
            self.log("Build Only: failed to read active dataset.")
            self._update_status()
            return

        # Quick sanity
        try:
            rows = int(len(df))
            cols = [str(c) for c in getattr(df, "columns", [])]
        except Exception:
            rows = -1
            cols = []

        self.log(f"Build Only: active CSV: {self.current_csv_path}")
        self.log(f"CSV check: rows={rows}, cols={cols[:12]}")
        if rows <= 0:
            self.log("Build Only: dataset has 0 rows.")
            self._update_status()
            return

        # KPIs + tables ALWAYS from active df
        eff_player = _effective_player_for_df(df, self.player_name)
        if eff_player and (eff_player.strip() != (self.player_name or '').strip()):
            self.log(f"ℹ️ Using player '{eff_player}' for this dataset (player box didn't match dataset).")
            self.player_name = eff_player
            try:
                self.page_home.player_edit.blockSignals(True)
                self.page_home.player_edit.setText(eff_player)
            finally:
                self.page_home.player_edit.blockSignals(False)
        self.page_home.set_kpis(_compute_kpis(df, eff_player, self.ranked_only))
        try:
            opp_rows, map_rows = _compute_opponents_maps(df, self.player_name, self.ranked_only, top_n=999999)
            self.page_opponents.set_rows(opp_rows)
            self.page_maps.set_rows(map_rows)
        except Exception:
            pass
        self._update_status()

        png_builder = _try_import_callable(
            "core.dashboard_png_v26",
            candidates=[
                "build_dashboard_png",
                "build_png",
                "build_dashboard",
                "render_dashboard_png",
                "generate_dashboard_png",
            ],
        )
        report_builder = _try_import_callable(
            "core.report_html_v26",
            candidates=[
                "build_report_html",
                "build_html",
                "build_report",
                "render_report_html",
                "generate_report_html",
            ],
        )

        player = (self.player_name or '').strip() or None

        if png_builder is None:
            self.log("Build Only: core.dashboard_png builder not found.")
            try:
                import importlib
                importlib.invalidate_caches()
                importlib.import_module("core.dashboard_png")
            except Exception as e:
                self.log(f"Import error: {e}")
                self.log(traceback.format_exc())
            return

        # build PNG (Option B first: pass df)
        try:
            built = False
            for call in (
                lambda: png_builder(df=df, output_path=str(self.paths.png_path), player_name=player, ranked_only=self.ranked_only, log=self.log),
                lambda: png_builder(output_path=str(self.paths.png_path), player_name=player, ranked_only=self.ranked_only, log=self.log, df=df),
                lambda: png_builder(csv_path=str(self.current_csv_path), output_path=str(self.paths.png_path), player_name=player),
                lambda: png_builder(str(self.current_csv_path), str(self.paths.png_path)),
                lambda: png_builder(str(self.current_csv_path)),
            ):
                try:
                    call()
                    built = True
                    break
                except TypeError:
                    continue

            if not built:
                self.log("Build Only: PNG builder signature mismatch.")
                return

            self.log(f"✅ Built PNG: {self.paths.png_path}")
            self.page_home.set_preview_png(self.paths.png_path)
        except Exception as e:
            self.log(f"Build Only: PNG build failed: {e}")
            self.log(traceback.format_exc())
            return

        # build report (Option B first: pass df)
        if report_builder is None:
            self.log("Build Only: core.report_html builder not found (skipping report).")
            self._update_status()
            return

        try:
            built = False
            for call in (
                lambda: report_builder(df=df, output_path=self.paths.report_path, player_name=player, ranked_only=self.ranked_only),
                lambda: report_builder(output_path=self.paths.report_path, player_name=player, ranked_only=self.ranked_only, df=df),
                lambda: report_builder(csv_path=Path(self.current_csv_path), output_path=self.paths.report_path, player_name=player),
                lambda: report_builder(csv_path=str(self.current_csv_path), output_path=str(self.paths.report_path), player_name=player),
                lambda: report_builder(str(self.current_csv_path), str(self.paths.report_path)),
                lambda: report_builder(str(self.current_csv_path)),
            ):
                try:
                    call()
                    built = True
                    break
                except TypeError:
                    continue

            if built:
                self.log(f"✅ Built report: {self.paths.report_path}")
            else:
                self.log("Build Only: report builder signature mismatch (skipped).")
        except Exception as e:
            self.log(f"Build Only: report build failed: {e}")
            self.log(traceback.format_exc())

            self._update_status()


    def on_open_output_folder(self) -> None:
        folder = self.paths.output_dir
        if not folder.exists():
            self.log("Open Output Folder: output directory not found.")
            return
        ok = QDesktopServices.openUrl(_to_file_url(folder))
        if not ok:
            self.log("Open Output Folder: failed to open.")

    def on_open_png(self) -> None:
        path = self.paths.png_path
        if not path.exists():
            self.log("Open Dashboard PNG: dashboard.png not found.")
            return
        ok = _safe_open_path(path)
        if not ok:
            self.log("Open Dashboard PNG: failed to open with system handler.")

    def on_open_report(self) -> None:
        path = self.paths.report_path
        if not path.exists():
            self.log("Open Report HTML: report.html not found.")
            return
        ok = QDesktopServices.openUrl(_to_file_url(path))
        if not ok:
            self.log("Open Report HTML: failed to open in browser.")

    # ---------- Scrape controls ----------
    def _scrape_all(self) -> None:
        self._start_scrape(mode="all", target_new=0)

    def _scrape_recent(self, n: int) -> None:
        self._start_scrape(mode="recent", target_new=int(n))

    def _start_scrape(self, mode: str, target_new: int) -> None:
        if not _SCRAPER_AVAILABLE or run_scrape is None or ScrapeConfig is None:
            QMessageBox.warning(
                self,
                "Scraper not available",
                "Scraper core not loaded.\n\nMake sure core/poxnora_scraper.py exists and Playwright is installed:\n"
                "  pip install playwright\n  python -m playwright install chromium",
            )
            self.log("Scrape unavailable: scraper core not loaded.")
            return

        player = (self.player_name or "").strip()
        if not player:
            QMessageBox.information(
                self,
                "Player name required",
                "Enter your player name on the Home page first, then try Scrape again.",
            )
            self._set_page(0)
            self.page_home.player_edit.setFocus()
            return

        out_dir = self.paths.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        player_slug = _slug(self.player_name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = out_dir / f"poxnora_{player_slug}_matches_clean.csv"
        recent_path = out_dir / f"poxnora_{player_slug}_{mode}_{target_new}_{stamp}.csv"
        self.current_csv_path = csv_path
        storage_state = out_dir / "storage_state.json"
        browsers_dir = out_dir / "pw-browsers"

        cfg = ScrapeConfig(  # type: ignore[misc]
            output_dir=out_dir,
            csv_path=Path(csv_path),
            recent_csv_path=recent_path,
            browsers_dir=browsers_dir,
            headless=False,
            remember_login=True,
            storage_state_path=storage_state,
            max_pages=2000,
        )

        # prevent overlapping runs
        if self._scrape_thread is not None and self._scrape_thread.isRunning():
            QMessageBox.information(self, "Scrape already running", "A scrape is already running. Cancel it or wait.")
            return

        self.page_scrape.set_running(True)
        self.page_scrape.set_waiting_for_continue(False)
        self.log(f"Scrape requested: mode={mode}, target_new={target_new}")

        self._scrape_thread = QtCore.QThread(self)
        self._scrape_worker = ScrapeWorker(cfg=cfg, mode=mode, target_new=target_new)
        self._scrape_worker.moveToThread(self._scrape_thread)

        self._scrape_thread.started.connect(self._scrape_worker.run)

        self._scrape_worker.sig_log.connect(self.log)
        self._scrape_worker.sig_status.connect(self.log)
        self._scrape_worker.sig_progress.connect(lambda cur, total: self.log(f"Progress: {cur}/{total}"))

        def enable_continue() -> None:
            self.log("Waiting for login. Open Match History in the browser, then click 'Continue after login'.")
            self.page_scrape.set_waiting_for_continue(True)

        self._scrape_worker.sig_need_continue.connect(enable_continue)

        def on_done(path_str: str) -> None:
            self.page_scrape.set_running(False)
            self.page_scrape.set_waiting_for_continue(False)
            self.current_csv_path = Path(path_str)
            self.log(f"✅ Scrape done: {path_str}")
            self._refresh_home()

        def on_error(msg: str) -> None:
            self.page_scrape.set_running(False)
            self.page_scrape.set_waiting_for_continue(False)
            self.log(f"❌ Scrape error: {msg}")
            QMessageBox.warning(self, "Scrape error", msg)

        self._scrape_worker.sig_done.connect(on_done)
        self._scrape_worker.sig_error.connect(on_error)

        # Safe cleanup: never wait() from inside the worker thread
        def finish_thread() -> None:
            try:
                if self._scrape_thread:
                    self._scrape_thread.quit()
            except Exception:
                pass

        self._scrape_worker.sig_done.connect(lambda _p: QtCore.QTimer.singleShot(0, finish_thread))
        self._scrape_worker.sig_error.connect(lambda _e: QtCore.QTimer.singleShot(0, finish_thread))

        self._scrape_thread.finished.connect(self._scrape_thread.deleteLater)
        self._scrape_thread.start()

    def _scrape_continue(self) -> None:
        if self._scrape_worker is None:
            QMessageBox.information(self, "No scrape running", "Start a scrape first.")
            return
        self.page_scrape.set_waiting_for_continue(False)
        self.log("Continue clicked. Scraping match history...")
        self._scrape_worker.continue_after_login()

    def _scrape_cancel(self) -> None:
        if self._scrape_worker is None:
            return
        self.log("Cancelling scrape...")
        self._scrape_worker.cancel()
        self.page_scrape.set_waiting_for_continue(False)


# -----------------------------
# Entrypoint
# -----------------------------
def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)

    win = AppWindow(start_page=0)
    win.show()

    sys.exit(app.exec())


def run() -> None:
    main()


if __name__ == "__main__":
    run()
