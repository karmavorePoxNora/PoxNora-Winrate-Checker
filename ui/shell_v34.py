from __future__ import annotations

import sys
import re
from typing import List

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from ui.app_qt_v34 import AppWindow, APP_TITLE


# ============================================================
# v7 FULL SKIN additions: robust opponents trend hook + hint suppression
# ============================================================

def _try_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None

def _find_dataframe(obj) -> object | None:
    """Find a pandas DataFrame on obj (or nested common attributes) without hard-coding app internals."""
    pd = _try_import_pandas()
    if pd is None:
        return None

    # direct common names first
    for name in ("active_df", "df_active", "current_df", "dataset_df", "matches_df", "df"):
        if hasattr(obj, name):
            v = getattr(obj, name)
            try:
                if isinstance(v, pd.DataFrame):
                    return v
            except Exception:
                pass

    # scan attributes
    try:
        for v in obj.__dict__.values():
            try:
                if isinstance(v, pd.DataFrame):
                    return v
            except Exception:
                continue
    except Exception:
        pass

    # try a few nested holders
    for name in ("page_home", "page_opponents", "pages", "state"):
        if hasattr(obj, name):
            v = getattr(obj, name)
            try:
                if isinstance(v, pd.DataFrame):
                    return v
            except Exception:
                pass
            try:
                if hasattr(v, "__dict__"):
                    for vv in v.__dict__.values():
                        try:
                            if isinstance(vv, pd.DataFrame):
                                return vv
                        except Exception:
                            continue
            except Exception:
                pass

    return None

def _infer_player_from_df(df) -> str:
    """Deterministically infer the player name from a match dataset (Win/Loss columns)."""
    try:
        import pandas as pd  # type: ignore
        if df is None or len(df) == 0:
            return ""
        cols = {str(c).strip().lower(): c for c in df.columns}
        if "win" not in cols or "loss" not in cols:
            return ""
        w = df[cols["win"]].astype(str)
        l = df[cols["loss"]].astype(str)
        s = pd.concat([w, l], ignore_index=True)
        s = s[s.notna() & (s.astype(str).str.strip() != "")]
        if len(s) == 0:
            return ""
        # Most frequent name across Win/Loss is the tracked player in this dataset.
        return str(s.value_counts().idxmax()).strip()
    except Exception:
        return ""

def _find_player_text(win: QtWidgets.QWidget) -> str:
    """Find the *actual* active player from UI state (no guessing / no random fallback).

    Order:
      1) Explicit win.page_home.player_edit (or close variants)
      2) Known attributes (active_player, current_player)
      3) Placeholder-based scan
      4) Infer from active_df Win/Loss (deterministic), and if possible write it back into the UI
    """
    # 1) Home page player edit (preferred)
    try:
        ph = getattr(win, "page_home", None)
        for name in ("player_edit", "edit_player", "playerLineEdit", "le_player"):
            e = getattr(ph, name, None) if ph is not None else None
            if isinstance(e, QtWidgets.QLineEdit):
                t = e.text().strip()
                if t:
                    return t
    except Exception:
        pass

    # 2) Explicit attributes if present
    for name in ("player_name", "active_player", "current_player"):
        if hasattr(win, name):
            try:
                t = str(getattr(win, name) or "").strip()
                if t:
                    return t
            except Exception:
                pass

    # 3) Likely line edits (by placeholder)
    edits = win.findChildren(QtWidgets.QLineEdit)
    for e in edits:
        try:
            ph = (e.placeholderText() or "").lower()
            if "player" in ph or "karmavore" in ph:
                t = e.text().strip()
                if t:
                    return t
        except Exception:
            continue

    # 4) Deterministic inference from active_df
    try:
        df = getattr(win, "active_df", None)
        guess = _infer_player_from_df(df)
        if guess:
            # write back to home player edit if available
            ph = getattr(win, "page_home", None)
            e = getattr(ph, "player_edit", None) if ph is not None else None
            if isinstance(e, QtWidgets.QLineEdit) and not e.text().strip():
                e.setText(guess)
            return guess
    except Exception:
        pass

    return ""

def _find_opponents_table(page: QtWidgets.QWidget) -> QtWidgets.QTableWidget | None:
    # Prefer attribute commonly used
    for name in ("table", "tbl", "table_opponents"):
        if hasattr(page, name):
            t = getattr(page, name)
            if isinstance(t, QtWidgets.QTableWidget):
                return t

    tables = page.findChildren(QtWidgets.QTableWidget)
    if not tables:
        return None

    # Prefer one whose first header mentions opponent
    for t in tables:
        try:
            h0 = t.horizontalHeaderItem(0)
            if h0 and "opponent" in (h0.text() or "").lower():
                return t
        except Exception:
            pass

    # Otherwise choose the one with most rows
    try:
        return max(tables, key=lambda x: x.rowCount())
    except Exception:
        return tables[0]

def _suppress_trend_hint_text(page: QtWidgets.QWidget) -> None:
    """Hide any labels that claim a trend popup exists, unless we successfully hook it."""
    for lab in page.findChildren(QtWidgets.QLabel):
        try:
            txt = (lab.text() or "").lower()
            if ("double" in txt and "click" in txt and "trend" in txt) or ("double-click" in txt and "trend" in txt):
                lab.hide()
            # also hide very explicit "double click for pop-out" style helpers
            if "double" in txt and "click" in txt and ("pop" in txt or "popup" in txt):
                lab.hide()
        except Exception:
            continue

def _call_existing_trend_handler(win: QtWidgets.QWidget, page: QtWidgets.QWidget, opponent: str, row: int) -> bool:
    # v21: always use our deterministic popup so 'Last 50/100/200' is MOST RECENT
    return False


# Optional SVG renderer (falls back gracefully)
try:
    from PySide6 import QtSvg  # type: ignore
except Exception:
    QtSvg = None  # type: ignore


# -------------------------
# Inline SVG icon pack
# -------------------------
_ICON_COLOR = "#EAF0FF"

def _svg_icon(svg: str, size: int = 18) -> QtGui.QIcon:
    if QtSvg is None:
        return QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode("utf-8")))
    pm = QtGui.QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    renderer.render(p)
    p.end()
    return QtGui.QIcon(pm)

def icon_home() -> QtGui.QIcon:
    return _svg_icon(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">
    <path d="M4 10.5 12 4l8 6.5V20a1.5 1.5 0 0 1-1.5 1.5H5.5A1.5 1.5 0 0 1 4 20v-9.5Z"
          stroke="{_ICON_COLOR}" stroke-width="2" stroke-linejoin="round"/>
    <path d="M9.5 21v-6.5h5V21" stroke="{_ICON_COLOR}" stroke-width="2" stroke-linejoin="round"/>
    </svg>''')

def icon_swords() -> QtGui.QIcon:
    return _svg_icon(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">
    <path d="M3 21l6-6" stroke="{_ICON_COLOR}" stroke-width="2" stroke-linecap="round"/>
    <path d="M5 5l7 7" stroke="{_ICON_COLOR}" stroke-width="2" stroke-linecap="round"/>
    <path d="M19 3l-7 7" stroke="{_ICON_COLOR}" stroke-width="2" stroke-linecap="round"/>
    <path d="M21 21l-6-6" stroke="{_ICON_COLOR}" stroke-width="2" stroke-linecap="round"/>
    </svg>''')

def icon_map() -> QtGui.QIcon:
    return _svg_icon(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">
    <path d="M10 20 4 18V6l6 2 4-2 6 2v12l-6-2-4 2Z" stroke="{_ICON_COLOR}" stroke-width="2" stroke-linejoin="round"/>
    <path d="M10 8v12M14 6v12" stroke="{_ICON_COLOR}" stroke-width="2" stroke-linecap="round"/>
    </svg>''')

def icon_magnet() -> QtGui.QIcon:
    return _svg_icon(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">
    <path d="M7 3v7a5 5 0 0 0 10 0V3" stroke="{_ICON_COLOR}" stroke-width="2" stroke-linecap="round"/>
    <path d="M7 10H3" stroke="{_ICON_COLOR}" stroke-width="2" stroke-linecap="round"/>
    <path d="M21 10h-4" stroke="{_ICON_COLOR}" stroke-width="2" stroke-linecap="round"/>
    </svg>''')

def icon_box() -> QtGui.QIcon:
    return _svg_icon(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">
    <path d="M21 8l-9 5-9-5" stroke="{_ICON_COLOR}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M3 8l9-5 9 5v10l-9 5-9-5V8Z" stroke="{_ICON_COLOR}" stroke-width="2" stroke-linejoin="round"/>
    </svg>''')


# -------------------------
# Sparkline rendering
# -------------------------
def _spark_pixmap(values: List[float], w: int = 130, h: int = 26) -> QtGui.QPixmap:
    pm = QtGui.QPixmap(w, h)
    pm.fill(Qt.transparent)
    if not values:
        return pm

    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0

    pts = []
    n = len(values)
    for i, v in enumerate(values):
        x = 2 + (w - 4) * (i / max(1, n - 1))
        t = (v - vmin) / (vmax - vmin)
        y = (h - 3) - t * (h - 6)
        pts.append(QtCore.QPointF(x, y))

    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)

    path = QtGui.QPainterPath()
    path.moveTo(pts[0])
    for pt in pts[1:]:
        path.lineTo(pt)
    path.lineTo(pts[-1].x(), h - 2)
    path.lineTo(pts[0].x(), h - 2)
    path.closeSubpath()

    grad = QtGui.QLinearGradient(0, 0, 0, h)
    grad.setColorAt(0.0, QtGui.QColor(217, 70, 239, 70))
    grad.setColorAt(1.0, QtGui.QColor(59, 130, 246, 0))
    p.fillPath(path, grad)

    glow_pen = QtGui.QPen(QtGui.QColor(217, 70, 239, 120), 5)
    glow_pen.setCapStyle(Qt.RoundCap)
    glow_pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(glow_pen)
    p.drawPolyline(pts)

    line_pen = QtGui.QPen(QtGui.QColor(234, 240, 255, 220), 2.0)
    line_pen.setCapStyle(Qt.RoundCap)
    line_pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(line_pen)
    p.drawPolyline(pts)

    p.end()
    return pm




class OpponentTrendDialog(QtWidgets.QDialog):
    """Simple, premium-styled fallback dialog only when the app's real popup isn't reachable."""
    def __init__(self, parent: QtWidgets.QWidget, title: str, message: str):
        super().__init__(parent)
        try:
            self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        except Exception:
            pass
        try:
            if parent is not None:
                self.setStyleSheet(parent.styleSheet())
        except Exception:
            pass
        self.setWindowTitle(title)
        self.resize(860, 520)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)

        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(12)

        hdr = QtWidgets.QHBoxLayout()
        h = QtWidgets.QLabel(title)
        h.setObjectName("CardTitle")
        hdr.addWidget(h, 1)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        hdr.addWidget(close_btn, 0, Qt.AlignRight)
        lay.addLayout(hdr)

        # Body: if message contains window lines, render 4 glass stat cards
        lines = [ln.strip() for ln in (message or "").splitlines() if ln.strip()]
        if any(ln.lower().startswith("overall:") for ln in lines) and any("last 50" in ln.lower() for ln in lines):
            grid = QtWidgets.QGridLayout()
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(14)

            def _pill_label(text: str, kind: str) -> QtWidgets.QLabel:
                lab = QtWidgets.QLabel(text)
                lab.setObjectName("WinPill")
                lab.setAlignment(Qt.AlignCenter)
                if kind == "green":
                    bg = "rgba(69,240,181,0.16)"
                    bd = "rgba(69,240,181,0.42)"
                elif kind == "red":
                    bg = "rgba(255,92,128,0.16)"
                    bd = "rgba(255,92,128,0.42)"
                else:
                    bg = "rgba(200,107,255,0.16)"
                    bd = "rgba(200,107,255,0.42)"
                lab.setStyleSheet(
                    f"padding:6px 10px; border-radius:14px; "
                    f"background:{bg}; border:1px solid {bd}; "
                    f"font-weight:700; font-size:12px; color: rgba(236,242,255,0.95);"
                )
                return lab

            def _mk_card(title_txt: str, value_txt: str) -> QtWidgets.QFrame:
                fr = QtWidgets.QFrame()
                fr.setObjectName("MiniCard")
                vl = QtWidgets.QVBoxLayout(fr)
                vl.setContentsMargins(14, 12, 14, 12)
                vl.setSpacing(8)

                top = QtWidgets.QHBoxLayout()
                top.setSpacing(10)

                t = QtWidgets.QLabel(title_txt)
                t.setObjectName("MiniCardTitle")
                top.addWidget(t, 1)

                wr = None
                try:
                    mwr = re.search(r"([0-9]+\.?[0-9]*)\s*%", value_txt)
                    if mwr:
                        wr = float(mwr.group(1))
                except Exception:
                    wr = None

                kind = "purple"
                if wr is not None:
                    kind = "green" if wr >= 50.0 else "red"
                pill = _pill_label(f"{wr:.1f}%" if wr is not None else "—", kind)
                top.addWidget(pill, 0, Qt.AlignRight)

                vl.addLayout(top)

                sub = QtWidgets.QLabel(value_txt)
                sub.setObjectName("MiniCardValue")
                sub.setWordWrap(True)
                sub.setAlignment(Qt.AlignLeft | Qt.AlignTop)
                vl.addWidget(sub, 1)

                return fr

            def _pick(prefix: str) -> str:
                for ln in lines:
                    if ln.lower().startswith(prefix):
                        return ln.split(":", 1)[1].strip()
                return "—"

            cards = [
                ("Overall", _pick("overall")),
                ("Last 50", _pick("last 50")),
                ("Last 100", _pick("last 100")),
                ("Last 200", _pick("last 200")),
            ]
            for i, (tt, vv) in enumerate(cards):
                grid.addWidget(_mk_card(tt, vv), i // 2, i % 2)

            lay.addLayout(grid, 1)
        else:
            body = QtWidgets.QLabel(message)
            body.setObjectName("CardSub")
            body.setWordWrap(True)
            body.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lay.addWidget(body, 1)

        outer.addWidget(card)

class ShellWindow(AppWindow):
    def __init__(self, start_page: int = 0):
        super().__init__(start_page=start_page)
        self._apply_eclipse_qss()
        self._apply_card_shadows()
        self._upgrade_sidebar_nav()
        self._fix_tables_palette()
        self._upgrade_kpis_with_sparks()
        self._restore_opponents_trend_hook()

    def _apply_eclipse_qss(self) -> None:

        qss = r'''
        /* =========================
           ECLIPSE v7 — Full Skin
           ========================= */

        QWidget {
            font-family: "Segoe UI";
            color: rgba(236,242,255,0.92);
        }

        /* App background: deep indigo + subtle radial glow */
        QMainWindow, QWidget#ShellRoot {
            background:
                qradialgradient(cx:0.15, cy:0.10, radius:1.2,
                    stop:0 rgba(32, 48, 92, 190),
                    stop:0.35 rgba(14, 22, 44, 230),
                    stop:1 rgba(6, 8, 14, 255));
        }

        /* Generic card */
        QFrame#Card {
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 rgba(26, 34, 58, 205),
                stop:0.5 rgba(14, 18, 30, 210),
                stop:1 rgba(8, 10, 18, 225));
            border: 1px solid rgba(255,255,255,0.10);
            border-top: 1px solid rgba(255,255,255,0.16);
            border-radius: 22px;
        }
        QLabel#CardTitle { font-size: 15px; font-weight: 900; }
        QLabel#CardSub { color: rgba(170,186,212,0.84); }

        /* Inputs */
        QLineEdit, QComboBox {
            background: rgba(6, 8, 14, 0.30);
            border: 1px solid rgba(255,255,255,0.10);
            border-top: 1px solid rgba(255,255,255,0.14);
            border-radius: 14px;
            padding: 10px 12px;
            color: rgba(236,242,255,0.94);
        }
        QLineEdit:focus, QComboBox:focus {
            border-color: rgba(96,165,250,0.55);
            background: rgba(10, 14, 24, 0.36);
        }
        QComboBox::drop-down { border: 0px; }

        /* ComboBox popup (was white) */
        QComboBox QAbstractItemView {
            background: rgba(10, 14, 24, 0.98);
            border: 1px solid rgba(255,255,255,0.10);
            selection-background-color: rgba(59,130,246,0.22);
            selection-color: rgba(236,242,255,0.98);
            color: rgba(236,242,255,0.92);
            outline: 0;
        }
        QComboBox QAbstractItemView::item {
            padding: 8px 10px;
            border-radius: 8px;
        }
        QComboBox QAbstractItemView::item:hover {
            background: rgba(59,130,246,0.14);
        }

        /* Table headers */
        QHeaderView::section {
            background: rgba(12, 18, 32, 0.92);
            color: rgba(210, 222, 245, 0.92);
            border: 0px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding: 8px 10px;
        }


        /* Buttons */
        QPushButton {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            border-top: 1px solid rgba(255,255,255,0.16);
            border-radius: 12px;
            padding: 9px 14px;
            color: rgba(236,242,255,0.92);
        }
        QPushButton:hover {
            background: rgba(59,130,246,0.12);
            border-color: rgba(96,165,250,0.55);
        }
        QPushButton:pressed { background: rgba(59,130,246,0.18); }

        /* Segmented pills (All games / Ranked only) */
        QToolButton#SegPill {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.10);
            border-top: 1px solid rgba(255,255,255,0.16);
            border-radius: 14px;
            padding: 8px 12px;
            color: rgba(236,242,255,0.90);
        }
        QToolButton#SegPill:hover {
            background: rgba(255,255,255,0.06);
            border-color: rgba(214,228,255,0.22);
        }
        QToolButton#SegPill:checked {
            background: rgba(59,130,246,0.16);
            border-color: rgba(96,165,250,0.55);
        }

        /* Sidebar */
        QFrame#Sidebar {
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 rgba(18, 24, 40, 220),
                stop:1 rgba(8, 10, 18, 235));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
        }
        QToolButton#NavButton {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-top: 1px solid rgba(255,255,255,0.12);
            border-radius: 14px;
            padding: 10px 12px;
            text-align: left;
            color: rgba(236,242,255,0.92);
        }
        QToolButton#NavButton:hover {
            background: rgba(255,255,255,0.06);
            border-color: rgba(214,228,255,0.22);
        }
        QToolButton#NavButton:checked {
            background: rgba(59,130,246,0.14);
            border-color: rgba(96,165,250,0.55);
        }

        /* KPI tiles */
        QFrame#KPIBadge {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 rgba(22, 32, 58, 210),
                stop:0.45 rgba(12, 16, 28, 225),
                stop:1 rgba(6, 8, 14, 240));
            border: 1px solid rgba(255,255,255,0.08);
            border-top: 1px solid rgba(255,255,255,0.14);
            border-radius: 18px;
        }
        QLabel#KPIBadgeTitle { color: rgba(170,186,212,0.88); font-weight: 800; }
        QLabel#KPIBadgeValue { color: rgba(236,242,255,0.96); font-size: 22px; font-weight: 950; }

        /* Tables — premium dark + subtle blue selection */
        QTableView, QTableWidget {
            background: rgba(8, 10, 18, 0.55);
            alternate-background-color: rgba(10, 14, 24, 0.62);
            gridline-color: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.06);
            border-top: 1px solid rgba(255,255,255,0.10);
            border-radius: 16px;
        }
        QHeaderView::section {
            background: rgba(14,22,40,0.92);
            color: rgba(236,242,255,0.88);
            border: 0px;
            padding: 10px 12px;
            font-weight: 900;
        }
        QTableCornerButton::section { background: rgba(14,22,40,0.92); border:0px; }

        QTableView::item, QTableWidget::item {
            padding: 10px 12px;
            background: transparent;
            color: rgba(236,242,255,0.92);
            border: 0px;
            outline: none;
        }
        QTableView::item:selected, QTableWidget::item:selected {
            background: rgba(59,130,246,0.20);
            border: 0px;
        }

        /* Logs */
        QPlainTextEdit, QTextEdit {
            background: rgba(6, 8, 14, 0.60);
            border: 1px solid rgba(255,255,255,0.08);
            border-top: 1px solid rgba(255,255,255,0.12);
            border-radius: 16px;
            padding: 10px 12px;
            color: rgba(236,242,255,0.90);
            selection-background-color: rgba(59,130,246,0.24);
        }

        /* pyqtgraph widgets should be transparent so the card shows through */
        QGraphicsView { background: transparent; }


        /* ===== V9 Polish Pass ===== */

        /* Dropdown popup (fix white list) */
        QComboBox QAbstractItemView {
            background: rgba(10, 14, 24, 245);
            border: 1px solid rgba(120,160,255,80);
            border-radius: 14px;
            padding: 6px;
            outline: 0;
            selection-background-color: rgba(59,130,246,85);
            selection-color: rgba(240,248,255,235);
        }
        QComboBox QAbstractItemView::item {
            padding: 8px 10px;
            border-radius: 10px;
        }

        /* Primary pill buttons (match reference) */
        QPushButton {
            background: rgba(20, 28, 48, 210);
            border: 1px solid rgba(255,255,255,30);
            border-top: 1px solid rgba(255,255,255,44);
            border-radius: 14px;
            padding: 9px 14px;
        }
        QPushButton:hover {
            border: 1px solid rgba(120,160,255,90);
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 rgba(28, 44, 90, 230),
                stop:1 rgba(22, 30, 54, 210));
        }
        QPushButton:pressed {
            background: rgba(12, 18, 34, 230);
        }
        QPushButton#PrimaryButton, QPushButton[primary="true"] {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 rgba(65, 140, 255, 210),
                stop:1 rgba(120, 70, 255, 170));
            border: 1px solid rgba(160,190,255,85);
        }
        QPushButton#PrimaryButton:hover, QPushButton[primary="true"]:hover {
            border: 1px solid rgba(200,220,255,105);
        }

        /* Segmented pills (All games / 20 / 100) */
        QPushButton[seg="true"] {
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(18, 26, 44, 220);
            border: 1px solid rgba(255,255,255,26);
        }
        QPushButton[seg="true"]:checked {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 rgba(64,156,255,200),
                stop:1 rgba(160,90,255,160));
            border: 1px solid rgba(180,210,255,95);
        }

        /* Tables selection (subtle blue) */
        QTableView::item:selected, QTableWidget::item:selected {
            background: rgba(59,130,246,70);
        }

        /* Trend popup / dialogs */
        QDialog {
            background: transparent;
        }
        QDialog QFrame#Card {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 rgba(22, 30, 52, 245),
                stop:1 rgba(10, 14, 24, 245));
            border: 1px solid rgba(255,255,255,26);
            border-top: 1px solid rgba(255,255,255,44);
            border-radius: 26px;
        }


        /* Dropdown list popup — keep it dark (no white) */
        QComboBox QAbstractItemView {
            background: rgba(12, 18, 32, 0.98);
            color: rgba(235,245,255,0.95);
            selection-background-color: rgba(85,183,255,0.22);
            selection-color: rgba(235,245,255,0.98);
            border: 1px solid rgba(120, 150, 220, 0.25);
            outline: 0;
            padding: 6px;
        }
        QComboBox QAbstractItemView::item {
            padding: 6px 10px;
            border-radius: 6px;
        }


        /* v16 stability */
        QTableView, QTableWidget {
          gridline-color: rgba(0,0,0,0);
        }
        QTableView::item, QTableWidget::item { border: none; }
        QPushButton:disabled {
          color: rgba(220,235,255,120);
          background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 rgba(60,70,95,110),
            stop:1 rgba(35,40,60,110));
          border: 1px solid rgba(255,255,255,22);
        }

'''
        self.setStyleSheet(qss)


    
    def _apply_card_shadows(self) -> None:
        """Add soft drop shadows to cards/KPI badges for the glass look."""
        try:
            from PySide6.QtWidgets import QGraphicsDropShadowEffect
        except Exception:
            return

        def _shadow(widget: QtWidgets.QWidget, blur: int = 56, y: int = 14, alpha: int = 170):
            try:
                eff = QGraphicsDropShadowEffect(widget)
                eff.setBlurRadius(blur)
                eff.setOffset(0, y)
                eff.setColor(QtGui.QColor(0, 0, 0, alpha))
                widget.setGraphicsEffect(eff)
            except Exception:
                pass

        # Cards + KPI badges
        for w in self.findChildren(QtWidgets.QFrame):
            name = w.objectName() or ""
            if name in {"Card", "KPIBadge", "PgCard"}:
                _shadow(w)

    def _upgrade_sidebar_nav(self) -> None:
        def _set(btn: QtWidgets.QToolButton, label: str, ic: QtGui.QIcon):
            btn.setText(label)
            btn.setIcon(ic)
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            btn.setIconSize(QtCore.QSize(18, 18))
            btn.setCursor(Qt.PointingHandCursor)

        try:
            _set(self.nav_home, "Home", icon_home())
            _set(self.nav_opponents, "Opponents", icon_swords())
            _set(self.nav_maps, "Maps", icon_map())
            _set(self.nav_scrape, "Scrape", icon_magnet())
            _set(self.nav_exports, "Exports", icon_box())
        except Exception:
            pass

    def _fix_tables_palette(self) -> None:
        for tbl in self.findChildren(QtWidgets.QTableWidget):
            try:
                tbl.setAlternatingRowColors(True)
                tbl.setShowGrid(False)
                tbl.setFocusPolicy(Qt.NoFocus)
                pal = tbl.palette()
                pal.setColor(QtGui.QPalette.Base, QtGui.QColor(9, 12, 18, 190))
                pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(13, 18, 28, 215))
                pal.setColor(QtGui.QPalette.Text, QtGui.QColor(234, 240, 255, 235))
                tbl.setPalette(pal)
                if tbl.viewport():
                    tbl.viewport().setAutoFillBackground(True)
                    tbl.viewport().setPalette(pal)
                    tbl.viewport().setAttribute(Qt.WA_StyledBackground, True)
            except Exception:
                pass

    def _upgrade_kpis_with_sparks(self) -> None:
        page = getattr(self, "page_home", None)
        if page is None:
            return

        badges = []
        for nm in ("kpi_games", "kpi_wr", "kpi_50", "kpi_100", "kpi_150", "kpi_200"):
            b = getattr(page, nm, None)
            if b is not None:
                badges.append(b)

        for b in badges:
            if b.findChild(QtWidgets.QLabel, "KPISpark") is not None:
                continue
            lay = b.layout()
            if lay is None:
                continue
            spark = QtWidgets.QLabel()
            spark.setObjectName("KPISpark")
            spark.setFixedHeight(28)
            spark.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            lay.addWidget(spark)

        if getattr(page, "_shell_wrapped_set_kpis", False):
            return

        orig = page.set_kpis

        def wrapped(k: dict[str, str]):
            orig(k)
            try:
                self._update_kpi_sparks()
            except Exception:
                pass

        page.set_kpis = wrapped  # type: ignore
        page._shell_wrapped_set_kpis = True  # type: ignore

        try:
            self._update_kpi_sparks()
        except Exception:
            pass

    def _update_kpi_sparks(self) -> None:
        page = getattr(self, "page_home", None)
        df = getattr(self, "active_df", None)
        player = (getattr(self, "player_name", "") or "").strip().lower()
        if page is None or df is None or not player:
            return

        try:
            cols = {str(c).strip().lower(): c for c in df.columns}
            c_win = cols.get("win")
            c_loss = cols.get("loss")
            if not (c_win and c_loss):
                return
            w = df[c_win].astype(str).str.strip().str.lower()
            l = df[c_loss].astype(str).str.strip().str.lower()
            mask = (w == player) | (l == player)
            sub = df[mask]
            if len(sub) < 8:
                return
            wins = (sub[c_win].astype(str).str.strip().str.lower() == player).astype(int).to_numpy(dtype=float)
        except Exception:
            return

        series = wins.tolist()

        def rolling(series: List[float], win: int) -> List[float]:
            out = []
            for i in range(len(series)):
                s = series[max(0, i - win + 1): i + 1]
                out.append(sum(s) / len(s) * 100.0)
            return out

        def tail(vals: List[float], n: int = 60) -> List[float]:
            return vals[-n:] if len(vals) > n else vals

        r20 = tail(rolling(series, 20))
        r50 = tail(rolling(series, 50))
        r100 = tail(rolling(series, 100))
        r150 = tail(rolling(series, 150))
        r200 = tail(rolling(series, 200))
        games = tail([float(i) for i in range(1, len(series) + 1)])

        mapping = {
            "kpi_games": games,
            "kpi_wr": r20,
            "kpi_50": r50,
            "kpi_100": r100,
            "kpi_150": r150,
            "kpi_200": r200,
        }

        for attr, vals in mapping.items():
            badge = getattr(page, attr, None)
            if badge is None:
                continue
            lab = badge.findChild(QtWidgets.QLabel, "KPISpark")
            if lab is None:
                continue
            lab.setPixmap(_spark_pixmap(vals, 130, 26))


    # -------------------------
    # v4: Opponents trend hook
    # -------------------------
    def _restore_opponents_trend_hook(self) -> None:
        page = getattr(self, "page_opponents", None)
        if page is None:
            return

        # Always suppress any "double click for trend" helper labels until hook is live
        _suppress_trend_hint_text(page)

        def _attach():
            try:
                pg = getattr(self, "page_opponents", None)
                if pg is None:
                    return
                tbl = _find_opponents_table(pg)
                if tbl is None:
                    return
                if getattr(tbl, "_shell_v4_trend_hooked", False):
                    return

                def on_dbl(_idx=None):
                    try:
                        row = tbl.currentRow()
                        if row < 0:
                            return
                        it = tbl.item(row, 0)
                        if it is None:
                            return
                        opp = it.text().strip()
                        if not opp:
                            return

                        # Prefer calling the app's real popup, if available
                        if _call_existing_trend_handler(self, pg, opp, row):
                            return

                        # Otherwise: compute minimal vs stats from current df for a clean fallback
                        df = _find_dataframe(self)
                        player = _find_player_text(self)
                        if df is None or not player:
                            OpponentTrendDialog(
                                self,
                                f"Trend vs {opp}",
                                "Trend window is unavailable right now.\n\n"
                                "Reason: no active dataset/player detected from the UI state.\n"
                                "Fix: load/select a dataset + player, then try again.",
                            ).exec()
                            return

                        pd = _try_import_pandas()
                        cols = {str(c).strip().lower(): c for c in df.columns}
                        c_win = cols.get("win")
                        c_loss = cols.get("loss")
                        c_date = cols.get("date")
                        if not (c_win and c_loss):
                            OpponentTrendDialog(
                                self,
                                f"Trend vs {opp}",
                                "Trend window is unavailable for this dataset.\n\n"
                                "Reason: missing Win/Loss columns.",
                            ).exec()
                            return

                        p = player.strip().lower()
                        o = opp.strip().lower()
                        w = df[c_win].astype(str).str.strip().str.lower()
                        l = df[c_loss].astype(str).str.strip().str.lower()
                        sub = df[((w == p) & (l == o)) | ((w == o) & (l == p))].copy()
                        # Ensure MOST RECENT windows are correct: sort by Date if present.
                        try:
                            if c_date is not None:
                                import pandas as pd  # type: ignore
                                dt = pd.to_datetime(sub[c_date], format="%b %d, %Y %I:%M:%S %p", errors="coerce")
                                if dt.notna().sum() < max(3, int(0.10 * len(dt))):
                                    dt = pd.to_datetime(sub[c_date], errors="coerce")
                                sub["_dt_sort"] = dt
                                sub = sub.sort_values("_dt_sort", kind="mergesort").drop(columns=["_dt_sort"], errors="ignore")
                        except Exception:
                            pass
                        if sub.empty:
                            OpponentTrendDialog(self, f"Trend vs {opp}", "No games found vs this opponent.").exec()
                            return

                        wins = (sub[c_win].astype(str).str.strip().str.lower() == p).astype(int).to_numpy(dtype=float)

                        def _wl(arr):
                            g = int(len(arr))
                            wct = int(arr.sum())
                            lct = g - wct
                            wr = (wct / g * 100.0) if g else 0.0
                            return g, wct, lct, wr

                        overall = _wl(wins)
                        last50 = _wl(wins[-50:])
                        last100 = _wl(wins[-100:])
                        last200 = _wl(wins[-200:])

                        msg = (
                            f"Overall: {overall[3]:.1f}% ({overall[1]}-{overall[2]}, {overall[0]} games)\n"
                            f"Last 50: {last50[3]:.1f}% ({last50[1]}-{last50[2]}, {last50[0]} games)\n"
                            f"Last 100: {last100[3]:.1f}% ({last100[1]}-{last100[2]}, {last100[0]} games)\n"
                            f"Last 200: {last200[3]:.1f}% ({last200[1]}-{last200[2]}, {last200[0]} games)\n"
                        )

                        # Date awareness (don't show blank)
                        if c_date is None:
                            msg += "\n(No date data in this dataset.)"

                        OpponentTrendDialog(self, f"Trend vs {opp}", msg).exec()

                    except Exception as e:
                        OpponentTrendDialog(self, "Trend error", f"Could not open trend window:\n{e}").exec()

                tbl.doubleClicked.connect(on_dbl)
                tbl._shell_v4_trend_hooked = True

            except Exception:
                return

        _attach()
        QtCore.QTimer.singleShot(250, _attach)
        QtCore.QTimer.singleShot(900, _attach)



def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    win = ShellWindow(start_page=0)
    win.show()
    sys.exit(app.exec())


def run() -> None:
    main()


if __name__ == "__main__":
    run()
