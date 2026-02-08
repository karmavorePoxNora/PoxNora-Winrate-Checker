# ui/pg_dashboard_v34.py
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

try:
    import numpy as np  # type: ignore
    import pyqtgraph as pg  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
    pg = None  # type: ignore


def _ranked_truthy(v: str) -> bool:
    s = (str(v or "").strip().lower())
    return s in {"1", "true", "yes", "y", "ranked", "on"}


def _safe_month(dt) -> Optional[str]:
    try:
        # dt is numpy/pandas datetime64 or python datetime
        return f"{dt.year:04d}-{dt.month:02d}"
    except Exception:
        return None



def _format_month_labels(m_labels: list[str]) -> list[str]:
    # m_labels like '2024-02' -> "Feb '24"
    out: list[str] = []
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for s in m_labels:
        try:
            ss = str(s)
            y, m = ss.split("-")[0], ss.split("-")[1]
            mi = int(m) - 1
            yy = y[2:4]
            out.append(f"{months[mi]} '{yy}")
        except Exception:
            out.append(str(s))
    return out

def _parse_dates(series) -> Optional["object"]:
    """Parse the dataset Date column reliably (view-layer only).
    Our CSVs commonly use: 'Dec 3, 2025 8:49:16 AM'
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None
    try:
        dt = pd.to_datetime(series, format="%b %d, %Y %I:%M:%S %p", errors="coerce")
        if getattr(dt, "notna", None) is not None and dt.notna().sum() < max(3, int(0.05 * len(dt))):
            dt = pd.to_datetime(series, errors="coerce")
        return dt
    except Exception:
        try:
            return pd.to_datetime(series, errors="coerce")
        except Exception:
            return None



def _infer_player_from_df(df) -> str:
    # Deterministic: most frequent name across Win/Loss columns
    try:
        wins = df["Win"].astype(str)
        loss = df["Loss"].astype(str)
        import pandas as pd  # type: ignore
        counts = pd.concat([wins, loss]).value_counts()
        if len(counts) > 0:
            return str(counts.index[0])
    except Exception:
        pass
    return ""


def _player_mask_and_outcome(df, player: str):
    w = df["Win"].astype(str)
    l = df["Loss"].astype(str)
    pm = (w == player) | (l == player)
    # win from player's perspective
    is_win = (w == player)
    opp = w.where(l == player, l)  # if player lost -> opponent is Win else Loss
    # If player won, opponent is Loss; if player lost, opponent is Win
    opp = df["Loss"].astype(str)
    try:
        opp = df["Loss"].astype(str).where(df["Win"].astype(str) == player, df["Win"].astype(str))
    except Exception:
        pass
    return pm, is_win.astype(int), opp.astype(str)


@dataclass
class _GlowSpec:
    core: QtGui.QColor
    glow: QtGui.QColor
    fill: QtGui.QColor


def _mk_colors(kind: str) -> _GlowSpec:
    if kind == "purple":
        core = QtGui.QColor(194, 99, 255, 255)
        glow = QtGui.QColor(194, 99, 255, 190)
        fill = QtGui.QColor(194, 99, 255, 55)
    else:
        core = QtGui.QColor(64, 156, 255, 255)
        glow = QtGui.QColor(64, 156, 255, 190)
        fill = QtGui.QColor(64, 156, 255, 55)
    return _GlowSpec(core=core, glow=glow, fill=fill)


class _Card(QtWidgets.QFrame):
    def __init__(self, title: str, subtitle: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PgCard")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 14)
        lay.setSpacing(10)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(10)

        tbox = QtWidgets.QVBoxLayout()
        tbox.setSpacing(2)

        t = QtWidgets.QLabel(title)
        t.setObjectName("PgCardTitle")
        s = QtWidgets.QLabel(subtitle)
        s.setObjectName("PgCardSub")
        s.setWordWrap(True)

        tbox.addWidget(t)
        if subtitle:
            tbox.addWidget(s)

        top.addLayout(tbox, 1)
        lay.addLayout(top)

        self.body = QtWidgets.QVBoxLayout()
        self.body.setSpacing(8)
        lay.addLayout(self.body, 1)


class _Tooltip(QtWidgets.QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PgTooltip")
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setVisible(False)
        self.setTextFormat(QtCore.Qt.RichText)
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.setContentsMargins(10, 8, 10, 8)

    def show_at(self, pos: QtCore.QPoint, html: str) -> None:
        self.setText(html)
        self.adjustSize()
        self.move(pos)
        self.setVisible(True)

    def hide_tip(self) -> None:
        self.setVisible(False)


class _GlowPlot(pg.PlotWidget if pg else QtWidgets.QWidget):
    def __init__(self, kind: str, parent=None) -> None:
        if pg:
            super().__init__(parent=parent, background=(10, 14, 24))
        else:
            super().__init__(parent)
            return

        self._kind = kind
        self._colors = _mk_colors(kind)
        self._x = None
        self._y = None

        self.setObjectName("PgPlot")
        self.setMouseEnabled(x=True, y=False)
        self.showGrid(x=True, y=True, alpha=0.18)
        self.getPlotItem().setMenuEnabled(False)
        self.getPlotItem().hideButtons()
        self.getPlotItem().setClipToView(True)

        ax = self.getPlotItem().getAxis("bottom")
        ax.setStyle(tickTextOffset=6)
        ay = self.getPlotItem().getAxis("left")
        ay.setStyle(tickTextOffset=6)

        # Axis + grid styling (ECLIPSE)
        axis_pen = pg.mkPen((120, 140, 175, 90), width=2)
        text_pen = pg.mkPen((190, 205, 235, 90))
        ax.setPen(axis_pen)
        ay.setPen(axis_pen)
        ax.setTextPen(text_pen)
        ay.setTextPen(text_pen)

        self._tooltip = _Tooltip(self)
        self._proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=25, slot=self._on_mouse)

        self._curve_glow_wide = None
        self._curve_glow_mid = None
        self._curve_core = None
        self._fill_item = None

    def set_series(self, x, y, y_min=0.0, y_max=100.0, x_labels: Optional[list[str]] = None, games_override=None) -> None:
        if not pg:
            return

        self.clear()
        self._x = x
        self._y = y
        self._x_labels = x_labels
        self._games_override = games_override

        # Prevent extreme zoom/pan ranges (keeps axes sane)
        try:
            self.setMouseEnabled(x=False, y=False)
            self.setMenuEnabled(False)
            vb = self.getViewBox()
            vb.setMouseEnabled(x=False, y=False)
            vb.setMenuEnabled(False)
            if len(x) > 0:
                xmin = float(x[0])
                xmax = float(x[-1])
                if xmax < xmin:
                    xmin, xmax = xmax, xmin
                vb.setLimits(xMin=xmin, xMax=xmax, yMin=y_min, yMax=y_max)
        except Exception:
            pass

        # view config
        self.setYRange(y_min, y_max, padding=0.08)

        # glow layering
        c = self._colors
        pen_wide = pg.mkPen(c.glow, width=10)
        pen_mid = pg.mkPen(QtGui.QColor(c.glow.red(), c.glow.green(), c.glow.blue(), 170), width=6)
        pen_core = pg.mkPen(QtGui.QColor(c.core.red(), c.core.green(), c.core.blue(), 255), width=4)

        self._curve_glow_wide = self.plot(x, y, pen=pen_wide)
        self._curve_glow_mid = self.plot(x, y, pen=pen_mid)
        self._curve_core = self.plot(x, y, pen=pen_core)

        # subtle points to ensure data is visible even if line is thin on some GPUs
        try:
            sym = pg.mkBrush(QtGui.QColor(c.core.red(), c.core.green(), c.core.blue(), 220))
            self._scatter_item = self.plot(x, y, pen=None, symbol="o", symbolSize=5, symbolBrush=sym, symbolPen=None)
        except Exception:
            self._scatter_item = None

        # gradient fill under curve
        base = pg.PlotDataItem(x, np.full_like(y, y_min), pen=None)

        grad = QtGui.QLinearGradient(0.0, 0.0, 0.0, 1.0)
        grad.setCoordinateMode(QtGui.QGradient.ObjectBoundingMode)
        top = QtGui.QColor(c.core.red(), c.core.green(), c.core.blue(), 85)
        mid = QtGui.QColor(c.core.red(), c.core.green(), c.core.blue(), 35)
        bot = QtGui.QColor(c.core.red(), c.core.green(), c.core.blue(), 0)
        grad.setColorAt(0.0, top)
        grad.setColorAt(0.55, mid)
        grad.setColorAt(1.0, bot)

        fill = pg.FillBetweenItem(self._curve_core, base, brush=QtGui.QBrush(grad))
        self.addItem(fill)
        self._fill_item = fill


        # subtle horizontal guide lines (mock-style)
        try:
            self._guide_lines = []
            for _yv in (50.0, 75.0, 100.0):
                _pen = pg.mkPen((220, 235, 255, 40), width=1, style=QtCore.Qt.PenStyle.DashLine)
                _ln = pg.InfiniteLine(pos=_yv, angle=0, pen=_pen, movable=False)
                self.addItem(_ln)
                self._guide_lines.append(_ln)
        except Exception:
            self._guide_lines = []
        # z-order (keep data above fill/grid)
        try:
            if self._fill_item is not None:
                self._fill_item.setZValue(1)
            if self._curve_glow_wide is not None:
                self._curve_glow_wide.setZValue(2)
            if self._curve_glow_mid is not None:
                self._curve_glow_mid.setZValue(3)
            if self._curve_core is not None:
                self._curve_core.setZValue(4)
            if getattr(self, "_scatter_item", None) is not None:
                self._scatter_item.setZValue(5)
        except Exception:
            pass

        # optional x labels
        if x_labels is not None:
            ax = self.getPlotItem().getAxis("bottom")
            ticks = []
            # Adaptive tick density (prevents label pile-ups on long histories)
            if len(x) > 0:
                max_labels = 9
                step = max(1, int(math.ceil(len(x) / float(max_labels))))
                for i in range(0, len(x), step):
                    ticks.append((x[i], x_labels[i]))
                # Always include the last label (most recent)
                if ticks and ticks[-1][0] != x[-1]:
                    ticks.append((x[-1], x_labels[-1]))
            ax.setTicks([ticks])

    def _on_mouse(self, evt) -> None:
        if not pg or self._x is None or self._y is None:
            return
        pos = evt[0]
        if not self.sceneBoundingRect().contains(pos):
            self._tooltip.hide_tip()
            return
        mouse_point = self.getPlotItem().vb.mapSceneToView(pos)
        mx = mouse_point.x()

        x = self._x
        y = self._y
        if len(x) == 0:
            self._tooltip.hide_tip()
            return

        # nearest index
        idx = int(np.clip(np.searchsorted(x, mx), 0, len(x) - 1))
        if idx > 0 and abs(x[idx - 1] - mx) < abs(x[idx] - mx):
            idx -= 1

        xv = x[idx]
        yv = y[idx]

        # Label + games count (monthly uses games_override; rolling uses xv)
        label = None
        try:
            if getattr(self, '_x_labels', None) is not None and idx < len(self._x_labels):
                label = str(self._x_labels[idx])
        except Exception:
            label = None

        games_val = None
        try:
            go = getattr(self, '_games_override', None)
            if go is not None and idx < len(go):
                games_val = int(go[idx])
        except Exception:
            games_val = None
        if games_val is None:
            try:
                games_val = int(xv)
            except Exception:
                games_val = 0

        title = 'Point' if not label else label
        html = f"""
        <div style='font-size:12px;'>
          <div style='font-weight:600; margin-bottom:4px;'>{title}</div>
          <div>Games: <span style='font-weight:600'>{games_val}</span></div>
          <div>Winrate: <span style='font-weight:600'>{float(yv):.1f}%</span></div>
        </div>
        """

        # place tooltip near mouse (widget coords)
        wpos = self.mapFromGlobal(QtGui.QCursor.pos())
        wpos = wpos + QtCore.QPoint(14, -10)
        # clamp inside
        x0 = max(8, min(wpos.x(), self.width() - self._tooltip.width() - 8))
        y0 = max(8, min(wpos.y(), self.height() - self._tooltip.height() - 8))
        self._tooltip.show_at(QtCore.QPoint(x0, y0), html)


class LiveDashboard(QtWidgets.QWidget):
    """2×2 live analytics board (pyqtgraph). UI-only."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("LiveDashboard")

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        outer.addLayout(grid, 1)

        self.card_roll = _Card("Rolling Winrate", "Rolling winrate (last 200 games)")
        self.plot_roll = _GlowPlot("purple")
        self.card_roll.body.addWidget(self.plot_roll, 1)

        self.card_month = _Card("Monthly Performance", "Last 5 months • winrate (line) + games (bars)")
        self.plot_month = _GlowPlot("blue")
        self.card_month.body.addWidget(self.plot_month, 1)

        self.card_opp = _Card("Top Opponents", "Most common matchups")
        self.tbl_opp = QtWidgets.QTableWidget(0, 3)
        self.tbl_opp.setHorizontalHeaderLabels(["Opponent", "Games", "Winrate"])
        self.tbl_opp.setObjectName("MiniTable")
        self.tbl_opp.verticalHeader().setVisible(False)
        self.tbl_opp.setShowGrid(False)
        self.tbl_opp.setAlternatingRowColors(False)
        self.tbl_opp.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_opp.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_opp.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_opp.horizontalHeader().setStretchLastSection(True)
        self.tbl_opp.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tbl_opp.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tbl_opp.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.card_opp.body.addWidget(self.tbl_opp, 1)

        self.card_map = _Card("Top Maps", "Highest-volume maps")
        self.tbl_map = QtWidgets.QTableWidget(0, 3)
        self.tbl_map.setHorizontalHeaderLabels(["Map", "Games", "Winrate"])
        self.tbl_map.setObjectName("MiniTable")
        self.tbl_map.verticalHeader().setVisible(False)
        self.tbl_map.setShowGrid(False)
        self.tbl_map.setAlternatingRowColors(False)
        self.tbl_map.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_map.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_map.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_map.horizontalHeader().setStretchLastSection(True)
        self.tbl_map.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tbl_map.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tbl_map.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.card_map.body.addWidget(self.tbl_map, 1)

        # readability: prevent clipped rows; keep the mini-tables usable on large datasets
        for _t in (self.tbl_opp, self.tbl_map):
            _t.setMinimumHeight(230)
            _t.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
            _t.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            _t.horizontalHeader().setMinimumHeight(34)
            _t.verticalHeader().setDefaultSectionSize(28)
            _t.setWordWrap(False)


        # v21: remove tables from live view; expand charts
        grid.addWidget(self.card_roll, 0, 0)
        grid.addWidget(self.card_month, 0, 1)
        grid.setRowStretch(0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        try:
            self.card_opp.setVisible(False)
            self.card_map.setVisible(False)
        except Exception:
            pass

        self._install_style()

    def _install_style(self) -> None:
        # Local styling so it looks premium even if global QSS changes.
        self.setStyleSheet("""
        #PgCard {
          background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
            stop:0 rgba(18, 24, 38, 90),
            stop:1 rgba(12, 16, 26, 90));
          border: 1px solid rgba(255, 255, 255, 20);
          border-top: 1px solid rgba(255, 255, 255, 34);
          border-radius: 22px;
        }
        #PgCardTitle { color: rgba(255, 255, 255, 90); font-size: 16px; font-weight: 700; }
        #PgCardSub { color: rgba(190, 205, 235, 90); font-size: 11px; }
        #MiniTable {
          gridline-color: rgba(0,0,0,0);

          background: transparent;
          alternate-background-color: transparent;
          border: none;
          color: rgba(235, 245, 255, 90);
          font-size: 12px;
        }
        #MiniTable QHeaderView::section {
          background: rgba(12, 16, 26, 90);
          color: rgba(210, 224, 245, 90);
          border: none;
          padding: 6px 8px;
          font-weight: 600;
        }
        #MiniTable::item { padding: 8px 8px; border-bottom: 1px solid rgba(255, 255, 255, 14); }
        #MiniTable::item:selected { background: rgba(59, 130, 246, 70); }
        
        QLabel[isPill="true"] {
          padding: 3px 10px;
          border-radius: 999px;
          font-weight: 700;
          color: rgba(235, 245, 255, 90);
          border: 1px solid rgba(120, 160, 255, 70);
          background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 rgba(30, 60, 120, 90),
            stop:1 rgba(40, 110, 255, 90));
        }
        QLabel[isPill="true"][pillKind="cyan"] {
          border: 1px solid rgba(120, 255, 220, 70);
          background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 rgba(20, 100, 90, 90),
            stop:1 rgba(40, 220, 170, 80));
        }

        #PgTooltip {
          background: rgba(12, 16, 26, 235);
          border: 1px solid rgba(140, 180, 255, 220);
          border-radius: 12px;
          color: rgba(235, 245, 255, 255);
        }
        """)

        if pg:
            pg.setConfigOptions(antialias=True)
            # Force dark theme defaults (no white canvases)
            pg.setConfigOption('background', (10, 14, 24))
            pg.setConfigOption('foreground', (210, 222, 245))

    def update_from_dataframe(self, df, player_name: str = "", ranked_only: bool = False) -> None:
        if pg is None or np is None:
            return
        if df is None or len(df) == 0:
            self._clear()
            return

        # normalize & filter
        player = (player_name or "").strip()
        if not player:
            player = _infer_player_from_df(df)

        dff = df
        try:
            if ranked_only and "Ranked" in dff.columns:
                dff = dff[dff["Ranked"].apply(_ranked_truthy)]
        except Exception:
            pass

        # player games only
        try:
            pm, is_win, opp = _player_mask_and_outcome(dff, player)
            dffp = dff[pm].copy()
            is_win = is_win[pm]
            opp = opp[pm]
        except Exception:
            self._clear()
            return

        if len(dffp) == 0:
            self._clear()
            return

        # order by Date if possible (case-insensitive)
        dt = None
        _date_col = None
        try:
            for c in dffp.columns:
                if str(c).strip().lower() in ("date", "datetime", "played", "timestamp"):
                    _date_col = c
                    break
        except Exception:
            _date_col = None
        if _date_col is not None:
            dt = _parse_dates(dffp[_date_col])
        if dt is not None:
            dffp["_dt"] = dt
            dffp = dffp.sort_values("_dt")
        # rolling winrate (LIVE view clamps to most recent 200 games)
        # Keep FULL player history for monthly; clamp only the rolling plot data.
        dffp_full = dffp
        is_win_full = is_win

        dffp_roll = dffp_full
        is_win_roll = is_win_full
        try:
            n_recent = 200
            if len(dffp_full) > n_recent:
                dffp_roll = dffp_full.tail(n_recent).copy()
                is_win_roll = is_win_full.loc[dffp_roll.index]
        except Exception:
            dffp_roll = dffp_full
            is_win_roll = is_win_full

        w = is_win_roll.to_numpy(dtype=float)
        if len(w) == 0:
            self._clear()
            return

        win_window = min(50, int(len(w)))
        if win_window < 2:
            x_roll = np.arange(1, 1 + len(w), dtype=float)
            y_roll = (w * 100.0).astype(float)
        else:
            win50 = np.convolve(w, np.ones(win_window)/float(win_window), mode="valid") * 100.0
            x_roll = np.arange(1, 1 + len(w), dtype=float)

            # Fill the first (win_window-1) points with expanding mean so the chart has a real line from game 1.
            y_roll = np.empty(len(w), dtype=float)
            if len(w) > 0:
                csum = np.cumsum(w, dtype=float)
                # expanding mean for 1..(win_window-1)
                pre = min(win_window - 1, len(w))
                if pre > 0:
                    y_roll[:pre] = (csum[:pre] / np.arange(1, pre + 1, dtype=float)) * 100.0
                # rolling mean once the window is full
                if len(win50):
                    y_roll[win_window-1:] = win50.astype(float)
        # downsample to ~1200 points max
        if len(x_roll) > 1600:
            stride = int(np.ceil(len(x_roll) / 1200))
            x_roll = x_roll[::stride]
            y_roll = y_roll[::stride]

        # monthly
        months = None
        if dt is not None:
            try:
                import pandas as pd  # type: ignore
                months = pd.to_datetime(dffp["_dt"], errors="coerce").dt.to_period("M").astype(str)
            except Exception:
                months = None
        if months is not None:
            dffp["_m"] = months
            try:
                import pandas as pd  # type: ignore

                # LIVE monthly view: last 5 months based on the latest scraped Date in the dataset
                try:
                    latest_dt = pd.to_datetime(dffp_full["_dt"], errors="coerce").dropna().max()
                except Exception:
                    latest_dt = None

                if latest_dt is not None:
                    try:
                        cutoff = latest_dt - pd.DateOffset(months=5)
                        dffp_m = dffp_full[pd.to_datetime(dffp_full["_dt"], errors="coerce") >= cutoff].copy()
                        is_win_m = is_win_full.loc[dffp_m.index]
                    except Exception:
                        dffp_m = dffp
                        is_win_m = is_win
                else:
                    dffp_m = dffp
                    is_win_m = is_win

                # aggregate
                g = dffp_m.assign(_w=is_win_m.values).groupby("_m").agg(games=("_w", "size"), wins=("_w", "sum"))
                g["wr"] = (g["wins"] / g["games"]) * 100.0
                g = g.reset_index()
                m_labels = g["_m"].tolist()
                m_wr = g["wr"].to_numpy(dtype=float)
                m_games = g["games"].to_numpy(dtype=float)
                # Yearly winrate (latest year) for a small badge
                year_wr = None
                try:
                    yy = pd.to_datetime(dffp["_dt"], errors="coerce").dt.year
                    latest_year = int(yy.dropna().max()) if yy.notna().any() else None
                    if latest_year is not None:
                        ym = (yy == latest_year)
                        yw = is_win.values[ym.values]
                        if len(yw) > 0:
                            year_wr = float((yw.sum() / float(len(yw))) * 100.0)
                except Exception:
                    year_wr = None

            except Exception:
                m_labels, m_wr, m_games = [], np.array([]), np.array([])
        else:
            m_labels, m_wr, m_games = [], np.array([]), np.array([])

        # plots
        if len(x_roll) >= 1:
            self.plot_roll.set_series(x_roll, y_roll, y_min=0.0, y_max=100.0)
            try:
                # force 1..N axis span for the last-200 window
                self.plot_roll.setXRange(0.0, float(len(dffp_roll)), padding=0.0)
                axb = self.plot_roll.getAxis("bottom")
                ticks = [(0, "0"), (50, "50"), (100, "100"), (150, "150"), (200, "200")]
                axb.setTicks([ticks])
            except Exception:
                pass
        else:
            self.plot_roll.clear()

        # monthly plot: we fake bars by drawing stepped fill (simple)
        if len(m_wr) > 0:
            x = np.arange(len(m_wr), dtype=float)
            # line
            self.plot_month.set_series(x, m_wr, y_min=0.0, y_max=100.0, x_labels=_format_month_labels(m_labels), games_override=m_games)            # bars overlay on same plot item
            try:
                # normalize games to 0..100 for a subtle background bar
                if len(m_games) > 0 and float(m_games.max()) > 0:
                    gn = (m_games / float(m_games.max())) * 100.0
                    bg = pg.BarGraphItem(x=x, height=gn, width=0.55, brush=pg.mkBrush(QtGui.QColor(64, 156, 255, 35)), pen=None)
                    self.plot_month.addItem(bg)
            except Exception:
                pass
        else:
            self.plot_month.clear()

        # tables
        self._fill_top_tables(dffp, player, is_win, opp)

    def _fill_top_tables(self, dffp, player: str, is_win, opp) -> None:
        try:
            import pandas as pd  # type: ignore
            # opponents
            tmp = pd.DataFrame({"opp": opp.astype(str).values, "w": is_win.values})
            g = tmp.groupby("opp").agg(games=("w", "size"), wins=("w", "sum"))
            g["wr"] = (g["wins"] / g["games"]) * 100.0
            g = g.sort_values(["games"], ascending=False).head(12)
            self._set_table(self.tbl_opp, [(idx, int(r.games), float(r.wr)) for idx, r in g.iterrows()])

            # maps (case-insensitive)
            _map_col = None
            try:
                for c in dffp.columns:
                    if str(c).strip().lower() in ("map", "mapname", "map_name", "arena"):
                        _map_col = c
                        break
            except Exception:
                _map_col = None
            if _map_col is not None:
                mdf = pd.DataFrame({"map": dffp[_map_col].astype(str).values, "w": is_win.values})
                mg = mdf.groupby("map").agg(games=("w", "size"), wins=("w", "sum"))
                mg["wr"] = (mg["wins"] / mg["games"]) * 100.0
                mg = mg.sort_values(["games"], ascending=False).head(12)
                self._set_table(self.tbl_map, [(idx, int(r.games), float(r.wr)) for idx, r in mg.iterrows()])
            else:
                self._set_table(self.tbl_map, [])
        except Exception:
            self._set_table(self.tbl_opp, [])
            self._set_table(self.tbl_map, [])

    def _set_table(self, tbl: QtWidgets.QTableWidget, rows: list[Tuple[str, int, float]]) -> None:
        tbl.setRowCount(0)
        for name, games, wr in rows:
            r = tbl.rowCount()
            tbl.insertRow(r)
            it0 = QtWidgets.QTableWidgetItem(str(name))
            it1 = QtWidgets.QTableWidgetItem(str(games))
            # Winrate pill
            pill = QtWidgets.QLabel(f"{wr:.1f}%")
            pill.setProperty("isPill", True)
            pill.setAlignment(QtCore.Qt.AlignCenter)
            # color hint per table
            if tbl is getattr(self, "tbl_opp", None):
                pill.setProperty("pillKind", "blue")
            else:
                pill.setProperty("pillKind", "cyan")

            try:
                it0.setForeground(QtGui.QBrush(QtGui.QColor(235, 245, 255, 90)))
                it1.setForeground(QtGui.QBrush(QtGui.QColor(235, 245, 255, 90)))
                it2.setForeground(QtGui.QBrush(QtGui.QColor(220, 235, 255, 90)))
                # pseudo "pill" background for winrate
                it2.setBackground(QtGui.QBrush(QtGui.QColor(46, 90, 180, 70)))
            except Exception:
                pass
            it0.setFlags(it0.flags() ^ QtCore.Qt.ItemIsEditable)
            it1.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            it2.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            tbl.setItem(r, 0, it0)
            tbl.setItem(r, 1, it1)
            tbl.setCellWidget(r, 2, pill)
            tbl.setItem(r, 2, QtWidgets.QTableWidgetItem(''))

    def _clear(self) -> None:
        try:
            if pg:
                self.plot_roll.clear()
                self.plot_month.clear()
        except Exception:
            pass
        self.tbl_opp.setRowCount(0)
        self.tbl_map.setRowCount(0)
