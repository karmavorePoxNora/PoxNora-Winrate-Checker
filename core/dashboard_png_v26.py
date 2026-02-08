# core/dashboard_png_v26.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# =========================================================
# THEME — Eclipse Neon
# =========================================================
BG = "#0b1220"
CARD = "#111a2b"
GRID = "#26324a"
TEXT = "#dbe6ff"
MUTED = "#9fb4d8"

PURPLE = "#c86bff"
BLUE = "#55b7ff"
GREEN = "#45f0b5"


def _to_path(p: Optional[Union[str, Path]]) -> Optional[Path]:
    if p is None:
        return None
    return p if isinstance(p, Path) else Path(str(p))


def _safe_pd():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None



def _parse_dates_fast(pd, series):
    """Fast, consistent Date parsing for PNG export (view-layer only)."""
    try:
        dt = pd.to_datetime(series, format="%b %d, %Y %I:%M:%S %p", errors="coerce")
        if dt is not None and dt.notna().sum() < max(3, int(0.05 * len(dt))):
            dt = pd.to_datetime(series, errors="coerce")
        return dt
    except Exception:
        try:
            return pd.to_datetime(series, errors="coerce")
        except Exception:
            return None


def _ranked_truthy(v: str) -> bool:
    s = (str(v or "").strip().lower())
    return s in {"1", "true", "yes", "y", "ranked", "on"}


def _glow_line(ax, x, y, color, lw=2.6):
    for w, a in [(10, 0.05), (7, 0.08), (5, 0.14)]:
        ax.plot(x, y, color=color, linewidth=w, alpha=a, solid_capstyle="round")
    ln, = ax.plot(x, y, color=color, linewidth=lw, solid_capstyle="round")
    ln.set_path_effects([pe.Stroke(linewidth=lw + 2.0, foreground=color, alpha=0.35), pe.Normal()])


def _style_axes(ax):
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(color=GRID, alpha=0.55, linewidth=0.7)
    ax.tick_params(colors=MUTED, labelsize=9)


def _title(ax, title: str, subtitle: str = ""):
    ax.text(0.02, 1.05, title, transform=ax.transAxes, color=TEXT, fontsize=13, fontweight="bold", va="bottom")
    if subtitle:
        ax.text(0.02, 1.01, subtitle, transform=ax.transAxes, color="#8fa4c8", fontsize=9, va="top")


def _pill(ax, x, y, text: str, kind: str = "blue"):
    if kind == "purple":
        bg = (200/255, 107/255, 255/255, 0.16)
        bd = (200/255, 107/255, 255/255, 0.35)
    elif kind == "green":
        bg = (69/255, 240/255, 181/255, 0.14)
        bd = (69/255, 240/255, 181/255, 0.35)
    else:
        bg = (85/255, 183/255, 255/255, 0.14)
        bd = (85/255, 183/255, 255/255, 0.35)

    ax.text(
        x, y, text,
        transform=ax.transAxes,
        color=TEXT,
        fontsize=10,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.75", facecolor=bg, edgecolor=bd, linewidth=1.0)
    )


def build_dashboard_png(
    csv_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    df: Optional[Any] = None,
    player_name: Optional[str] = None,
    ranked_only: bool = False,
    log=None,
):
    """
    Signature matches the app's flexible builder calls.
    - Accepts df OR csv_path.
    - Writes PNG to output_path (required by app).
    """
    pd = _safe_pd()
    if pd is None:
        raise RuntimeError("pandas is required to build dashboard PNG.")

    out = _to_path(output_path) or Path("dashboard.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    if df is None:
        cpath = _to_path(csv_path)
        if cpath is None:
            raise RuntimeError("build_dashboard_png requires df or csv_path.")
        df = pd.read_csv(cpath)

    if df is None or len(df) == 0:
        raise RuntimeError("Empty dataset.")

    # canonical columns (from dataset.py)
    cols = {str(c).strip().lower(): c for c in df.columns}
    c_win = cols.get("win")
    c_loss = cols.get("loss")
    c_ranked = cols.get("ranked")
    c_date = cols.get("date")
    c_map = cols.get("map")
    c_opp = cols.get("opponent")

    if not (c_win and c_loss):
        raise RuntimeError("Dataset missing Win/Loss columns.")

    dff = df.copy()

    # ranked filter
    if ranked_only and c_ranked is not None:
        try:
            dff = dff[dff[c_ranked].apply(_ranked_truthy)]
        except Exception:
            pass

    # determine player (strict: provided first, else infer from Win/Loss mode)
    player = (player_name or "").strip()
    if not player:
        try:
            s = pd.concat([dff[c_win].astype(str), dff[c_loss].astype(str)], ignore_index=True)
            s = s[s.notna() & (s.astype(str).str.strip() != "")]
            player = str(s.mode().iloc[0]) if len(s) else ""
        except Exception:
            player = ""
    if not player:
        raise RuntimeError("Player name missing and could not be inferred.")

    p = player.strip().lower()
    w = dff[c_win].astype(str).str.strip().str.lower()
    l = dff[c_loss].astype(str).str.strip().str.lower()

    pm = (w == p) | (l == p)
    dffp = dff[pm].copy()
    if len(dffp) == 0:
        raise RuntimeError("No games for this player in dataset.")

    is_win = (dffp[c_win].astype(str).str.strip().str.lower() == p).astype(int)

    # opponent inference if not present
    if c_opp is None:
        opp = np.where(w[pm].values == p, l[pm].values, w[pm].values).astype(str)
    else:
        opp = dffp[c_opp].astype(str).values

    # dates
    dt = None
    if c_date is not None:
        try:
            dt = _parse_dates_fast(pd, dffp[c_date])
        except Exception:
            dt = None
    if dt is not None:
        dffp["_dt"] = dt
        dffp = dffp.sort_values("_dt")
        is_win = is_win.loc[dffp.index]

    # ============================
    # Metrics
    # ============================
    games = int(len(dffp))
    wins = int(is_win.sum())
    losses = games - wins
    wr = (wins / games * 100.0) if games else 0.0

    # rolling 50
    win_arr = is_win.to_numpy(dtype=float)
    if len(win_arr) >= 5:
        # fast rolling mean via convolution
        k = min(50, len(win_arr))
        win50 = np.convolve(win_arr, np.ones(k)/k, mode="valid") * 100.0
        x_roll = np.arange(k, k + len(win50))
        y_roll = win50
    else:
        x_roll = np.arange(len(win_arr))
        y_roll = win_arr * 100.0

    # downsample to keep it readable on huge datasets
    if len(x_roll) > 1400:
        stride = int(np.ceil(len(x_roll) / 1200))
        x_roll = x_roll[::stride]
        y_roll = y_roll[::stride]

    # monthly
    m_labels, m_wr, m_games = [], np.array([]), np.array([])
    if dt is not None and dt.notna().any():
        try:
            m = pd.to_datetime(dffp["_dt"], errors="coerce").dt.to_period("M").astype(str)
            dffp["_m"] = m
            g = dffp.assign(_w=is_win.values).groupby("_m").agg(games=("_w", "size"), wins=("_w", "sum")).reset_index()
            g["wr"] = (g["wins"] / g["games"]) * 100.0
            m_labels = g["_m"].tolist()
            m_wr = g["wr"].to_numpy(dtype=float)
            m_games = g["games"].to_numpy(dtype=float)
        except Exception:
            pass

    # yearly (fixed set for report header)
    year_wrs = {}
    if dt is not None and dt.notna().any():
        try:
            yy = pd.to_datetime(dffp["_dt"], errors="coerce").dt.year
            for _y in (2025, 2024, 2023):
                masky = (yy == _y)
                yw = is_win.values[masky.values]
                year_wrs[_y] = float((yw.sum() / float(len(yw))) * 100.0) if len(yw) else None
        except Exception:
            year_wrs = {2025: None, 2024: None, 2023: None}
    else:
        year_wrs = {2025: None, 2024: None, 2023: None}

    # top opponents
    try:
        t = pd.DataFrame({"opp": opp.astype(str), "w": is_win.values})
        go = t.groupby("opp").agg(games=("w", "size"), wins=("w", "sum"))
        go["wr"] = (go["wins"] / go["games"]) * 100.0
        go = go.sort_values("games", ascending=False).head(10)
    except Exception:
        go = None

    # top maps
    gm = None
    if c_map is not None:
        try:
            t = pd.DataFrame({"map": dffp[c_map].astype(str).values, "w": is_win.values})
            gm = t.groupby("map").agg(games=("w", "size"), wins=("w", "sum"))
            gm["wr"] = (gm["wins"] / gm["games"]) * 100.0
            gm = gm.sort_values("games", ascending=False).head(10)
        except Exception:
            gm = None

    # ============================
    # FIGURE — 2x2 grid like the mock
    # ============================
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.subplots_adjust(bottom=0.10)
    gs = fig.add_gridspec(2, 2, wspace=0.08, hspace=0.22)

    ax_roll = fig.add_subplot(gs[0, 0])
    ax_month = fig.add_subplot(gs[0, 1])
    ax_opp = fig.add_subplot(gs[1, 0])
    ax_map = fig.add_subplot(gs[1, 1])

    for ax in (ax_roll, ax_month, ax_opp, ax_map):
        _style_axes(ax)

    # header
    fig.text(0.03, 0.965, "PoxNora Tracker — Analytics Board", color=TEXT, fontsize=16, fontweight="bold")
    fig.text(0.03, 0.935, "All games" if not ranked_only else "Ranked only", color="#8fa4c8", fontsize=10)

    # top pills
    fig.text(0.56, 0.957, f"Games\n{games}", color=TEXT, fontsize=9, ha="center")
    fig.text(0.64, 0.957, f"W-L\n{wins}-{losses}", color=TEXT, fontsize=9, ha="center")
    fig.text(0.72, 0.957, f"Winrate\n{wr:.1f}%", color=TEXT, fontsize=9, ha="center")
    # Yearly winrate badges (fixed years for consistency)
    x_years = [0.78, 0.86, 0.94]
    years = [2025, 2024, 2023]
    for xi, yr in zip(x_years, years):
        val = year_wrs.get(yr)
        fig.text(xi, 0.957, f"{yr}\n{val:.1f}%" if val is not None else f"{yr}\n—", color=TEXT, fontsize=9, ha="center")

    # Rolling
    _title(ax_roll, "Rolling Winrate", "Rolling winrate (last 50 games)")
    _glow_line(ax_roll, x_roll, y_roll, PURPLE)
    ax_roll.fill_between(x_roll, y_roll, 0, color=PURPLE, alpha=0.10)
    ax_roll.set_ylim(0, 100)
    # subtle guide lines (mock-style)
    for yv in (50, 75, 100):
        ax_roll.axhline(yv, color=(0.9,0.95,1.0,0.12), linewidth=1.0, linestyle='--')
    ax_roll.set_xlabel("Games", color=MUTED)
    ax_roll.set_ylabel("Winrate %", color=MUTED)

    # Monthly
    _title(ax_month, "Monthly Performance", "Monthly winrate + bars = games")
    if len(m_wr) > 0:
        x = np.arange(len(m_wr))
        if len(m_games) and float(np.max(m_games)) > 0:
            gn = (m_games / float(np.max(m_games))) * 100.0
            ax_month.bar(x, gn, color=BLUE, alpha=0.14, width=0.65)
        _glow_line(ax_month, x, m_wr, BLUE)
        ax_month.fill_between(x, m_wr, 0, color=BLUE, alpha=0.10)
        ax_month.set_ylim(0, 100)
        # subtle guide lines (mock-style)
        for yv in (50, 75, 100):
            ax_month.axhline(yv, color=(0.9,0.95,1.0,0.12), linewidth=1.0, linestyle='--')
        # labels
        lbls = []
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        for s in m_labels:
            try:
                yy, mm = str(s).split("-")
                lbls.append(f"{months[int(mm)-1]} '{yy[2:4]}")
            except Exception:
                lbls.append(str(s))
        # Adaptive tick density (keeps labels clean for large histories)
        try:
            nlab = int(len(lbls))
        except Exception:
            nlab = len(lbls)

        max_ticks = 9
        step = int(math.ceil(nlab / float(max_ticks))) if nlab > max_ticks else 1
        xt = x[::step]
        xl = lbls[::step]

        # Always include the last month label (most recent)
        try:
            if len(xt) and int(xt[-1]) != int(x[-1]):
                xt = np.append(xt, x[-1])
                xl = xl + [lbls[-1]]
        except Exception:
            pass

        ax_month.set_xticks(xt)
        ax_month.set_xticklabels(xl, rotation=40, ha="right", fontsize=7, color=MUTED)
        ax_month.tick_params(axis="x", pad=6)
    else:
        ax_month.text(0.5, 0.5, "Monthly view requires dates", ha="center", va="center", color="#6e83a9", transform=ax_month.transAxes)

    # Top Opponents
    _title(ax_opp, "Top Opponents", "Most common matchups")
    if go is not None and len(go) > 0:
        names = list(go.index)[::-1]
        games_o = go["games"].to_numpy(dtype=float)[::-1]
        wr_o = go["wr"].to_numpy(dtype=float)[::-1]
        y = np.arange(len(names))
        ax_opp.barh(y, games_o, color=(0.35,0.72,1.0,0.35))
        for i, (nm, gmz, wrr) in enumerate(zip(names, games_o, wr_o)):
            ax_opp.text(0.01, i, str(nm), transform=ax_opp.get_yaxis_transform(), color=TEXT, fontsize=9, va="center")
            ax_opp.text(0.70, i, f"{int(gmz)}", transform=ax_opp.get_yaxis_transform(), color=MUTED, fontsize=9, va="center")
            # pill
            ax_opp.text(0.92, i, f"{wrr:.1f}%", transform=ax_opp.get_yaxis_transform(), color=TEXT, fontsize=9, va="center",
                        bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.6",
                                  facecolor=(0.30,0.62,1.0,0.18),
                                  edgecolor=(0.55,0.75,1.0,0.35),
                                  linewidth=1.0))
        ax_opp.set_yticks([])
        ax_opp.set_xlim(0, max(1.0, float(np.max(games_o)) * 1.15))
        ax_opp.set_xlabel("Games", color=MUTED)
    else:
        ax_opp.text(0.5, 0.5, "No opponent data", ha="center", va="center", color="#6e83a9", transform=ax_opp.transAxes)

    # Top Maps
    _title(ax_map, "Top Maps", "Highest-volume maps")
    if gm is not None and len(gm) > 0:
        names = list(gm.index)[::-1]
        games_m = gm["games"].to_numpy(dtype=float)[::-1]
        wr_m = gm["wr"].to_numpy(dtype=float)[::-1]
        y = np.arange(len(names))
        ax_map.barh(y, games_m, color=(0.27,0.94,0.71,0.32))
        for i, (nm, gmz, wrr) in enumerate(zip(names, games_m, wr_m)):
            ax_map.text(0.01, i, str(nm), transform=ax_map.get_yaxis_transform(), color=TEXT, fontsize=9, va="center")
            ax_map.text(0.70, i, f"{int(gmz)}", transform=ax_map.get_yaxis_transform(), color=MUTED, fontsize=9, va="center")
            ax_map.text(0.92, i, f"{wrr:.1f}%", transform=ax_map.get_yaxis_transform(), color=TEXT, fontsize=9, va="center",
                        bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.6",
                                  facecolor=(0.27,0.94,0.71,0.14),
                                  edgecolor=(0.27,0.94,0.71,0.35),
                                  linewidth=1.0))
        ax_map.set_yticks([])
        ax_map.set_xlim(0, max(1.0, float(np.max(games_m)) * 1.15))
        ax_map.set_xlabel("Games", color=MUTED)
    else:
        ax_map.text(0.5, 0.5, "No map data", ha="center", va="center", color="#6e83a9", transform=ax_map.transAxes)

    # save
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

    if log:
        try:
            log(f"✅ Built PNG: {out}")
        except Exception:
            pass
    return out
