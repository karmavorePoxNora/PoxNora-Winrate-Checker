import os
import sys
import subprocess
import time
import re
import json
import webbrowser
from pathlib import Path
import threading
import ctypes
import textwrap
import urllib.request
import urllib.error

import pandas as pd
import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright

import tkinter as tk
from tkinter import ttk, messagebox


# ============================================================
# Version + Updates
# ============================================================
APP_NAME = "PoxNoraTracker"
APP_TITLE = "PoxNora Tracker"
APP_VERSION = "1.2.0"  # bump this when you ship a new build

# Taskbar identity (helps Windows show your icon consistently)
APP_ID = "BrandonKing.PoxNoraTracker"

# Optional GitHub repo for update checks:
# Set to ("YourGitHubUser", "YourRepoName") once you publish releases.
GITHUB_REPO = ("", "")  # e.g. ("BrandonKing", "PoxNoraTool")
GITHUB_RELEASES_LATEST_API = "https://api.github.com/repos/{owner}/{repo}/releases/latest"


# ============================================================
# App identity + installer-safe folders
# ============================================================
LOCAL_APPDATA = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
APP_DIR = LOCAL_APPDATA / APP_NAME

OUTPUT_DIR = APP_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = APP_DIR / "config.json"

# Playwright storage state (remember login)
STORAGE_STATE_FILE = APP_DIR / "storage_state.json"

# Stable browser storage (critical for EXE + MSI)
BROWSERS_DIR = APP_DIR / "pw-browsers"
BROWSERS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(BROWSERS_DIR)

# ============================================================
# URLs
# ============================================================
LOGIN_URL = "https://www.poxnora.com/account/login.do"
MATCH_URL = "https://www.poxnora.com/account/matchhistory.do"

# ============================================================
# Outputs
# ============================================================
OUT_CSV = OUTPUT_DIR / "poxnora_matches_clean.csv"
OUT_DASHBOARD_PNG = OUTPUT_DIR / "poxnora_dashboard_dark.png"
OUT_REPORT_HTML = OUTPUT_DIR / "report.html"
OUT_H2H_CSV = OUTPUT_DIR / "poxnora_h2h.csv"
OUT_MAPS_CSV = OUTPUT_DIR / "poxnora_maps.csv"
OUT_SUMMARY_CSV = OUTPUT_DIR / "poxnora_summary.csv"


# ============================================================
# Packaged-resource helper (PyInstaller)
# ============================================================
def resource_path(rel_path: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel_path)


# ============================================================
# Config
# ============================================================
def load_config():
    default = {
        "player": "",
        "only_ranked_1v1": True,
        "max_pages": 2000,
        "headless": False,
        "remember_login": True,
    }
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            default.update(data)
        except Exception:
            pass
    return default


def save_config(cfg: dict):
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


# ============================================================
# Dark matplotlib defaults
# ============================================================
def apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor": "#0b0f14",
        "axes.facecolor": "#0b0f14",
        "savefig.facecolor": "#0b0f14",
        "text.color": "#e6edf3",
        "axes.labelcolor": "#e6edf3",
        "xtick.color": "#e6edf3",
        "ytick.color": "#e6edf3",
        "axes.edgecolor": "#2b3440",
        "grid.color": "#2b3440",
        "font.size": 11,
    })


# ============================================================
# Dark ttk styling (sleeker buttons + blue progressbar)
# ============================================================
def apply_ttk_dark(root: tk.Tk):
    root.configure(bg="#0b0f14")
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    BG = "#0b0f14"
    CARD = "#0f1622"
    BORDER = "#223044"
    TEXT = "#e6edf3"
    MUTED = "#9fb0c3"

    BTN = "#1b2a41"
    BTN_HOVER = "#243a5a"
    BTN_PRESS = "#162a46"
    BLUE = "#2f81f7"

    style.configure(".", background=BG, foreground=TEXT)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT)

    style.configure("TEntry",
                    fieldbackground=CARD,
                    foreground=TEXT,
                    bordercolor=BORDER,
                    lightcolor=BORDER,
                    darkcolor=BORDER,
                    padding=6)

    style.configure("TButton",
                    background=BTN,
                    foreground=TEXT,
                    bordercolor=BORDER,
                    focusthickness=0,
                    padding=(14, 10),
                    relief="flat")
    style.map("TButton",
              background=[("active", BTN_HOVER), ("pressed", BTN_PRESS), ("disabled", "#0d131d")],
              foreground=[("disabled", MUTED)])

    style.configure("TCheckbutton", background=BG, foreground=TEXT)
    style.map("TCheckbutton", foreground=[("disabled", MUTED)])

    style.configure("Blue.Horizontal.TProgressbar",
                    troughcolor=CARD,
                    background=BLUE,
                    bordercolor=BORDER,
                    lightcolor=BLUE,
                    darkcolor=BLUE)


# ============================================================
# Helpers
# ============================================================
def open_path(path: Path):
    try:
        os.startfile(str(path.resolve()))  # type: ignore[attr-defined]
    except Exception:
        try:
            webbrowser.open(path.resolve().as_uri())
        except Exception:
            pass


def open_output_folder():
    open_path(OUTPUT_DIR)


def _wrap_label(s: str, width: int = 18, max_lines: int = 2) -> str:
    s = str(s).strip()
    lines = textwrap.wrap(s, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if len(lines[-1]) >= 3:
            lines[-1] = lines[-1][:-3] + "..."
        else:
            lines[-1] = lines[-1] + "..."
    return "\n".join(lines) if lines else s


def _parse_version(v: str) -> tuple[int, int, int]:
    # accepts "v1.2.3" or "1.2.3"
    v = v.strip()
    if v.lower().startswith("v"):
        v = v[1:]
    parts = v.split(".")
    nums = []
    for p in parts[:3]:
        try:
            nums.append(int(re.sub(r"[^\d].*$", "", p)))
        except Exception:
            nums.append(0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])


def check_for_updates() -> tuple[bool, str]:
    """
    Returns (has_update, message). If update exists, message includes latest tag.
    Requires GITHUB_REPO to be set.
    """
    owner, repo = GITHUB_REPO
    if not owner or not repo:
        return (False, "Update checks are disabled (no GitHub repo configured).")

    url = GITHUB_RELEASES_LATEST_API.format(owner=owner, repo=repo)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"{APP_NAME}/{APP_VERSION}",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        return (False, f"Update check failed (HTTP {e.code}).")
    except Exception as e:
        return (False, f"Update check failed: {e}")

    tag = str(data.get("tag_name", "")).strip()
    html_url = str(data.get("html_url", "")).strip()
    if not tag:
        return (False, "No release tag found on GitHub.")

    cur = _parse_version(APP_VERSION)
    latest = _parse_version(tag)

    if latest > cur:
        msg = f"Update available: {tag} (you have {APP_VERSION})."
        if html_url:
            msg += f"\n\nOpen release page?\n{html_url}"
        return (True, msg)

    return (False, f"You're up to date ({APP_VERSION}).")


# ============================================================
# Playwright Chromium install (first-run)
# ============================================================
def ensure_playwright_chromium(log):
    ok_marker = BROWSERS_DIR / ".chromium_ok"
    if ok_marker.exists():
        return
    log("Installing Playwright Chromium (first run only)...")
    subprocess.check_call(["python", "-m", "playwright", "install", "chromium"])
    ok_marker.touch()
    log("✅ Chromium installed.")


# ============================================================
# Scraping DOM helpers
# ============================================================
def stable_wait(page):
    for _ in range(6):
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
            break
        except Exception:
            time.sleep(0.6)
    time.sleep(0.9)


def scrape_rows(page):
    rows = page.locator("div.rowBg1, div.rowBg2")
    out = []
    rc = rows.count()
    for i in range(rc):
        row = rows.nth(i)
        cells = row.locator("[class*='rowsCell']")
        cc = cells.count()
        row_data = []
        for j in range(cc):
            txt = cells.nth(j).inner_text().strip()
            if txt:
                row_data.append(txt)
        if row_data:
            out.append(row_data)
    return out


def _absolute_url_from_href(current_url: str, href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    m = re.match(r"^(https?://[^/]+)", current_url)
    base = m.group(1) if m else ""
    if href.startswith("/"):
        return base + href
    if current_url.endswith("/"):
        return current_url + href
    return current_url.rsplit("/", 1)[0] + "/" + href


def _detect_total_pages(page) -> int | None:
    try:
        hrefs = page.eval_on_selector_all(
            "a[href*='matchhistory.do']",
            "els => els.map(e => e.getAttribute('href')).filter(Boolean)"
        )
    except Exception:
        hrefs = []

    p_vals = []
    for h in hrefs:
        m = re.search(r"[?&]p=(\d+)", h)
        if m:
            p_vals.append(int(m.group(1)))
    if p_vals:
        return max(p_vals)

    i_vals = []
    for h in hrefs:
        m = re.search(r"[?&]i=(\d+)", h)
        if m:
            i_vals.append(int(m.group(1)))
    if i_vals:
        try:
            rpp = max(1, len(scrape_rows(page)))
        except Exception:
            rpp = 20
        max_offset = max(i_vals)
        return int(max_offset // rpp) + 1

    return None


def scrape_to_clean_csv_by_clicking_next(page, max_pages: int, log, progress_cb=None):
    stable_wait(page)

    try:
        page.locator("h1.content_h1:has-text('Match History')").wait_for(timeout=10000)
    except Exception:
        log("Not detecting Match History header. Attempting to open match history page...")
        page.goto(MATCH_URL, wait_until="networkidle")
        stable_wait(page)
        page.locator("h1.content_h1:has-text('Match History')").wait_for(timeout=10000)

    log("✅ Match History detected. Starting scrape...")

    raw_rows = []
    seen_page_sigs = set()

    def signature(rows):
        if not rows:
            return None
        sample = rows[:3]
        flat = []
        for r in sample:
            flat.append("|".join(r[:8]))
        return "||".join(flat)

    def wait_for_page_change(prev_sig: str | None):
        start = time.time()
        while time.time() - start < 12:
            try:
                rows_now = scrape_rows(page)
                sig_now = signature(rows_now)
                if sig_now and prev_sig and sig_now != prev_sig:
                    return True
            except Exception:
                pass
            time.sleep(0.4)
        return False

    def click_next_if_found() -> bool:
        try:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(0.4)
        except Exception:
            pass

        role_patterns = [
            re.compile(r"^\s*next\s*$", re.I),
            re.compile(r"^\s*next\s*»\s*$", re.I),
            re.compile(r"^\s*»\s*$"),
            re.compile(r"^\s*>\s*$"),
        ]
        for pat in role_patterns:
            loc = page.get_by_role("link", name=pat)
            if loc.count() > 0:
                try:
                    if loc.first.is_visible() and loc.first.is_enabled():
                        loc.first.click()
                        return True
                except Exception:
                    pass

        css_candidates = [
            "div.paging a:has-text('Next')",
            "div.paging a:has-text('»')",
            "div.paging a[rel='next']",
            "ul.pagination a:has-text('Next')",
            "ul.pagination a:has-text('»')",
            "a[rel='next']",
            "a:has-text('Next')",
            "a:has-text('»')",
        ]
        for sel in css_candidates:
            loc = page.locator(sel)
            if loc.count() > 0:
                try:
                    if loc.first.is_visible() and loc.first.is_enabled():
                        loc.first.click()
                        return True
                except Exception:
                    pass

        try:
            hrefs = page.eval_on_selector_all(
                "a[href*='matchhistory.do']",
                "els => els.map(e => e.getAttribute('href')).filter(Boolean)"
            )
        except Exception:
            hrefs = []

        def score(h: str) -> int:
            m = re.search(r"[?&]p=(\d+)", h)
            if m:
                return int(m.group(1))
            m = re.search(r"[?&]i=(\d+)", h)
            if m:
                return int(m.group(1))
            return -1

        scored = [(score(h), h) for h in hrefs]
        scored = [(s, h) for s, h in scored if s >= 0]
        scored.sort(key=lambda x: x[0])

        for _, h in scored[-6:]:
            try:
                abs_url = _absolute_url_from_href(page.url, h)
                page.goto(abs_url, wait_until="networkidle")
                return True
            except Exception:
                continue

        return False

    total_pages = _detect_total_pages(page)
    effective_max = max_pages
    if total_pages is not None:
        effective_max = min(max_pages, total_pages)
        log(f"Detected total pages: ~{total_pages}. Will scrape up to: {effective_max}.")
        if progress_cb:
            progress_cb(0, effective_max, determinate=True)
    else:
        log("Total pages not detectable — progress will be indeterminate.")
        if progress_cb:
            progress_cb(0, max_pages, determinate=False)

    page_num = 0
    prev_sig = None

    while True:
        if page_num >= effective_max:
            log(f"⚠ Reached max_pages={effective_max}. Stopping.")
            break

        rows = scrape_rows(page)
        if not rows:
            log("No rows detected — stopping.")
            break

        sig = signature(rows)
        if sig and sig in seen_page_sigs:
            log("Detected repeated page signature — stopping.")
            break
        if sig:
            seen_page_sigs.add(sig)

        raw_rows.extend(rows)
        log(f"Scraped page {page_num + 1}: +{len(rows)} rows")

        if progress_cb:
            progress_cb(page_num + 1, effective_max, determinate=(total_pages is not None))

        prev_sig = sig

        if not click_next_if_found():
            log("No Next link found — stopping.")
            break

        stable_wait(page)

        if not wait_for_page_change(prev_sig):
            log("Next clicked but page content did not change — stopping to avoid loop.")
            break

        page_num += 1

    if not raw_rows:
        raise RuntimeError("Scrape returned 0 rows.")

    max_cols = max(len(r) for r in raw_rows)
    df_raw = pd.DataFrame(raw_rows, columns=[f"col_{i+1}" for i in range(max_cols)])

    df = df_raw.iloc[:, ::2].copy()

    if df.shape[1] < 8:
        raise RuntimeError(f"After mirror-cleaning, fewer than 8 columns were found ({df.shape[1]}).")

    df = df.iloc[:, :8].copy()
    df.columns = ["Win", "Loss", "Type", "Ranked", "Map", "Rating", "Date", "Duration"]

    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    df = df.drop_duplicates()
    df.to_csv(OUT_CSV, index=False)
    log(f"✅ Saved clean CSV: {OUT_CSV} (rows: {len(df)})")


# ============================================================
# Analytics
# ============================================================
def compute_h2h(df: pd.DataFrame, player: str) -> pd.DataFrame:
    p = player.lower()

    def opponent(row):
        w = str(row["Win"])
        l = str(row["Loss"])
        if w.lower() == p:
            return l
        if l.lower() == p:
            return w
        return None

    d = df.copy()
    d["Opponent"] = d.apply(opponent, axis=1)
    d = d[d["Opponent"].notna()].copy()
    d["is_win"] = (d["Win"].astype(str).str.lower() == p).astype(int)

    h2h = d.groupby("Opponent")["is_win"].agg(games="count", wins="sum").reset_index()
    h2h["losses"] = h2h["games"] - h2h["wins"]
    h2h["wr"] = (h2h["wins"] / h2h["games"]) * 100
    return h2h.sort_values(["games", "wr"], ascending=[False, False])


def compute_maps(df: pd.DataFrame, player: str) -> pd.DataFrame:
    p = player.lower()
    d = df.copy()
    d["is_win"] = (d["Win"].astype(str).str.lower() == p).astype(int)

    maps = d.groupby("Map")["is_win"].agg(games="count", wins="sum").reset_index()
    maps["losses"] = maps["games"] - maps["wins"]
    maps["wr"] = (maps["wins"] / maps["games"]) * 100
    return maps.sort_values(["games", "wr"], ascending=[False, False])


def compute_streaks(df: pd.DataFrame, player: str) -> dict:
    """
    Expects df already filtered to matches involving player.
    Uses chronological order.
    """
    p = player.lower()
    d = df.copy()

    # Parse date best-effort
    d["Date_dt"] = pd.to_datetime(d["Date"], errors="coerce", utc=False)
    # If parsing fails for many rows, keep original order as fallback
    if d["Date_dt"].notna().sum() >= max(5, int(len(d) * 0.3)):
        d = d.sort_values("Date_dt")
    # else: keep current order

    d["is_win"] = (d["Win"].astype(str).str.lower() == p).astype(int)

    # current streak
    current = 0
    current_type = None
    for v in d["is_win"].iloc[::-1].tolist():
        if current_type is None:
            current_type = v
            current = 1
        else:
            if v == current_type:
                current += 1
            else:
                break
    current_label = "Win" if current_type == 1 else "Loss"
    current_streak = f"{current_label} {current}"

    # best win / loss streak
    best_win = 0
    best_loss = 0
    run = 0
    run_type = None
    for v in d["is_win"].tolist():
        if run_type is None or v != run_type:
            run_type = v
            run = 1
        else:
            run += 1
        if run_type == 1:
            best_win = max(best_win, run)
        else:
            best_loss = max(best_loss, run)

    # last 30
    last_n = d.tail(30)
    last30_games = len(last_n)
    last30_wins = int(last_n["is_win"].sum()) if last30_games else 0
    last30_wr = (last30_wins / last30_games * 100.0) if last30_games else 0.0

    return {
        "current_streak": current_streak,
        "best_win_streak": best_win,
        "best_loss_streak": best_loss,
        "last30_games": last30_games,
        "last30_wr": float(last30_wr),
    }


def write_html_report(player: str, summary: dict, streaks: dict, out_path: Path):
    # Embed images via relative names in output folder
    dash_name = OUT_DASHBOARD_PNG.name

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>PoxNora Report — {player}</title>
<style>
  body {{
    margin: 0; padding: 24px;
    background: #0b0f14; color: #e6edf3;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  .card {{
    background: #111826; border: 1px solid #223044;
    border-radius: 16px; padding: 16px 18px; margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,.25);
  }}
  h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
  h2 {{ margin: 0 0 10px 0; font-size: 18px; }}
  .muted {{ color: #9fb0c3; }}
  .kpi {{
    display:flex; gap: 12px; flex-wrap: wrap; margin-top: 10px;
  }}
  .pill {{
    background: #0f1622; border: 1px solid #223044;
    border-radius: 999px; padding: 8px 12px;
  }}
  img {{
    width: 100%;
    border-radius: 16px;
    border: 1px solid #223044;
    background: #0b0f14;
  }}
  .small {{ font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <h1>PoxNora Report — {player}</h1>
    <div class="kpi">
      <div class="pill">Games: <b>{summary["games"]}</b></div>
      <div class="pill">Wins: <b>{summary["wins"]}</b></div>
      <div class="pill">Losses: <b>{summary["losses"]}</b></div>
      <div class="pill">Win rate: <b>{summary["win_rate"]:.2f}%</b></div>
      <div class="pill">Current streak: <b>{streaks["current_streak"]}</b></div>
      <div class="pill">Best win streak: <b>{streaks["best_win_streak"]}</b></div>
      <div class="pill">Best loss streak: <b>{streaks["best_loss_streak"]}</b></div>
      <div class="pill">Last 30 WR: <b>{streaks["last30_wr"]:.2f}%</b></div>
    </div>
    <p class="muted small">Output folder: {OUTPUT_DIR}</p>
  </div>

  <div class="card">
    <h2>Dashboard</h2>
    <img src="{dash_name}" alt="Dashboard" />
  </div>
</div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def build_dashboard(player: str, only_ranked_1v1: bool, log):
    if not OUT_CSV.exists():
        raise RuntimeError(f"Missing CSV: {OUT_CSV}. Run scrape first.")

    df = pd.read_csv(OUT_CSV)
    df["Type_norm"] = df["Type"].astype(str).str.strip().str.lower()
    df["Ranked_norm"] = df["Ranked"].astype(str).str.strip().str.lower()

    if only_ranked_1v1:
        df = df[(df["Type_norm"] == "1v1") & (df["Ranked_norm"] == "yes")].copy()

    p = player.lower()
    df = df[(df["Win"].astype(str).str.lower() == p) | (df["Loss"].astype(str).str.lower() == p)].copy()

    wins = int((df["Win"].astype(str).str.lower() == p).sum())
    losses = int((df["Loss"].astype(str).str.lower() == p).sum())
    total = wins + losses

    if total == 0:
        raise RuntimeError("No matches found for that player in this CSV (after filters).")

    wr = (wins / total) * 100.0

    h2h = compute_h2h(df, player)
    maps = compute_maps(df, player)
    streaks = compute_streaks(df, player)

    # Save summaries
    pd.DataFrame([{
        "player": player,
        "games": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(float(wr), 4),
        "current_streak": streaks["current_streak"],
        "best_win_streak": streaks["best_win_streak"],
        "best_loss_streak": streaks["best_loss_streak"],
        "last30_wr": round(float(streaks["last30_wr"]), 4),
    }]).to_csv(OUT_SUMMARY_CSV, index=False)

    h2h.to_csv(OUT_H2H_CSV, index=False)
    maps.to_csv(OUT_MAPS_CSV, index=False)

    # Build dashboard image: Opponents + Maps + Rating over time
    apply_dark_style()

    top_opp = h2h.head(12).copy()
    top_maps = maps.head(12).copy()

    top_opp["Opponent_wrapped"] = top_opp["Opponent"].apply(lambda s: _wrap_label(s, width=16, max_lines=2))
    top_maps["Map_wrapped"] = top_maps["Map"].apply(lambda s: _wrap_label(s, width=20, max_lines=2))

    # rating over time
    d = df.copy()
    d["Date_dt"] = pd.to_datetime(d["Date"], errors="coerce", utc=False)
    d["Rating_num"] = pd.to_numeric(d["Rating"].astype(str).str.replace(",", ""), errors="coerce")

    # If dates parse well, sort; otherwise keep order
    if d["Date_dt"].notna().sum() >= max(10, int(len(d) * 0.3)):
        d = d.sort_values("Date_dt")
        x = d["Date_dt"]
    else:
        x = list(range(1, len(d) + 1))

    y = d["Rating_num"]
    y_roll = y.rolling(window=20, min_periods=5).mean()

    # Layout with gridspec
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 0.9])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # Header text
    fig.suptitle(f"{APP_TITLE} — {player}", fontsize=18, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.94, f"Games: {total}    Wins: {wins}    Losses: {losses}    Win Rate: {wr:.2f}%    "
                         f"Current: {streaks['current_streak']}    Last 30 WR: {streaks['last30_wr']:.2f}%",
             fontsize=11, ha="left")

    # Opponents
    ax1.set_title("Top Opponents (by games)")
    ax1.grid(True, axis="x", alpha=0.3)
    y1 = top_opp["Opponent_wrapped"].iloc[::-1]
    x1 = top_opp["games"].iloc[::-1]
    ax1.barh(y1, x1)
    ax1.set_xlabel("Games")
    ax1.tick_params(axis="y", labelsize=9)
    ax1.set_xlim(0, max(1, int(x1.max() * 1.35)))
    for i, (_, r) in enumerate(top_opp.iloc[::-1].iterrows()):
        ax1.text(r["games"] * 1.02, i, f"{int(r['wins'])}-{int(r['losses'])} ({r['wr']:.1f}%)",
                 va="center", fontsize=9)

    # Maps
    ax2.set_title("Top Maps (by games) — Win Rate")
    ax2.grid(True, axis="x", alpha=0.3)
    y2 = top_maps["Map_wrapped"].iloc[::-1]
    x2 = top_maps["wr"].iloc[::-1]
    ax2.barh(y2, x2)
    ax2.set_xlabel("Win Rate %")
    ax2.set_xlim(0, 102)
    ax2.tick_params(axis="y", labelsize=9)
    for i, (_, r) in enumerate(top_maps.iloc[::-1].iterrows()):
        ax2.text(min(101.5, r["wr"] + 1.2), i, f"{int(r['wins'])}-{int(r['losses'])} / {int(r['games'])}",
                 va="center", fontsize=9)

    # Rating over time
    ax3.set_title("Rating Over Time")
    ax3.grid(True, axis="y", alpha=0.3)

    # raw
    ax3.plot(x, y, linewidth=1, alpha=0.35, label="Rating")
    # rolling average
    ax3.plot(x, y_roll, linewidth=2.0, label="20-game avg")

    ax3.set_ylabel("Rating")
    ax3.legend(loc="upper left", frameon=True)

    fig.savefig(OUT_DASHBOARD_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {"games": total, "wins": wins, "losses": losses, "win_rate": float(wr)}
    write_html_report(player, summary, streaks, OUT_REPORT_HTML)

    log("✅ Built dashboard + reports.")
    log(f" - {OUT_DASHBOARD_PNG}")
    log(f" - {OUT_REPORT_HTML}")
    log(f" - {OUT_H2H_CSV}")
    log(f" - {OUT_MAPS_CSV}")
    log(f" - {OUT_SUMMARY_CSV}")

    open_path(OUT_REPORT_HTML)
    open_path(OUT_DASHBOARD_PNG)
    open_output_folder()


# ============================================================
# GUI App
# ============================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # Bind a stable AppUserModelID so the taskbar icon shows correctly
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:
            pass

        apply_ttk_dark(self)

        self.title(f"{APP_TITLE}  v{APP_VERSION}")
        try:
            self.iconbitmap(resource_path("poxnora.ico"))
        except Exception:
            pass

        self.geometry("900x620")
        self.minsize(900, 620)

        self.cfg = load_config()

        self.player_var = tk.StringVar(value=str(self.cfg.get("player", "")))
        self.rank1v1_var = tk.BooleanVar(value=bool(self.cfg.get("only_ranked_1v1", True)))
        self.headless_var = tk.BooleanVar(value=bool(self.cfg.get("headless", False)))
        self.remember_var = tk.BooleanVar(value=bool(self.cfg.get("remember_login", True)))
        self.max_pages_var = tk.StringVar(value=str(self.cfg.get("max_pages", 2000)))

        self.continue_event = threading.Event()
        self.worker_running = False

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=14)
        top.pack(fill="x")

        title = ttk.Label(top, text=f"{APP_TITLE}", font=("Segoe UI", 14, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 8))

        ver = ttk.Label(top, text=f"v{APP_VERSION}", foreground="#9fb0c3")
        ver.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=(0, 8))

        ttk.Label(top, text="Player name:").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.player_var, width=30).grid(row=1, column=1, padx=8, sticky="w")

        ttk.Checkbutton(top, text="Ranked 1v1 only", variable=self.rank1v1_var).grid(row=1, column=2, padx=8, sticky="w")
        ttk.Checkbutton(top, text="Headless browser (advanced)", variable=self.headless_var).grid(row=1, column=3, padx=8, sticky="w")
        ttk.Checkbutton(top, text="Remember login", variable=self.remember_var).grid(row=1, column=4, padx=8, sticky="w")

        ttk.Label(top, text="Max pages:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.max_pages_var, width=10).grid(row=2, column=1, padx=8, sticky="w", pady=(8, 0))

        btns = ttk.Frame(self, padding=(14, 0, 14, 0))
        btns.pack(fill="x")

        self.btn_scrape = ttk.Button(btns, text="Scrape + Build (Recommended)", command=self.start_scrape_worker)
        self.btn_scrape.pack(side="left", padx=(0, 10), pady=10)

        self.btn_build = ttk.Button(btns, text="Build Only (from existing CSV)", command=self.build_only)
        self.btn_build.pack(side="left", padx=(0, 10), pady=10)

        self.btn_continue = ttk.Button(btns, text="Continue after login", command=self.signal_continue, state="disabled")
        self.btn_continue.pack(side="left", padx=(0, 10), pady=10)

        self.btn_updates = ttk.Button(btns, text="Check for updates", command=self.check_updates_clicked)
        self.btn_updates.pack(side="left", padx=(0, 10), pady=10)

        self.btn_open_output = ttk.Button(btns, text="Open Output Folder", command=open_output_folder)
        self.btn_open_output.pack(side="right", pady=10)

        status_frame = ttk.Frame(self, padding=(14, 0, 14, 0))
        status_frame.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left")

        self.progress = ttk.Progressbar(
            status_frame,
            mode="indeterminate",
            length=320,
            style="Blue.Horizontal.TProgressbar"
        )
        self.progress.pack(side="right")

        log_frame = ttk.Frame(self, padding=14)
        log_frame.pack(fill="both", expand=True)

        ttk.Label(log_frame, text="Log:").pack(anchor="w")

        self.log_text = tk.Text(
            log_frame, height=18, wrap="word",
            bg="#0f1622", fg="#e6edf3", insertbackground="#e6edf3",
            highlightthickness=1, highlightbackground="#223044"
        )
        self.log_text.pack(fill="both", expand=True, pady=(6, 0))

        self._log(f"Output folder: {OUTPUT_DIR}")
        self._log("Tip: Outputs always go to LocalAppData (no permission issues).")

    # ----------------- UI helpers -----------------
    def _log(self, msg: str):
        def _do():
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
            self.update_idletasks()
        self.after(0, _do)

    def start_busy(self, text: str):
        def _do():
            self.status_var.set(text)
            if str(self.progress["mode"]) == "indeterminate":
                self.progress.start(12)
        self.after(0, _do)

    def stop_busy(self, text: str = "Ready."):
        def _do():
            try:
                self.progress.stop()
            except Exception:
                pass
            self.status_var.set(text)
        self.after(0, _do)

    def set_progress(self, value: int, maximum: int, determinate: bool):
        def _do():
            if determinate:
                self.progress.stop()
                self.progress.config(mode="determinate", maximum=max(1, maximum))
                self.progress["value"] = max(0, min(value, maximum))
            else:
                self.progress.config(mode="indeterminate")
                self.progress.start(12)
        self.after(0, _do)

    def _save_cfg_from_ui(self):
        cfg = load_config()
        cfg["player"] = self.player_var.get().strip()
        cfg["only_ranked_1v1"] = bool(self.rank1v1_var.get())
        cfg["headless"] = bool(self.headless_var.get())
        cfg["remember_login"] = bool(self.remember_var.get())
        try:
            cfg["max_pages"] = int(self.max_pages_var.get().strip())
        except Exception:
            cfg["max_pages"] = 2000
        save_config(cfg)
        self.cfg = cfg

    def _disable_main_buttons(self):
        self.after(0, lambda: (
            self.btn_scrape.config(state="disabled"),
            self.btn_build.config(state="disabled"),
            self.btn_open_output.config(state="disabled"),
            self.btn_updates.config(state="disabled"),
        ))

    def _enable_main_buttons(self):
        self.after(0, lambda: (
            self.btn_scrape.config(state="normal"),
            self.btn_build.config(state="normal"),
            self.btn_open_output.config(state="normal"),
            self.btn_updates.config(state="normal"),
        ))

    # ----------------- actions -----------------
    def signal_continue(self):
        self.btn_continue.config(state="disabled")
        self.continue_event.set()

    def check_updates_clicked(self):
        def worker():
            self.start_busy("Checking for updates...")
            has_update, msg = check_for_updates()
            self.stop_busy("Ready.")
            if has_update:
                if messagebox.askyesno("Update available", msg + "\n\nOpen the release page in your browser?"):
                    owner, repo = GITHUB_REPO
                    if owner and repo:
                        webbrowser.open(f"https://github.com/{owner}/{repo}/releases/latest")
            else:
                messagebox.showinfo("Update check", msg)

        threading.Thread(target=worker, daemon=True).start()

    def start_scrape_worker(self):
        if self.worker_running:
            return

        self._save_cfg_from_ui()
        player = self.player_var.get().strip()
        if not player:
            messagebox.showerror("Missing player", "Enter your in-game player name.")
            return

        self.worker_running = True
        self.continue_event.clear()

        self._disable_main_buttons()
        self.btn_continue.config(state="disabled")
        self.start_busy("Opening browser for login...")

        def worker():
            try:
                headless = bool(self.cfg.get("headless", False))
                max_pages = int(self.cfg.get("max_pages", 2000))
                only_ranked_1v1 = bool(self.rank1v1_var.get())
                remember = bool(self.remember_var.get())

                with sync_playwright() as pw:
                    ensure_playwright_chromium(self._log)

                    browser = pw.chromium.launch(headless=headless)

                    context_kwargs = {}
                    if remember and STORAGE_STATE_FILE.exists():
                        # attempt to reuse prior login
                        context_kwargs["storage_state"] = str(STORAGE_STATE_FILE)

                    context = browser.new_context(**context_kwargs)
                    page = context.new_page()

                    self._log("Opening login page...")
                    page.goto(LOGIN_URL)

                    self.stop_busy("Login in browser, open Match History, then click Continue.")
                    self._log("👉 Log in manually in the browser window (unless it auto-logged you in).")
                    self._log("👉 After login, click Match History so you can SEE the match list.")
                    self._log("👉 Then click 'Continue after login' in this app.")

                    self.after(0, lambda: self.btn_continue.config(state="normal"))
                    self.continue_event.wait()

                    # Save login state after user confirms they're logged in (if enabled)
                    if remember:
                        try:
                            STORAGE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
                            context.storage_state(path=str(STORAGE_STATE_FILE))
                            self._log(f"✅ Saved login state: {STORAGE_STATE_FILE}")
                        except Exception as e:
                            self._log(f"⚠ Could not save login state: {e}")

                    self.start_busy("Scraping match history...")
                    scrape_to_clean_csv_by_clicking_next(
                        page,
                        max_pages=max_pages,
                        log=self._log,
                        progress_cb=lambda v, m, determinate: self.set_progress(v, m, determinate)
                    )

                    self.start_busy("Building dashboard...")
                    build_dashboard(player, only_ranked_1v1=only_ranked_1v1, log=self._log)

                    try:
                        browser.close()
                    except Exception:
                        pass

                self.stop_busy("Done. Output opened.")
                self._enable_main_buttons()
                self.after(0, lambda: messagebox.showinfo("Done", "Scrape + dashboard complete!\nOutputs opened automatically."))

            except Exception as e:
                self.stop_busy("Error.")
                self._log(f"❌ Error: {e}")
                self._enable_main_buttons()
                self.after(0, lambda: messagebox.showerror("Error", str(e)))

            finally:
                self.worker_running = False

        threading.Thread(target=worker, daemon=True).start()

    def build_only(self):
        self._save_cfg_from_ui()
        player = self.player_var.get().strip()
        if not player:
            messagebox.showerror("Missing player", "Enter your in-game player name.")
            return

        self._disable_main_buttons()
        self.start_busy("Building dashboard...")

        def worker():
            try:
                self.set_progress(0, 1, determinate=False)
                build_dashboard(player, only_ranked_1v1=bool(self.rank1v1_var.get()), log=self._log)
                self.set_progress(1, 1, determinate=True)
                self.stop_busy("Done. Output opened.")
                self._enable_main_buttons()
                self.after(0, lambda: messagebox.showinfo("Done", "Dashboard complete!\nOutputs opened automatically."))
            except Exception as e:
                self.stop_busy("Error.")
                self._log(f"❌ Error: {e}")
                self._enable_main_buttons()
                self.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()


def main():
    APP_DIR.mkdir(parents=True, exist_ok=True)
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
