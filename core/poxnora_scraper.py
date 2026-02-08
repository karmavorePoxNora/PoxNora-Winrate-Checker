# core/poxnora_scraper.py
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
from playwright.sync_api import sync_playwright


LOGIN_URL = "https://www.poxnora.com/account/login.do"
MATCH_URL = "https://www.poxnora.com/account/matchhistory.do"

DEFAULT_COLUMNS = ["Win", "Loss", "Type", "Ranked", "Map", "Rating", "Date", "Duration"]

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int], None]


@dataclass(frozen=True)
class ScrapeConfig:
    output_dir: Path
    csv_path: Path
    recent_csv_path: Path
    browsers_dir: Optional[Path] = None  # if set, used for PLAYWRIGHT_BROWSERS_PATH
    headless: bool = False
    remember_login: bool = True
    storage_state_path: Optional[Path] = None
    max_pages: int = 2000


class ScrapeCancelled(RuntimeError):
    pass


# =====================================================
# Playwright / browser install helper (optional)
# =====================================================
def ensure_playwright_chromium(log: LogFn, browsers_dir: Optional[Path]) -> None:
    """
    Optional helper for packaged EXE flows: installs chromium once into a known folder.
    If browsers_dir is None, does nothing.
    """
    if not browsers_dir:
        return

    browsers_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers_dir)

    ok_marker = browsers_dir / ".chromium_ok"
    if ok_marker.exists():
        return

    log("Installing Playwright Chromium (first run only)...")
    subprocess.check_call(["python", "-m", "playwright", "install", "chromium"])
    ok_marker.touch()
    log("✅ Chromium installed.")


def stable_wait(page) -> None:
    """Best-effort settle to reduce race conditions on poxnora pages."""
    for _ in range(6):
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
            break
        except Exception:
            time.sleep(0.6)
    time.sleep(0.6)


# =====================================================
# Scrape + cleaning
# =====================================================
def scrape_rows(page) -> list[list[str]]:
    """
    Poxnora matchhistory appears as div rows with duplicated mirrored cells.
    We collect raw cell text for each row.
    """
    rows = page.locator("div.rowBg1, div.rowBg2")
    out: list[list[str]] = []
    for i in range(rows.count()):
        row = rows.nth(i)
        cells = row.locator("[class*='rowsCell']")
        row_data: list[str] = []
        for j in range(cells.count()):
            txt = cells.nth(j).inner_text().strip()
            if txt:
                row_data.append(txt)
        if row_data:
            out.append(row_data)
    return out


def rows_to_df_clean(raw_rows: list[list[str]]) -> pd.DataFrame:
    """
    Converts raw text rows into a clean DataFrame with DEFAULT_COLUMNS.
    Removes mirrored duplicates: Win,Win,Loss,Loss,... by taking every 2nd column.
    """
    if not raw_rows:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    max_cols = max(len(r) for r in raw_rows)
    df_raw = pd.DataFrame(raw_rows, columns=[f"col_{i+1}" for i in range(max_cols)])

    # Remove mirrored duplicates: Win,Win,Loss,Loss,... => take every 2nd col starting at 0
    df = df_raw.iloc[:, ::2].copy()

    if df.shape[1] < 8:
        raise RuntimeError(f"After mirror-cleaning, fewer than 8 columns were found ({df.shape[1]}).")

    df = df.iloc[:, :8].copy()
    df.columns = DEFAULT_COLUMNS

    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    return df.drop_duplicates()


# =====================================================
# Dedupe / merge helpers
# =====================================================
def _keyify(df: pd.DataFrame) -> pd.Series:
    """
    Strong-ish dedupe key. Not perfect, but good enough to prevent duplicates
    across re-scrapes / partial scrapes.
    """
    return (
        df["Win"].astype(str).str.lower().str.strip()
        + "|"
        + df["Loss"].astype(str).str.lower().str.strip()
        + "|"
        + df["Date"].astype(str).str.strip()
        + "|"
        + df["Map"].astype(str).str.strip()
        + "|"
        + df["Duration"].astype(str).str.strip()
        + "|"
        + df["Rating"].astype(str).str.strip()
    )


def df_dedupe_merge(existing: Optional[pd.DataFrame], incoming: pd.DataFrame) -> pd.DataFrame:
    if existing is None or len(existing) == 0:
        out = incoming.copy()
        out["_k"] = _keyify(out)
        out = out.drop_duplicates(subset=["_k"]).drop(columns=["_k"])
        return out

    merged = pd.concat([existing, incoming], ignore_index=True)
    merged["_k"] = _keyify(merged)
    merged = merged.drop_duplicates(subset=["_k"]).drop(columns=["_k"])
    return merged


# =====================================================
# Navigation / pagination
# =====================================================
def _try_go_matchhistory(page, log: LogFn) -> None:
    """Ensure we're on the Match History page with rows visible."""
    stable_wait(page)
    try:
        page.locator("h1.content_h1:has-text('Match History')").wait_for(timeout=6000)
        return
    except Exception:
        pass

    log("Not detecting Match History header. Opening match history...")
    page.goto(MATCH_URL, wait_until="networkidle")
    stable_wait(page)
    page.locator("h1.content_h1:has-text('Match History')").wait_for(timeout=10000)


def _click_next_page(page, log: LogFn) -> bool:
    """
    Try to paginate using a visible 'Next' link/button if present.
    """
    candidates = [
        "a[rel='next']",
        "a:has-text('Next')",
        "button:has-text('Next')",
        "a[aria-label*='Next' i]",
        "button[aria-label*='Next' i]",
        "li.pagination-next a",
        "a.pagination-next",
        "button.pagination-next",
        "a:has-text('›')",
        "a:has-text('»')",
        "button:has-text('›')",
        "button:has-text('»')",
        "a:has-text('>')",
        "button:has-text('>')",
    ]

    for sel in candidates:
        loc = page.locator(sel).first
        try:
            if loc.count() == 0:
                continue
            # Some render disabled; Playwright may throw on click, that's fine.
            loc.scroll_into_view_if_needed()
            page.wait_for_timeout(150)
            loc.click(timeout=1500)
            page.wait_for_load_state("networkidle", timeout=10000)
            page.wait_for_timeout(200)
            return True
        except Exception:
            continue

    return False


# poxnora uses ?p=1,2,3 style pagination often (even if a Next button exists)
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


def _with_page_num(url: str, new_p: int) -> str:
    u = urlparse(url)
    qs = parse_qs(u.query, keep_blank_values=True)
    qs["p"] = [str(new_p)]
    new_query = urlencode(qs, doseq=True)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))


def _current_page_num(url: str) -> int:
    try:
        u = urlparse(url)
        qs = parse_qs(u.query, keep_blank_values=True)
        return int(qs.get("p", ["1"])[0])
    except Exception:
        return 1


def _goto_next_by_url(page, log: LogFn) -> bool:
    """
    Fallback pagination: build next URL by incrementing p=.
    """
    cur_url = page.url
    cur_p = _current_page_num(cur_url)
    next_url = _with_page_num(cur_url, cur_p + 1)

    if next_url == cur_url:
        return False

    log(f"Going to next page: p={cur_p + 1}")
    try:
        page.goto(next_url, wait_until="networkidle")
        page.wait_for_timeout(200)
        return True
    except Exception:
        return False


def _next_page(page, log: LogFn) -> bool:
    """
    Unified next-page:
      1) Try clicking "Next"
      2) If that fails, try p= URL increment
    """
    if _click_next_page(page, log):
        stable_wait(page)
        return True

    if _goto_next_by_url(page, log):
        stable_wait(page)
        return True

    log("No next page detected — stopping.")
    return False


# =====================================================
# Scrape modes
# =====================================================
def scrape_recent_new_matches(
    page,
    target_new: int,
    max_pages: int,
    csv_path: Path,
    log: LogFn,
    progress_cb: Optional[ProgressFn] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> pd.DataFrame:
    """
    Scrape until we collect target_new NEW matches compared to existing csv_path.
    Saves nothing here; run_scrape handles merge + save.
    """
    _try_go_matchhistory(page, log)

    existing_keys: set[str] = set()
    if csv_path.exists():
        try:
            existing = pd.read_csv(csv_path)
            # only accept if it looks like a match-history csv
            if set(DEFAULT_COLUMNS).issubset({c.strip() for c in existing.columns}):
                tmp = existing[DEFAULT_COLUMNS].copy()
                tmp["_k"] = _keyify(tmp)
                existing_keys = set(tmp["_k"].tolist())
        except Exception:
            existing_keys = set()

    log(f"✅ Scraping until +{target_new} new matches are found...")
    collected_rows: list[list[str]] = []
    seen_page_sig: set[str] = set()

    def page_signature(rows: list[list[str]]) -> Optional[str]:
        if not rows:
            return None
        sample = rows[:3]
        return "||".join(["|".join(r[:8]) for r in sample])

    new_count = 0
    page_num = 0

    if progress_cb:
        progress_cb(new_count, target_new)

    while page_num < max_pages and new_count < target_new:
        if cancelled and cancelled():
            raise ScrapeCancelled("Cancelled")

        rows = scrape_rows(page)
        if not rows:
            log("No rows detected — stopping.")
            break

        sig = page_signature(rows)
        if sig and sig in seen_page_sig:
            log("Repeated page detected — stopping.")
            break
        if sig:
            seen_page_sig.add(sig)

        df_page = rows_to_df_clean(rows)
        df_page["_k"] = _keyify(df_page)

        for r in df_page.itertuples(index=False):
            k = r[-1]  # last entry is _k
            if k not in existing_keys:
                existing_keys.add(k)
                collected_rows.append(list(r[:-1]))
                new_count += 1
                if progress_cb:
                    progress_cb(new_count, target_new)
                if new_count >= target_new:
                    break

        log(f"Page {page_num + 1}: new +{new_count}/{target_new}")

        # paginate (THIS MUST BE INSIDE THE LOOP)
        page_num += 1
        if new_count >= target_new:
            break
        if page_num >= max_pages:
            break
        if not _next_page(page, log):
            break

    if not collected_rows:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    df_new = pd.DataFrame(collected_rows, columns=DEFAULT_COLUMNS)
    df_new["_k"] = _keyify(df_new)
    df_new = df_new.drop_duplicates(subset=["_k"]).drop(columns=["_k"])
    return df_new


def scrape_all_pages(
    page,
    max_pages: int,
    log: LogFn,
    cancelled: Optional[Callable[[], bool]] = None,
) -> pd.DataFrame:
    """
    Scrape match history pages until Next is not available or max_pages reached.
    """
    _try_go_matchhistory(page, log)

    collected: list[pd.DataFrame] = []
    seen_page_sig: set[str] = set()

    def page_signature(df: pd.DataFrame) -> str:
        head = df.head(3).astype(str).fillna("")
        return "||".join(["|".join(row.tolist()) for _, row in head.iterrows()])

    for i in range(max_pages):
        if cancelled and cancelled():
            raise ScrapeCancelled("Cancelled")

        rows = scrape_rows(page)
        if not rows:
            log("No rows detected — stopping.")
            break

        df_page = rows_to_df_clean(rows)
        sig = page_signature(df_page)
        if sig in seen_page_sig:
            log("Repeated page detected — stopping.")
            break
        seen_page_sig.add(sig)

        collected.append(df_page)
        log(f"Page {i + 1}: rows {len(df_page)}")

        if not _next_page(page, log):
            break

    if not collected:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    out = pd.concat(collected, ignore_index=True)
    out["_k"] = _keyify(out)
    out = out.drop_duplicates(subset=["_k"]).drop(columns=["_k"])
    return out


# =====================================================
# Public entrypoint called by the UI
# =====================================================
def run_scrape(
    cfg: ScrapeConfig,
    mode: str,
    target_new: int,
    log: LogFn,
    progress_cb: Optional[ProgressFn] = None,
    wait_for_continue: Optional[Callable[[], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> Tuple[Path, pd.DataFrame]:
    """
    mode:
      - "all"    : scrape all pages (up to max_pages)
      - "recent" : scrape until +target_new new matches found vs existing CSV
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_playwright_chromium(log, cfg.browsers_dir)

    storage_state = None
    if cfg.remember_login and cfg.storage_state_path and cfg.storage_state_path.exists():
        storage_state = str(cfg.storage_state_path)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=cfg.headless)
        context = browser.new_context(storage_state=storage_state)
        page = context.new_page()

        log("Opening login page...")
        page.goto(LOGIN_URL)
        stable_wait(page)

        # manual login handshake (matches your UI flow)
        log("👉 Log in manually in the browser window.")
        log("👉 After login, open Match History so you can SEE the match list.")
        log("👉 Then click Continue in the app.")

        if wait_for_continue:
            wait_for_continue()

        if cancelled and cancelled():
            raise ScrapeCancelled("Cancelled")

        # Save storage state (remember login) after user continues
        if cfg.remember_login and cfg.storage_state_path:
            try:
                cfg.storage_state_path.parent.mkdir(parents=True, exist_ok=True)
                context.storage_state(path=str(cfg.storage_state_path))
                log(f"✅ Saved login state -> {cfg.storage_state_path}")
            except Exception:
                pass

        if mode == "recent":
            df_new = scrape_recent_new_matches(
                page=page,
                target_new=target_new,
                max_pages=cfg.max_pages,
                csv_path=cfg.csv_path,
                log=log,
                progress_cb=progress_cb,
                cancelled=cancelled,
            )

            df_existing: Optional[pd.DataFrame] = None
            if cfg.csv_path.exists():
                try:
                    df_existing = pd.read_csv(cfg.csv_path)
                except Exception:
                    df_existing = None

            df_merged = df_dedupe_merge(df_existing, df_new)

            df_merged.to_csv(cfg.csv_path, index=False)
            df_new.to_csv(cfg.recent_csv_path, index=False)

            log(f"✅ Saved merged CSV: {cfg.csv_path}")
            log(f"✅ Saved recent scrape CSV: {cfg.recent_csv_path}")

            try:
                context.close()
                browser.close()
            except Exception:
                pass

            return cfg.csv_path, df_merged

        if mode == "all":
            df_all = scrape_all_pages(page=page, max_pages=cfg.max_pages, log=log, cancelled=cancelled)
            df_all.to_csv(cfg.csv_path, index=False)
            log(f"✅ Saved CSV: {cfg.csv_path}")

            try:
                context.close()
                browser.close()
            except Exception:
                pass

            return cfg.csv_path, df_all

        raise ValueError(f"Unknown mode: {mode}")
