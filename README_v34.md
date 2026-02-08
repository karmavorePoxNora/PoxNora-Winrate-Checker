# PoxNora Tracker (Desktop) — v34

A PySide6 desktop app for tracking PoxNora match history from a normalized CSV, with:
- Live dashboard (PyQtGraph): Rolling Winrate (last 200 games), Monthly Performance (last 5 months from latest scraped date)
- Opponents + Maps summaries (now includes search filters)
- Export: Dashboard PNG + HTML report

> **Versioning note**
> This repo keeps versioned entrypoints. For v34, run `main_v34.py`.

---

## Quick start (Windows)

1) **Install Python**
- Python 3.10+ recommended

2) **Create a virtual environment**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3) **Install dependencies**
```bash
pip install -r requirements_v34.txt
```

4) **Run the app**
```bash
python main_v34.py
```

---

## Data model (canonical CSV)

Required columns:
- `Win, Loss, Type, Ranked, Map, Rating, Date, Duration`

Opponent is derived (no literal Opponent column):
- if win → opponent = Loss
- if loss → opponent = Win

Date format:
- `%b %d, %Y %I:%M:%S %p`

---

## Exports

Inside the app:
- **Build Dashboard PNG**
- **Build HTML Report** (references the PNG)

Outputs are written under your project `output/` folder (path is shown in the app).

---

## After building (Windows)

After running `build_v34_windows.bat`, the app is **ready to launch**.

- Open: `dist\PoxNoraTracker\`
- Run: `PoxNoraTracker.exe`

Important:
- This is an **onedir** build — keep the whole `dist\PoxNoraTracker\` folder together.  
  Don’t move the `.exe` out by itself or Qt files will be missing.
- On first run, Windows may show **SmartScreen** (“unrecognized app”). Choose **More info → Run anyway** if you trust your build.

---

## Packaging (PyInstaller)

### Recommended: onedir (faster startup, fewer Qt edge cases)

From the repo root:
```bash
pip install pyinstaller
pyinstaller --noconsole --onedir --name PoxNoraTracker main_v34.py ^
  --collect-all PySide6 --collect-all pyqtgraph --collect-all matplotlib ^
  --add-data "ui;ui" --add-data "core;core"
```

Notes:
- **Windows** uses `;` in `--add-data "SRC;DEST"`.
- **macOS/Linux** uses `:` instead of `;` (example: `--add-data "ui:ui"`).

### Optional: onefile
`--onefile` works, but can be slower to launch and more likely to hit Qt plugin hiccups.

---

## What to commit to GitHub

Typical repo contents:
- `main_v34.py`
- `ui/` (all versioned ui files you ship, including v34)
- `core/` (PNG + HTML builders, dataset/stats, etc.)
- `requirements_v34.txt`
- `README_v34.md`

Do NOT commit:
- `.venv/`
- `__pycache__/`
- `output/` (generated files)
- `dist/`, `build/` (PyInstaller artifacts)

---

## Troubleshooting

### “No module named …”
Run:
```bash
pip install -r requirements_v34.txt
```

### Qt platform plugin / blank window (PyInstaller)
Use `--onedir` and ensure you included:
- `--collect-all PySide6`

If it persists, run from a terminal once to capture the exact error output.

