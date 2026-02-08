from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union


def build_report_html(
    csv_path: Union[str, Path],
    output_path: Union[str, Path],
    df: Optional[Any] = None,
    player_name: Optional[str] = None,
    ranked_only: bool = False,
    **_kwargs: Any,
) -> Path:
    """Generate a modern HTML report (visual upgrade only)."""
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        output_path.write_text(f"<html><body><h1>Report</h1><p>Pandas not available: {e}</p></body></html>", encoding="utf-8")
        return output_path

    if df is None:
        df = pd.read_csv(csv_path)

    cols = {str(c).strip().lower(): c for c in getattr(df, "columns", [])}
    c_win = cols.get("win")
    c_loss = cols.get("loss")
    c_map = cols.get("map")
    c_ranked = cols.get("ranked")
    c_type = cols.get("type")

    d = df.copy()

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
        return s in {"1","true","t","yes","y","ranked","on"}

    if ranked_only and c_ranked:
        try:
            d = d[d[c_ranked].apply(_truthy)]
        except Exception:
            pass

    if c_type:
        try:
            d = d[d[c_type].astype(str).str.strip().str.lower().isin({"1v1","1 v 1"})]
        except Exception:
            pass

    pname = (player_name or "").strip()
    if not pname and c_win and c_loss:
        try:
            names = pd.concat([d[c_win], d[c_loss]], ignore_index=True)
            names = names.dropna().astype(str).str.strip()
            names = names[names != ""]
            if len(names):
                pname = str(names.value_counts().idxmax())
        except Exception:
            pname = ""

    pname_norm = pname.strip().lower()

    # derive opponents + map stats
    opp_list = []
    win_flag = []
    map_list = []

    if c_win and c_loss and pname_norm:
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

    opp_rows = []
    map_rows = []
    try:
        if opp_list:
            g = (
                pd.DataFrame({"Opponent": opp_list, "Win": win_flag})
                .groupby("Opponent", dropna=False)["Win"]
                .agg(["count","sum"])
                .rename(columns={"count":"Games","sum":"Wins"})
            )
            g["Losses"] = g["Games"] - g["Wins"]
            g["Winrate"] = (g["Wins"] / g["Games"]) * 100.0
            g = g.sort_values(["Games","Winrate"], ascending=[False,False]).head(15)
            for name, row in g.iterrows():
                opp_rows.append((str(name), int(row["Games"]), int(row["Wins"]), int(row["Losses"]), float(row["Winrate"])))

        if map_list:
            gm = (
                pd.DataFrame({"Map": pd.Series(map_list).astype(str).str.strip().replace({"":"Unknown","nan":"Unknown"}).fillna("Unknown"), "Win": win_flag})
                .groupby("Map")["Win"]
                .agg(["count","sum"])
                .rename(columns={"count":"Games","sum":"Wins"})
            )
            gm["Losses"] = gm["Games"] - gm["Wins"]
            gm["Winrate"] = (gm["Wins"] / gm["Games"]) * 100.0
            gm = gm.sort_values(["Games","Winrate"], ascending=[False,False]).head(15)
            for name, row in gm.iterrows():
                map_rows.append((str(name), int(row["Games"]), int(row["Wins"]), int(row["Losses"]), float(row["Winrate"])))
    except Exception:
        pass

    title = pname or "PoxNora"
    mode = "Ranked 1v1" if ranked_only else "All games"

    dashboard_name = "dashboard.png"
    dashboard_path = output_path.parent / dashboard_name

    def pill(p: float) -> str:
        return f"{p:.1f}%"

    def make_table(rows, first_col):
        if not rows:
            return "<div class='empty'>No data available.</div>"
        trs = []
        for n,g,w,l,wr in rows:
            trs.append(f"<tr><td class='name'>{n}</td><td>{g}</td><td>{w}</td><td>{l}</td><td><span class='pill'>{pill(wr)}</span></td></tr>")
        return f"""
        <div class='table-wrap'>
          <table>
            <thead><tr><th>{first_col}</th><th>Games</th><th>Wins</th><th>Losses</th><th>Winrate</th></tr></thead>
            <tbody>{''.join(trs)}</tbody>
          </table>
        </div>
        """

    css = """
    :root{
      --bg:#070B12;
      --panel:rgba(14,22,38,0.92);
      --line:rgba(255,255,255,0.07);
      --text:#EAF0FF;
      --muted:rgba(185,200,230,0.72);
      --blue:#3B82F6;
      --purple:#A855F7;
      --pink:#D946EF;
    }
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      color:var(--text);
      background:
        radial-gradient(900px 520px at 18% 0%, rgba(65,82,255,0.12), transparent 60%),
        radial-gradient(900px 520px at 85% 15%, rgba(217,70,239,0.08), transparent 55%),
        linear-gradient(180deg, #0B1220 0%, var(--bg) 55%);
    }
    .wrap{max-width:1100px;margin:0 auto;padding:28px 18px 48px;}
    .top{display:flex;gap:14px;align-items:flex-end;justify-content:space-between;margin-bottom:18px;}
    .title{font-size:28px;font-weight:900;letter-spacing:0.2px;}
    .meta{color:var(--muted);font-size:13px;margin-top:6px;}
    .badge{border:1px solid var(--line);background:rgba(255,255,255,0.03);padding:8px 10px;border-radius:999px;font-size:12px;color:var(--muted);}
    .badge strong{color:var(--text);font-weight:800;}
    .badges{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end;}
    .grid{display:grid;grid-template-columns:1fr;gap:14px;}
    .card{background:var(--panel);border:1px solid var(--line);border-radius:18px;overflow:hidden;box-shadow: 0 18px 40px rgba(0,0,0,0.35);}
    .card-h{padding:14px 16px;border-bottom:1px solid var(--line);background: linear-gradient(90deg, rgba(59,130,246,0.10), rgba(168,85,247,0.06));}
    .h2{font-size:16px;font-weight:900;}
    .sub{font-size:12px;color:var(--muted);margin-top:4px;}
    .img-wrap{padding:14px;}
    img{width:100%;height:auto;border-radius:14px;border:1px solid rgba(255,255,255,0.07);background:#000;}
    table{width:100%;border-collapse:collapse;}
    th,td{padding:10px 10px;font-size:13px;border-bottom:1px solid rgba(255,255,255,0.06);}
    th{color:rgba(220,230,255,0.90);text-align:left;font-weight:900;font-size:12px;letter-spacing:0.3px;text-transform:uppercase;}
    tr:hover td{background:rgba(255,255,255,0.02);}
    .name{font-weight:800;}
    .pill{display:inline-block;padding:6px 10px;border-radius:999px;border:1px solid rgba(217,70,239,0.35);background:rgba(217,70,239,0.10);color:#F0ABFC;font-weight:900;font-size:12px;}
    .empty{padding:16px;color:var(--muted);}
    .table-wrap{padding:8px 10px 12px;}
    @media (min-width: 980px){.grid{grid-template-columns:1.1fr 0.9fr;}.span2{grid-column:1 / span 2;}}
    """

    dash_html = ""
    if dashboard_path.exists():
        dash_html = f"""
        <div class='card span2'>
          <div class='card-h'>
            <div class='h2'>Dashboard</div>
            <div class='sub'>Latest analytics board • v26</div>
          </div>
          <div class='img-wrap'>
            <img src='{dashboard_name}' alt='Dashboard PNG' />
          </div>
        </div>
        """

    html = f"""<!doctype html>
    <html>
      <head>
        <meta charset='utf-8' />
        <meta name='viewport' content='width=device-width, initial-scale=1' />
        <title>PoxNora Report v26 — {title}</title>
        <style>{css}</style>
      </head>
      <body>
        <div class='wrap'>
          <div class='top'>
            <div>
              <div class='title'>Report v26 — {title}</div>
              <div class='meta'>Dataset: <b>{csv_path.name}</b> • Mode: <b>{mode}</b></div>
            </div>
            <div class='badges'>
              <div class='badge'><strong>{'ON' if ranked_only else 'OFF'}</strong> Ranked filter</div>
              <div class='badge'><strong>{len(d)}</strong> rows</div>
            </div>
          </div>

          <div class='grid'>
            {dash_html}

            <div class='card'>
              <div class='card-h'>
                <div class='h2'>Top Opponents</div>
                <div class='sub'>Most common matchups in the active dataset</div>
              </div>
              {make_table(opp_rows, 'Opponent')}
            </div>

            <div class='card'>
              <div class='card-h'>
                <div class='h2'>Top Maps</div>
                <div class='sub'>Highest-volume maps in the active dataset</div>
              </div>
              {make_table(map_rows, 'Map')}
            </div>
          </div>
        </div>
      </body>
    </html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
