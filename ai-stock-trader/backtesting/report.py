"""Report generation (JSON + Markdown + self-contained HTML with an equity
curve rendered as inline SVG — no plotting dependency)."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtesting import metrics as M
from backtesting.engine import BTResult

DISCLAIMER = ("Past (simulated) performance is not evidence of future results. Backtests are subject to model "
              "error, data errors and regime change.")


def build(res: BTResult, title: str) -> dict:
    t = res.trades_frame()
    rep = {"title": title, "generated_at": datetime.now(timezone.utc).isoformat(),
           "synthetic_data": res.synthetic_data, "params": res.params,
           "metrics": M.compute(t, res.equity, res.start_equity), "breakdown": M.breakdown(t),
           "monte_carlo": M.monte_carlo(t["pnl"].to_numpy() if len(t) else [], res.start_equity),
           "halts": [(str(a), b) for a, b in res.halts], "disclaimer": DISCLAIMER}
    return rep


def _svg(equity: pd.Series, w: int = 760, h: int = 220) -> str:
    if len(equity) < 2:
        return ""
    e = equity.resample("1h").last().dropna() if len(equity) > 2000 else equity
    lo, hi = float(e.min()), float(e.max())
    span = hi - lo or 1.0
    pts = " ".join(f"{i * w / (len(e) - 1):.1f},{h - (v - lo) / span * (h - 10) - 5:.1f}" for i, v in enumerate(e))
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="equity curve">'
            f'<polyline fill="none" stroke="#3fb68b" stroke-width="1.5" points="{pts}"/></svg>'
            f"<div class='muted'>min {lo:,.0f} · max {hi:,.0f}</div>")


def write(res: BTResult, out_dir: Path, title: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = build(res, title)
    stem = title.lower().replace(" ", "_")
    pj = out_dir / f"{stem}.json"
    pj.write_text(json.dumps(rep, indent=2, default=str))
    m = rep["metrics"]
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:,.4f}</td></tr>" if isinstance(v, float)
                   else f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in m.items())
    bd = "".join(f"<h3>{html.escape(sec)}</h3><table>" + "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(json.dumps(v))}</td></tr>" for k, v in vals.items())
        + "</table>" for sec, vals in rep["breakdown"].items())
    warn = "<p class='warn'>SYNTHETIC TEST DATA — results say nothing about real markets.</p>" if res.synthetic_data else ""
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui;background:#0e1117;color:#e6e6e6;max-width:900px;margin:auto;padding:16px}}
table{{border-collapse:collapse;width:100%}}td{{border-bottom:1px solid #333;padding:4px 8px}}
.warn{{color:#ff6b6b;font-weight:600}}.muted{{color:#888;font-size:12px}}</style></head><body>
<h1>{html.escape(title)}</h1>{warn}<p class="muted">{html.escape(DISCLAIMER)}</p>
<h2>Equity curve</h2>{_svg(res.equity)}<h2>Metrics</h2><table>{rows}</table>
<h2>Monte Carlo</h2><pre>{html.escape(json.dumps(rep['monte_carlo'], indent=1))}</pre>{bd}</body></html>"""
    ph = out_dir / f"{stem}.html"
    ph.write_text(page)
    md = [f"# {title}", "", "**SYNTHETIC DATA**" if res.synthetic_data else "", DISCLAIMER, "", "| metric | value |",
          "|---|---|"] + [f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in m.items()]
    pm = out_dir / f"{stem}.md"
    pm.write_text("\n".join(md) + "\n")
    return {"json": pj, "html": ph, "md": pm}
