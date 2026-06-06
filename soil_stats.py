"""Global-land summary statistics for the focus "numbers" callouts in app.py.

Area-weighted (cos-lat, so polar cells don't dominate) summaries computed from
the model overlay grid: the mean soil-temperature change, and the share of land
each focus overlay covers, at 1950, today (2025), and 2080 under each projection
rate (r30 / r50). Kept in its own module so app.py stays about layout.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def summary_stats(overlay_path, cache_key=None):
    """Return the numbers behind each focus, or {} if the overlay isn't built.
    `cache_key` (the overlay file mtime) is only here to bust the cache when
    the pipeline regenerates the overlay."""
    if not os.path.exists(overlay_path):
        return {}
    ov = pd.read_parquet(overlay_path)
    ov["w"] = np.cos(np.radians(ov["lat"]))

    def sub(rate, yr):
        return ov[(ov["rate"] == rate) & (ov["year"] == yr)]

    def wmean(s):
        return float(np.average(s["soil_temp"], weights=s["w"]))

    def pct(s, cond):                      # area-weighted % of land meeting cond
        return float(100 * s.loc[cond, "w"].sum() / s["w"].sum())

    h1950, h2025 = sub("hist", 1950), sub("hist", 2025)
    out = {"temp": {"t1950": wmean(h1950), "t2025": wmean(h2025),
                    "past": wmean(h2025) - wmean(h1950),
                    "r30": wmean(sub("r30", 2080)) - wmean(h2025),
                    "r50": wmean(sub("r50", 2080)) - wmean(h2025)}}

    def area_block(cond_fn):
        d = {"y1950": pct(h1950, cond_fn(h1950)),
             "y2025": pct(h2025, cond_fn(h2025))}
        for tag in ("r30", "r50"):
            s = sub(tag, 2080)
            d[tag] = pct(s, cond_fn(s))
        return d

    out["permafrost"] = area_block(lambda s: s["soil_temp"] < 0)
    out["crop_limit"] = area_block(
        lambda s: (s["heat_aridity"] < 0.2) & (s["soil_temp"] > 0))
    out["habitability"] = area_block(lambda s: s["habitability"] >= 0.5)
    return out


def focus_summary_md(key, rate_tag, stats):
    """One-paragraph 'the numbers' callout for the active focus; the 2080 figure
    reflects the selected future warming rate (r30 / r50)."""
    if not stats:
        return ""
    rl = "recent 30-yr trend" if rate_tag == "r30" else "long-term 50-yr trend"
    if key == "all":
        t = stats["temp"]
        lo, hi = sorted((t["r30"], t["r50"]))
        return (f"**The numbers.** Area-weighted across all land, mean annual soil "
                f"temperature rose about **+{t['past']:.1f} °C** from 1950 to 2025 "
                f"({t['t1950']:.1f} → {t['t2025']:.1f} °C). Extending past trends, "
                f"it warms a further **+{lo:.1f} to +{hi:.1f} °C** by 2080 "
                f"(50-yr vs 30-yr trend).")
    if key == "permafrost":
        a = stats["permafrost"]
        now, fut = a["y2025"], a[rate_tag]
        return (f"**The numbers.** Permafrost-favorable land (annual mean soil "
                f"temp below 0 °C) covers about **{now:.1f}% of land today**, down "
                f"from **{a['y1950']:.1f}% in 1950**. By 2080 it shrinks to about "
                f"**{fut:.1f}%** ({rl}) — roughly **{100 * (now - fut) / now:.0f}% "
                f"less** frozen ground than today.")
    if key == "crop_limit":
        a = stats["crop_limit"]
        now, fut = a["y2025"], a[rate_tag]
        return (f"**The numbers.** Heat- and drought-limited land covers about "
                f"**{now:.1f}% of land today**, up from **{a['y1950']:.1f}% in "
                f"1950**. By 2080 it spreads to about **{fut:.1f}%** ({rl}) — about "
                f"**{100 * (fut - now) / now:.0f}% more** unfarmably hot-and-dry "
                f"ground.")
    if key == "habitability":
        a = stats["habitability"]
        now, fut = a["y2025"], a[rate_tag]
        return (f"**The numbers.** Highly habitable land (index ≥ 0.5) covers about "
                f"**{now:.1f}% of land today**, near its **{a['y1950']:.1f}%** in "
                f"1950. By 2080 the net total moves to about **{fut:.1f}%** ({rl}) "
                f"— but the bigger change is *where*: the livable band shifts "
                f"**poleward**, gaining in the high north while losing across the "
                f"subtropics.")
    return ""
