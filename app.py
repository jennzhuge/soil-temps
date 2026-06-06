"""
Interactive global soil-temperature map.

Run with:
    ../datavis/bin/streamlit run app.py

Built with Streamlit + Plotly so it's easy to customize:
  - Add a new FOCUS:      add a dict to the FOCUSES list (see section 3).
  - Richer hover popups:  edit build_hover_text().
  - Colors / styling:     tweak the go.Scattermap / go.Densitymap traces.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# 1. DATASETS
# To add another dataset later, just add a line here:
#   "Friendly name shown in the dropdown": "path/to/file.csv"
# Each CSV is expected to have columns: longitude, latitude, AnnualTs
# ---------------------------------------------------------------------------
DATASETS = {
    "Soil temperature (AnnualTs)": "dataset.csv",
}

# Which column holds the temperature value.
VALUE_COLUMN = "AnnualTs"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load a CSV, coerce numbers, and drop rows with missing values."""
    df = pd.read_csv(path)
    for col in ("longitude", "latitude", VALUE_COLUMN):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["longitude", "latitude", VALUE_COLUMN])
    return df.reset_index(drop=True)


def build_hover_text(row: pd.Series) -> str:
    """
    Text shown in the popup overlay when you hover over a point.

    >>> ADD CLIMATE INFO HERE LATER (climate zone, projected change, etc.). <<<
    """
    lines = [
        f"<b>{row[VALUE_COLUMN]:.2f} °C</b>",
        f"Latitude: {row['latitude']:.4f}",
        f"Longitude: {row['longitude']:.4f}",
    ]
    return "<br>".join(lines)


def sources_md(sources) -> str:
    """Render a list of (title, url) pairs as a ' · '-separated markdown line."""
    return " · ".join(f"[{title}]({url})" for title, url in sources)


# ---------------------------------------------------------------------------
# 3. FOCUSES  (preset overlays shown as tiles)
# ===========================================================================
# This is the part to extend. Each focus is one dict:
#
#   {
#     "key":   unique id string,
#     "label": title shown on the tile,
#     "blurb": one-line description on the tile,
#     "overlay": how to shade the map (see _overlay_trace):
#         {"kind": "none"}                              -> dots only, no fill
#         {"kind": "threshold", "col": ..., "thr": ...,  -> fill where col < thr
#          "fill": "R,G,B", "name": ...}
#         {"kind": "gradient", "col": ..., "fill": "R,G,B", "name": ...}
#                                                       -> fill graded by col (0-1)
#     "explanation": markdown shown when the focus is active,
#     "detail_images": list of (image_path, caption) shown when active,
#     "sources": list of (title, url) shown as links under the explanation,
#   }
#
# To add a new focus, copy a block and change the values.
# ---------------------------------------------------------------------------

PERMAFROST_EXPLANATION = """
**Permafrost** is ground that stays at or below 0 °C for at least two
consecutive years. A **mean annual soil temperature below 0 °C is the standard
indicator** that permafrost exists *at depth* — but it does **not** mean the
ground is frozen year-round at the surface. The top **"active layer" still
thaws every summer**, even where permafrost is present below; the annual *mean*
stays sub-zero because the long, cold winter outweighs the short summer thaw, so
heat never penetrates deep enough to thaw the layer beneath. 

The areas shown in blue are where the annual mean soil temperature is below 0 °C
in the selected year — **real NCEP data for past years**, the model's projection
for future years. Treat this filter as **permafrost-favourable**, not confirmed:
it's a 0–10 cm reading at coarse ~1.9° resolution, and a single annual mean can't
verify the two-year part of the definition.

**Why thawing permafrost matters**

- **Carbon feedback loop.** Permafrost locks away roughly 1,500 billion tonnes
  of organic carbon — about twice the carbon currently in the atmosphere. As it
  thaws, microbes decompose that material and release **CO₂ and methane**,
  which drives further warming, which thaws more permafrost.
- **Methane.** Much of the release is methane, a greenhouse gas dozens of times
  more potent than CO₂ over a 20-year span.
- **Ground collapse.** Thawing destabilises the soil, causing subsidence,
  landslides ("thaw slumps"), and damage to roads, pipelines, and buildings.
- **Ecosystem & hydrology shifts.** Drainage, wetlands, and habitats change as
  frozen ground gives way.
- **Buried hazards.** Thaw can release stored mercury and long-dormant microbes.

As the climate warms, these are the soils most at risk of crossing the 0 °C
threshold — making this an early-warning map.
"""

CROP_LIMIT_EXPLANATION = """
Land drops out of food production when it gets **too hot or too dry** — so this
filter scores each place on both and combines them (**0 = unfarmable, 1 = fine**):

- **heat** — fine in mild climates, falling toward 0 as it passes the **~29 °C**
  human-climate-niche edge (Xu et al. 2020), *multiplied by*
- **aridity** — rainfall ÷ evaporative demand; ~1 in humid climates, ~0 in
  deserts (land is "dryland" below 0.65; UNCCD).

Because they're **multiplied**, a place fails if *either* is near 0 — too hot,
too dry, or both. And they reinforce each other: warmer air dries the soil faster,
which is the compounding engine of **desertification**.

The **red** shows where that score is near zero in the selected year. The headline
is that it **spreads** toward 2080 — that growth is the desertification story.
(Frozen ground is excluded, so cold deserts don't turn red.) Treat it as
heat/desertification *risk*, not a hard line — the real limit also shifts with
rainfall, irrigation, and soil.
"""

HABITABILITY_EXPLANATION = """
**The habitability index (0–1).** Each location gets three sub-scores, every one
running from **0 (unsuitable)** to **1 (ideal)**, which we then multiply together:

- **Not too cold** — scores **0 when the soil is below freezing** (nothing grows in
  frozen ground) and rises to **1** at mild temperatures.
- **Not too hot** — scores **1** at mild temperatures and drops to **0 above
  ~29 °C**, the edge of the human climate niche (Xu et al. 2020).
- **Moisture** — from the aridity index (rainfall ÷ evaporative demand): near
  **0** in hyper-arid deserts, near **1** in humid climates (land counts as
  "dryland" below 0.65; UNCCD).

Because the three are **multiplied**, a place needs *all three* to score well —
not too cold, not too hot, **and** watered. If any one is near 0, the whole index
is near 0. On the map, the **green shading is the model's habitability index for
the selected year** — deeper green = more habitable — fading to nothing where
land is too cold, too hot, or too dry.

The headline of *this* filter is the **poleward shift**: unlike *Cropland heat
limit* (heat & dryness only), habitability also counts **cold**, so as the
climate warms you see the livable band **gain ground in the high north while
losing it across the subtropics** — not just a hot edge expanding. Press Play to
watch it move.

It's a deliberately simple, transparent index (not a crop model). Checked against
real ground truth, known farmland regions average **0.92** and barren deserts/ice
**0.11**.
"""

# The "All sites" tile doubles as the data/methodology panel.
DATA_INFO = """
**What this map shows.** Annual mean soil temperature across the globe — **real
data up to ~2025, then a model's projection out to 2080.** Press ▶ Play to watch
it change year by year.

**1 · The dots: Restor.** The 2,015 dots are real sites from
[Restor](https://restor.eco) — our real-world anchor. But `dataset.csv` gives only
*one* long-term average per site: too sparse to color a globe, and with no
year-by-year history to show change over time.

**2 · Why we added NCEP data.** To get soil temperature *everywhere, every year*,
we pulled in the **NCEP/NCAR Reanalysis** (NOAA PSL): gridded **monthly** soil
temperature, air temperature, and precipitation, worldwide, **1948–2025**. We
average each year's 12 months into annual values. This becomes the backbone of the
map and the data the model learns from.

**3 · The model (supervised learning).** We train an **XGBoost** model to answer
one question: *given a place's climate, what is its soil temperature?* The
**inputs** are 7 climate features (below) built from NCEP air temp + precip; the
**answer it learns from** (the label) is the real NCEP soil temperature. Every
land cell in every year 1948–2025 is one training example (~660,000), and we
score it on **held-out years** so it reflects real prediction, not memorisation.

**4 · Past = real, future = predicted.** Up to **~2025** the map shows the
**actual** NCEP soil temperature — no model. From **2030** on it shows the
**model's projection**. (A small step can appear at the hand-off; we leave it, to
be honest about where measurement ends and prediction begins.)

**5 · How the future is built.** The model only translates climate → soil temp,
so to reach a future year we:
1. **Project the climate forward** — for each place we measure how fast each
   feature (temperature, rainfall, …) has changed recently and extend that trend
   to the target year (the *30-yr* / *50-yr* toggle picks how fast).
2. **Run the model on it** — feed that projected future climate into the trained
   model, and it returns the projected soil temperature.
So future soil temperatures are the model reading a trend-extrapolated future
climate.

**The 7 features** (per location, from NCEP air + precip): mean annual air
temperature · annual precipitation · temperature seasonality · warmest-month
temp · coldest-month temp · **PET** (Thornthwaite evaporative demand) ·
**aridity index** (rainfall ÷ PET).

**Worth knowing.** NCEP is a *reanalysis* (a weather model blended with
observations), the 0–10 cm layer, on a coarse **~1.9°** grid. Air temperature is
by far the strongest predictor — so the model's high accuracy is *expected
physics*, not a discovery — and the future is a straight-line **trend
extrapolation** with no feedback physics: read it as a floor, not a forecast.
"""

DATA_SOURCES = [
    ("Restor (the highlighted sites)", "https://restor.eco"),
    ("Soil temp + air temp + precip: NCEP/NCAR Reanalysis (NOAA PSL)",
     "https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html"),
    ("Thornthwaite PET method — Wikipedia",
     "https://en.wikipedia.org/wiki/Potential_evaporation"),
    ("Human climate niche — Xu et al. 2020 (PNAS)",
     "https://www.pnas.org/doi/10.1073/pnas.1910114117"),
    ("Aridity / drylands — UNCCD",
     "https://www.unccd.int/sites/default/files/2024-12/aridity_report.pdf"),
    ("Permafrost — NSIDC",
     "https://nsidc.org/learn/parts-cryosphere/frozen-ground-permafrost"),
]

FOCUSES = [
    {
        "key": "all",
        "label": "All sites",
        "blurb": "Observed sites + about the data.",
        # no shaded overlay — show the observed dots alone
        "overlay": {"kind": "none"},
        "explanation": DATA_INFO,
        "detail_images": [],
        "sources": DATA_SOURCES,
    },
    {
        "key": "permafrost",
        "label": "Permafrost-favourable",
        "blurb": "Blue: soil below 0 °C — shrinks over time.",
        # overlay: blue shaded fill where predicted soil temp < 0 (shrinks as it warms)
        "overlay": {"kind": "threshold", "col": "soil_temp", "thr": 0.0,
                    "fill": "33,102,172", "name": "soil temp < 0 °C"},
        "explanation": PERMAFROST_EXPLANATION,
        "detail_images": [
            ("images/batagaika_crater.jpg",
             "The Batagaika crater, Siberia — a 'megaslump' that keeps growing "
             "as permafrost thaws. (NASA)"),
            ("images/permafrost_thaw_pond.jpg",
             "Thaw ponds near Abisko, Sweden, forming as frozen ground melts."),
        ],
        "sources": [
            ("Permafrost — Wikipedia", "https://en.wikipedia.org/wiki/Permafrost"),
            ("Active layer — Wikipedia", "https://en.wikipedia.org/wiki/Active_layer"),
            ("Frozen Ground & Permafrost — NSIDC",
             "https://nsidc.org/learn/parts-cryosphere/frozen-ground-permafrost"),
            ("GTN-P Measurement Standards & Guidelines",
             "https://gtnp.arcticportal.org/data/measurement-standards-and-monitoring-guidelines"),
            ("Images: Wikimedia Commons — Effects of thawing permafrost",
             "https://commons.wikimedia.org/wiki/Category:Effects_of_thawing_permafrost"),
        ],
    },
    {
        "key": "crop_limit",
        "label": "Heat & desertification",
        "blurb": "Red: land lost to heat + drought — spreads over time.",
        # overlay: red fill where heat+aridity suitability < 0.2 AND the ground is
        # warm (soil > 0 °C) — the gate keeps frozen polar/high deserts, which are
        # cold-limited not heat-limited, from showing up red.
        "overlay": {"kind": "threshold", "col": "heat_aridity", "thr": 0.2,
                    "gate_col": "soil_temp", "gate_min": 0.0,
                    "fill": "178,24,43", "name": "heat/aridity-limited"},
        "explanation": CROP_LIMIT_EXPLANATION,
        "detail_images": [
            ("images/cracked_dry_soil.jpg",
             "Cracked, desiccated ground — a hallmark of drought and "
             "desertification in drylands."),
        ],
        "sources": [
            ("Xu et al. 2020 — Future of the human climate niche (PNAS)",
             "https://www.pnas.org/doi/10.1073/pnas.1910114117"),
            ("Human climate niche — Wikipedia",
             "https://en.wikipedia.org/wiki/Human_climate_niche"),
            ("Extreme heat & soil temperature on crops — UNL CropWatch",
             "https://cropwatch.unl.edu/2016/impacts-extreme-heat-stress-and-increased-soil-temperature-plant-growth-and-development/"),
            ("Aridity index — Wikipedia",
             "https://en.wikipedia.org/wiki/Aridity_index"),
            ("Aridity trends & projections — UNCCD",
             "https://www.unccd.int/sites/default/files/2024-12/aridity_report.pdf"),
            ("Image: Wikimedia Commons — Drought",
             "https://commons.wikimedia.org/wiki/Category:Drought"),
        ],
    },
    {
        "key": "habitability",
        "label": "Habitability index",
        "blurb": "Green: livable land — shifts poleward over time.",
        # overlay: green shaded fill graded by the habitability index (0-1)
        "overlay": {"kind": "gradient", "col": "habitability", "fill": "0,104,55",
                    "name": "habitability"},
        "explanation": HABITABILITY_EXPLANATION,
        "detail_images": [],
        "sources": [
            ("Human climate niche — Xu et al. 2020 (PNAS)",
             "https://www.pnas.org/doi/10.1073/pnas.1910114117"),
            ("Aridity index / drylands — UNCCD",
             "https://www.unccd.int/sites/default/files/2024-12/aridity_report.pdf"),
        ],
    },
]

FOCUS_BY_KEY = {f["key"]: f for f in FOCUSES}


# ---------------------------------------------------------------------------
# 4. MODEL OUTPUTS + HELPERS (definitions; nothing renders until section 5)
# ---------------------------------------------------------------------------
OVERLAY_PATH = "outputs/overlay_grid.parquet"
POINTS_PATH = "outputs/points_timeseries.parquet"
METRICS_PATH = "outputs/metrics.json"

TS_SCALE, TS_MIN, TS_MAX = "RdYlBu_r", -15, 35   # soil-temp color scale (≤-15 = deep blue)
HIST_END = 2025                                  # last year of REAL soil temp

# Short methodology recap + skill numbers for the expander (full story lives in
# the All-sites panel; the 30/50-yr rates are explained under the trend chart).
PROJECTION_INFO = """
Soil temperature is real **NCEP/NCAR Reanalysis** data up to ~%d, then an
**XGBoost** model's prediction (climate → soil temperature, run on each location's
trend-extrapolated climate). Full walkthrough: open the **All sites** tile.

⚠️ **Reading the skill numbers below:** the R² is very high mainly because soil
temperature ≈ air temperature + a small offset — expected physics, not a
discovery. Drop the air-temperature features and it collapses, confirming air
temp is the dominant driver (which is why we keep it).
""" % HIST_END



@st.cache_data
def load_outputs():
    """Animated overlay grid + observed-point time series (None if pipeline unrun)."""
    import os
    import json
    if not (os.path.exists(OVERLAY_PATH) and os.path.exists(POINTS_PATH)):
        return None, None, {}
    overlay = pd.read_parquet(OVERLAY_PATH)
    points = pd.read_parquet(POINTS_PATH)
    overlay["hover"] = overlay["soil_temp"].round(1).astype(str) + " °C"
    points["hover"] = (points["soil_temp"].round(1).astype(str)
                       + " °C · observed site")
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return overlay, points, metrics


def select_focus(focus):
    st.session_state.active_focus = focus["key"]


def _overlay_trace(dfy, focus, ov_cbar=None):
    """A transparent shaded FILL (density heatmap) for the focus, or None."""
    ov = focus["overlay"]
    rgb = ov.get("fill")
    # transparent -> solid fill of the focus color (alpha encodes the value)
    fade = [[0, f"rgba({rgb},0)"], [1, f"rgba({rgb},1)"]]
    if ov["kind"] == "threshold":
        # binary region where a column is below a threshold (uniform fill);
        # an optional gate keeps cold/frozen cells out (e.g. cold deserts)
        cond = dfy[ov["col"]] < ov["thr"]
        if "gate_col" in ov:
            cond &= dfy[ov["gate_col"]] > ov["gate_min"]
        sel = dfy[cond]
        return go.Densitymap(
            lat=sel["lat"], lon=sel["lon"], z=[1] * len(sel),
            radius=18, zmin=0, zmax=1, showscale=False, opacity=0.55,
            colorscale=fade, hoverinfo="skip", name=ov["name"],
        )
    if ov["kind"] == "gradient":
        # continuous fill graded by a value column (fades out at low values)
        return go.Densitymap(
            lat=dfy["lat"], lon=dfy["lon"], z=dfy[ov["col"]],
            radius=14, zmin=0, zmax=ov.get("zmax", 1.0), opacity=0.6,
            colorscale=fade, showscale=ov_cbar is not None, colorbar=ov_cbar,
            hoverinfo="skip", name=ov["name"],
        )
    return None


def _points_trace(dfy, pts_cbar, pts_color):
    """Observed-site dots — the only markers. `pts_color` sets the column +
    color scale (absolute soil temp, or change-vs-present)."""
    return go.Scattermap(
        lat=dfy["lat"], lon=dfy["lon"], mode="markers", name="observed sites",
        text=dfy["hover"], hovertemplate="%{text}<extra></extra>",
        marker=dict(size=6, color=dfy[pts_color["col"]],
                    colorscale=pts_color["scale"], cmin=pts_color["cmin"],
                    cmax=pts_color["cmax"], opacity=1.0,
                    showscale=True, colorbar=pts_cbar),
    )


def _frame_traces(ov_df, pt_df, focus, pts_cbar, ov_cbar, pts_color):
    """Traces for one year: shaded fill (if any) below the observed dots."""
    fill = _overlay_trace(ov_df, focus, ov_cbar)
    return ([fill] if fill is not None else []) + [_points_trace(pt_df, pts_cbar, pts_color)]


def build_animation(focus, overlay_df, points_df, change_mode=False):
    """go.Figure with one frame per year: shaded fill + observed dots.
    change_mode colors the dots by warming-since-present (Δ°C) instead of
    absolute soil temperature, so the projected warming is visible."""
    years = sorted(overlay_df["year"].unique())
    start = 2020 if 2020 in years else years[0]
    ov_by = {y: overlay_df[overlay_df["year"] == y] for y in years}

    # Dot color: absolute soil temp, or change-vs-present (Δ°C, diverging).
    if change_mode:
        base_yr = int(points_df.loc[~points_df["is_future"], "year"].max())
        base = (points_df[points_df["year"] == base_yr][["lat", "lon", "soil_temp"]]
                .drop_duplicates(["lat", "lon"]).rename(columns={"soil_temp": "_b"}))
        points_df = points_df.merge(base, on=["lat", "lon"], how="left")
        points_df["disp"] = points_df["soil_temp"] - points_df["_b"]
        pts_color = dict(col="disp", scale="RdBu_r", cmin=-4, cmax=4)
        cbar_title = f"Δ°C vs {base_yr}"
    else:
        points_df = points_df.copy()
        points_df["disp"] = points_df["soil_temp"]
        pts_color = dict(col="disp", scale=TS_SCALE, cmin=TS_MIN, cmax=TS_MAX)
        cbar_title = "°C"
    pt_by = {y: points_df[points_df["year"] == y] for y in years}

    # Legends: the dots carry the temp/Δ scale. A gradient overlay (habitability)
    # adds a second 0-1 color bar placed SIDE BY SIDE with it.
    is_grad = focus["overlay"]["kind"] == "gradient"
    if is_grad:
        pts_cbar = dict(title=cbar_title, x=1.01, xanchor="left", y=0.5,
                        yanchor="middle", len=0.92)
        ov_cbar = dict(title="Habitability", x=1.12, xanchor="left", y=0.5,
                       yanchor="middle", len=0.92,
                       tickmode="array", tickvals=[0, 0.25, 0.5, 0.75, 1.0])
    else:
        pts_cbar, ov_cbar = dict(title=cbar_title), None

    def label_for(y):
        kind = "observed climate" if y <= HIST_END else "projected (trend)"
        return f"<b>Year {int(y)} · {kind}</b>"

    def ann(y):
        # bold dynamic label sitting just ABOVE the slider (bottom band)
        return dict(text=label_for(y), x=0.5, xanchor="center", xref="paper",
                    y=-0.015, yanchor="top", yref="paper", showarrow=False,
                    font=dict(size=18, color="#1c1c28"))

    frames = [
        go.Frame(name=str(y),
                 data=_frame_traces(ov_by[y], pt_by[y], focus, pts_cbar, ov_cbar, pts_color),
                 layout=go.Layout(annotations=[ann(y)]))
        for y in years
    ]
    fig = go.Figure(
        data=_frame_traces(ov_by[start], pt_by[start], focus, pts_cbar, ov_cbar, pts_color),
        frames=frames)

    play = dict(frame=dict(duration=650, redraw=True),
                transition=dict(duration=300), fromcurrent=True)
    fig.update_layout(
        map=dict(style="carto-positron", zoom=0.7, center=dict(lat=25, lon=10)),
        height=600, margin=dict(l=0, r=(95 if is_grad else 0), t=6, b=50),
        annotations=[ann(start)],   # bold year · observed/projected, above the slider
        hoverlabel=dict(bgcolor="white", font_size=13, bordercolor="#d6604d"),
        uirevision="keep", showlegend=False,
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            x=0.0, y=-0.08, xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, play]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ])],
        sliders=[dict(
            active=years.index(start), x=0.12, len=0.86, y=-0.08, pad=dict(t=2, b=2),
            currentvalue=dict(visible=False),
            steps=[dict(method="animate", label=str(int(y)),
                        args=[[str(y)], dict(mode="immediate",
                                             frame=dict(duration=0, redraw=True),
                                             transition=dict(duration=200))])
                   for y in years])],
    )
    return fig


def make_trend_chart(overlay_df):
    """Area-weighted GLOBAL-LAND mean soil temperature across the timeline:
    real past, plus both projection rates (30-yr and 50-yr) as a shaded range.
    Area-weighting (cos lat) keeps the many small polar grid cells from
    dominating, so the mean reflects land area, not cell count."""
    df = overlay_df.copy()
    df["w"] = np.cos(np.radians(df["lat"]))

    def wmean_by_year(tag):
        sub = df[df["rate"] == tag]
        g = sub.groupby("year")
        return (g.apply(lambda d: np.average(d["soil_temp"], weights=d["w"]))
                .sort_index())

    hist, r30, r50 = wmean_by_year("hist"), wmean_by_year("r30"), wmean_by_year("r50")

    # bridge each projection back to the last real point so the lines connect
    if len(hist):
        anchor = pd.Series({hist.index.max(): hist.iloc[-1]})
        r30 = pd.concat([anchor, r30]).sort_index()
        r50 = pd.concat([anchor, r50]).sort_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers",
                             name="real", line=dict(color="#2166ac", width=2)))
    # shade the band between the two projection rates (order-agnostic)
    fig.add_trace(go.Scatter(x=r50.index, y=r50.values, mode="lines",
                             name="projected · 50-yr trend",
                             line=dict(color="#f4a582", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=r30.index, y=r30.values, mode="lines",
                             name="projected · 30-yr trend",
                             line=dict(color="#b2182b", width=1.5, dash="dash"),
                             fill="tonexty", fillcolor="rgba(178,24,43,0.12)"))
    fig.add_vline(x=HIST_END + 2.5, line=dict(color="gray", width=1, dash="dot"))
    fig.add_annotation(x=HIST_END + 2.5, yref="paper", y=1.02, showarrow=False,
                       text="real ◂ ▸ projected", font=dict(size=10, color="gray"))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Global land mean annual soil temperatures (area-weighted)",
                   font=dict(size=13)),
        xaxis_title=None, yaxis_title="°C",
        legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
        showlegend=True,
    )
    return fig


RAW_INTRO = """
**dataset.csv:** This dataset is provided by **[Restor](https://restor.eco)**.
There are 2,015 rows with longitude, latitude, and **`AnnualTs`** with `AnnualTs` being
**multi-year averages of annual soil temperatures** (°C, roughly −9.6 to 34).

**Where this is going.** We bring in the **NCEP/NCAR Reanalysis** (NOAA PSL) —
real **annual soil temperatures** plus air temperature and precipitation
worldwide, 1948–2025 — and train a model on it to **predict soil temperatures
into the future**. Switch to **“Model & projection”** to see the real past, the
future projection, the focus overlays, and the year-by-year animation.
"""


def render_raw_dataset():
    """Just the raw observed data from dataset.csv — no model, no projection."""
    df = load_data(DATASETS["Soil temperature (AnnualTs)"])
    df["hover"] = df.apply(build_hover_text, axis=1)
    fig = go.Figure(go.Scattermap(
        lat=df["latitude"], lon=df["longitude"], mode="markers",
        text=df["hover"], hovertemplate="%{text}<extra></extra>",
        marker=dict(size=6, color=df[VALUE_COLUMN], colorscale=TS_SCALE,
                    cmin=TS_MIN, cmax=TS_MAX, opacity=1.0,
                    showscale=True, colorbar=dict(title="°C")),
    ))
    fig.update_layout(
        map=dict(style="carto-positron", zoom=0.7, center=dict(lat=25, lon=10)),
        height=600, margin=dict(l=0, r=0, t=0, b=0),
        hoverlabel=dict(bgcolor="white", font_size=13, bordercolor="#d6604d"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Observed soil temperature — {len(df):,} sites · "
               "multi-year average, no model or projection.")
    st.markdown(RAW_INTRO)
    with st.expander("Show dataset.csv"):
        st.dataframe(df[["latitude", "longitude", VALUE_COLUMN]], width="stretch")


# ===========================================================================
# 5. PAGE  (everything below renders top-to-bottom, left-to-right)
# ===========================================================================
st.set_page_config(page_title="Global Soil Temperatures", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; padding-bottom: 0.5rem; max-width: 100%; }
      #MainMenu, footer, header { visibility: hidden; }
      .app-title { font-size: 1.9rem; font-weight: 700; color: #1c1c28; margin-bottom: 0.6rem; }
      .app-tagline { font-size: 1.1rem; font-weight: 400; color: #6b6b76; margin-left: 0.5rem; }
      .app-sub   { font-size: 0.95rem; color: #6b6b76; margin-bottom: 1rem; }
      .section-label { font-size: 0.9rem; font-weight: 700; color: #1c1c28; margin: 0.3rem 0 0.1rem; }
      .tile-label { font-size: 1.0rem; font-weight: 700; color: #1c1c28; line-height: 1.15; }
      .tile-blurb { font-size: 0.85rem; color: #6b6b76; min-height: 2.4em; line-height: 1.25; }
      div[data-testid="stExpander"] { border: none; }
      div[data-testid="stImage"] img { border-radius: 6px; }
      /* focus tiles: larger label/blurb text, smaller buttons */
      div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0.1rem; }
      div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stButton"] button {
        padding: 0.0rem 0.3rem; font-size: 0.62rem; min-height: 0; line-height: 1.3; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title with the tagline inline beside it
st.markdown(
    '<div class="app-title">Global Soil Temperatures'
    '<span class="app-tagline">  —  how soil is warming, and what that means for '
    'permafrost, farmland, and where we can live</span></div>',
    unsafe_allow_html=True,
)

# Load model outputs once, and initialise the active focus
OVERLAY, POINTS, METRICS = load_outputs()
if "active_focus" not in st.session_state:
    st.session_state.active_focus = "all"

MODEL_VIEW = "Model & projection"
RAW_VIEW = "Raw dataset.csv only"
COLOR_ABS = "Soil temp"
COLOR_DELTA = "Change vs present"

# Default to the model/projection view (the animated story), not the static data.
view = st.segmented_control("View", [MODEL_VIEW, RAW_VIEW], default=MODEL_VIEW,
                            key="view_mode")

if view == RAW_VIEW:
    render_raw_dataset()
    st.stop()

# --- Map, then focus tiles, then explanation, then details (model view) ---
if OVERLAY is None or POINTS is None:
    st.warning("Run `../datavis/bin/python train_soil_temp.py` first to generate "
               "the prediction overlay and the per-year point data.")
    st.stop()

# How-to-read banner (so the map makes sense on arrival, no scrolling needed)
st.markdown(
    '<div class="app-sub"><b>How to read this:</b> the dots are the '
    '<b>sites from dataset.csv</b>; the shading is the soil-temperature field '
    'everywhere else — <b>real</b> annual soil temperatures up to ~2025, then the '
    '<b>model\'s prediction</b>. Press <b>▶ Play</b> to sweep 1950 → 2080. Pick a '
    '<b>focus</b> below to highlight permafrost, desertification, or habitability '
    'risk.</div>',
    unsafe_allow_html=True,
)

# Model-view controls (kept off the raw landing to reduce first-contact clutter)
ctrl_rate, ctrl_color = st.columns(2)
with ctrl_rate:
    rate = st.segmented_control(
        "Future warming rate", ["Recent 30-yr", "Long-term 50-yr"],
        default="Recent 30-yr", key="rate_mode",
        help="Which past trend we extend into the future — the recent 30-yr "
             "(1996–2025) or the longer 50-yr (1976–2025). Only affects projected "
             "years; the past is real either way. (See the note under the trend "
             "chart for what the two rates mean.)")
with ctrl_color:
    color = st.segmented_control(
        "Color", [COLOR_ABS, COLOR_DELTA], default=COLOR_ABS, key="color_mode",
        help="Color the dots by absolute soil temperature, or by how much they've "
             "warmed since ~2025 (Δ°C) — the change view makes the projected "
             "warming far easier to see than on the wide absolute scale.")

active = FOCUS_BY_KEY[st.session_state.active_focus]
rate_tag = "r50" if "50" in (rate or "") else "r30"
ov_r = OVERLAY[OVERLAY["rate"].isin(["hist", rate_tag])]
pt_r = POINTS[POINTS["rate"].isin(["hist", rate_tag])]
change_mode = color == COLOR_DELTA

st.plotly_chart(build_animation(active, ov_r, pt_r, change_mode), width="stretch")

# Focus tiles (set what the overlay shows) — directly under the map
st.markdown('<div class="section-label">Focus presets — choose the overlay</div>',
            unsafe_allow_html=True)
tile_cols = st.columns(len(FOCUSES))
for col, focus in zip(tile_cols, FOCUSES):
    with col:
        with st.container(border=True):
            st.markdown(f'<div class="tile-label">{focus["label"]}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="tile-blurb">{focus["blurb"]}</div>',
                        unsafe_allow_html=True)
            is_active = st.session_state.active_focus == focus["key"]
            st.button("Selected" if is_active else "Select",
                      key=f"focus_btn_{focus['key']}",
                      type="primary" if is_active else "secondary",
                      width="stretch", on_click=select_focus, args=(focus,))

# Active-focus explanation (text + images + sources), directly under the tiles
if active["explanation"]:
    text_col, img_col = st.columns([3, 2])
    with text_col:
        st.markdown(f"#### {active['label']}")
        st.markdown(active["explanation"])
        if active.get("sources"):
            st.markdown(f"**Sources:** {sources_md(active['sources'])}")
    with img_col:
        if active["key"] == "all":
            # area-weighted global-land mean across ALL grid cells (not just dots),
            # real past + both projection rates as a shaded range
            st.plotly_chart(make_trend_chart(OVERLAY), width="stretch")
            st.caption(
                "The shaded band spans two trend fits — the recent **30-yr** "
                "(1996–2025) and the longer **50-yr** (1976–2025); which is "
                "steeper depends on the location. Both are **straight-line "
                "extrapolations** that ignore the acceleration and compounding "
                "feedbacks of climate change (melting ice, permafrost carbon, "
                "weaker ocean uptake, fires), so the real future likely runs "
                "hotter — treat these projections as a **floor, not a forecast**.")
        for path, caption in active["detail_images"]:
            st.image(path, caption=caption, width="stretch")

# Method recap + model skill (always shown; sources live in the All-sites panel)
with st.expander("How the projection is made and model details", expanded=True):
    st.markdown(PROJECTION_INFO)
    if METRICS.get("soil_temp_cv_r2_with_airtemp") is not None:
        line = (
            f"**Model skill (held-out-year CV):** soil-temp R² "
            f"**{METRICS.get('soil_temp_cv_r2_with_airtemp')}** "
            f"(RMSE {METRICS.get('soil_temp_cv_rmse_with_airtemp')} °C); "
            f"without the air-temp features it falls to R² "
            f"**{METRICS.get('soil_temp_cv_r2_without_airtemp')}** "
            f"(RMSE {METRICS.get('soil_temp_cv_rmse_without_airtemp')} °C) — air "
            f"temperature is the dominant driver."
        )
        n = METRICS.get("n_train_rows")
        if n:
            line += f" Trained on **{n:,} cell-years** ({METRICS.get('train_years')})."
        st.markdown(line)
