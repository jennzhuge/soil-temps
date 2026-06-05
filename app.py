"""
Interactive global soil-temperature map.

Run with:
    ../datavis/bin/streamlit run app.py

Built with Streamlit + Plotly so it's easy to customize:
  - Add more datasets:  add an entry to the DATASETS dict.
  - Add a new FOCUS:     add a dict to the FOCUSES list (see section 3).
  - Richer hover popups:  edit build_hover_text().
  - Colors / styling:     tweak the px.scatter_map(...) call.
"""

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


# ---------------------------------------------------------------------------
# 3. FOCUSES  (preset filters shown as tiles)
# ===========================================================================
# This is the part to extend. Each focus is one dict:
#
#   {
#     "key":   unique id string,
#     "label": title shown on the tile,
#     "blurb": one-line description on the tile,
#     "low":   lower temperature bound in °C (None = no lower bound),
#     "high":  upper temperature bound in °C (None = no upper bound),
#     "thumb": small image path for the tile (None = no image),
#     "explanation": markdown shown when the focus is active,
#     "detail_images": list of (image_path, caption) shown when active,
#     "sources": list of (title, url) shown as links under the explanation,
#   }
#
# To add a new focus, copy a block and change the values. That's it.
# ---------------------------------------------------------------------------

PERMAFROST_EXPLANATION = """
**Permafrost** is ground that stays at or below 0 °C for at least two
consecutive years. A **mean annual soil temperature below 0 °C is the standard
indicator** that permafrost exists *at depth* — but it does **not** mean the
ground is frozen year-round at the surface. The top **"active layer" still
thaws every summer**, even where permafrost is present below; the annual *mean*
stays sub-zero because the long, cold winter outweighs the short summer thaw, so
heat never penetrates deep enough to thaw the layer beneath. 

The areas shown in blue are locations where the model predicts an annual mean soil 
temperature below 0 °C in the selected year. Treat this filter as **permafrost-favourable**, 
not confirmed: the reading is a proxy that depends on measurement depth, snow and 
vegetation cover, and soil type, and a single annual mean can't verify the two-year 
part of the definition.

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
Two well-established limits set the hottest farmable edge:

- **Human climate niche** — people and crops cluster around an 11–15 °C mean
  annual temperature; land above **~29 °C** is near-unliveable (only ~0.8 % of
  land today, mostly the Sahara; Xu et al. 2020).
- **Crop physiology** — most crops want soil at **20–30 °C**; above ~35–40 °C
  roots are damaged and many staples stop developing.

**How this filter works.** Temperature alone doesn't decide farmability — water
does too — so the focus combines the two into a single **heat-and-aridity score**
(0 = unfarmable, 1 = fine):

- a **heat term** near 1 in mild climates that falls toward 0 as temperature
  pushes past the ~29 °C niche edge, *multiplied by*
- an **aridity term** from the aridity index (rainfall ÷ potential
  evapotranspiration): near 1 in humid climates, near 0 in deserts; land counts
  as "dryland" below 0.65 (UNCCD).

**Why aridity matters:** crops are mostly water and need a steady supply through
the growing season. The aridity index asks whether rainfall keeps up with how
fast the atmosphere pulls moisture back out (evaporation + transpiration). When
that demand outpaces the rain, soil dries faster than it's replenished, so
plants wilt, yields fall, and without irrigation the land can't reliably grow
food — even if the temperature seems fine.

The final score is the **heat term multiplied by the aridity term**, so it stays
high only when *both* are — and collapses if either one is close to 0. In other words, a
place fails when it's **too hot, too dry, or both**. The two are also linked:
warmer air evaporates more water, so rising temperature drags the aridity term
down as well — the compounding engine of **desertification**.

The areas shaded **red** are where that combined score falls to near zero **in
the selected year**. As warming pushes more land past the point, the red region
**grows** toward 2080. Read it as "heat-stress / desertification risk," not a
hard line — the real limit also shifts with rainfall, irrigation, and soil.
"""

HABITABILITY_EXPLANATION = """
**The habitability index (0–1).** Each location gets three sub-scores, every one
running from **0 (unsuitable)** to **1 (ideal)**, which we then multiply together:

- **Warmth** — scores **0 when the soil is below freezing** (nothing grows in
  frozen ground) and rises to **1** at mild temperatures.
- **Not too hot** — scores **1** at mild temperatures and drops to **0 above
  ~29 °C**, the edge of the human climate niche (Xu et al. 2020).
- **Moisture** — from the aridity index (rainfall ÷ evaporative demand): near
  **0** in hyper-arid deserts, near **1** in humid climates (land counts as
  "dryland" below 0.65; UNCCD).

Because the three are **multiplied**, a place needs *all three* to score well —
warm enough, not too hot, **and** watered. If any one is near 0, the whole index
is near 0. On the map, the **green shading is the model's habitability index for
the selected year** — deeper green = more habitable — fading to nothing where
land is too cold, too hot, or too dry. Press Play to watch it shift **poleward**
as the climate warms: gaining ground in the high north while losing it across the
subtropics.

It's a deliberately simple, transparent index (not a crop model). Checked against
real ground truth, known farmland regions average **0.92** and barren deserts/ice
**0.11**.
"""

FOCUSES = [
    {
        "key": "all",
        "label": "All sites",
        "blurb": "Observed sites + about the data.",
        "low": None,
        "high": None,
        "thumb": None,
        # no shaded overlay — show the observed dots alone
        "overlay": {"kind": "none"},
        "explanation": "",
        "detail_images": [],
        "sources": [],
    },
    {
        "key": "permafrost",
        "label": "Permafrost-favourable",
        "blurb": "Blue: soil below 0 °C. Shrinks over time.",
        "low": None,
        "high": 0.0,
        # overlay: blue shaded fill where predicted soil temp < 0 (shrinks as it warms)
        "overlay": {"kind": "threshold", "col": "soil_temp", "thr": 0.0,
                    "fill": "33,102,172", "name": "soil temp < 0 °C"},
        "thumb": "images/permafrost_thaw_slump.jpg",
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
        "label": "Cropland heat limit",
        "blurb": "Red: too hot/arid to farm. Grows over time.",
        "low": 30.0,
        "high": None,
        # overlay: red shaded fill where heat+aridity suitability < 0.2 (grows as it warms)
        "overlay": {"kind": "threshold", "col": "heat_aridity", "thr": 0.2,
                    "fill": "178,24,43", "name": "heat/aridity-limited"},
        "thumb": "images/cracked_dry_soil.jpg",
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
        "blurb": "Green: more habitable land.",
        "low": None,
        "high": None,
        # overlay: green shaded fill graded by the habitability index (0-1)
        "overlay": {"kind": "gradient", "col": "habitability", "fill": "0,104,55",
                    "name": "habitability"},
        "thumb": None,
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
# 4. PAGE SETUP & STYLING
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Global Soil Temperature", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; padding-bottom: 0.5rem; max-width: 100%; }
      #MainMenu, footer, header { visibility: hidden; }
      .app-title { font-size: 1.9rem; font-weight: 700; color: #1c1c28; margin-bottom: 0.1rem; }
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

st.markdown('<div class="app-title">Global Soil Temperature</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Real observed sites (the dots) over a modelled '
    'prediction overlay. Press ▶ Play to watch 1970 → 2080: dots and overlay '
    'recolour each year. Past years use real climate; the future is a trend '
    'projection.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 5. MODEL OUTPUTS (produced by train_soil_temp.py)
# ---------------------------------------------------------------------------
OVERLAY_PATH = "outputs/overlay_grid.parquet"
POINTS_PATH = "outputs/points_timeseries.parquet"
METRICS_PATH = "outputs/metrics.json"

TS_SCALE, TS_MIN, TS_MAX = "RdYlBu_r", -10, 35   # soil-temp colour scale
HIST_END = 2017                                  # last year of real climate

PROJECTION_INFO = """
The coloured **overlay and the dot colours are model output**, not measurements
(the only measured quantity is *where* the dots are — the 2,015 observed sites).

**The model.** A gradient-boosted regression-tree model
(scikit-learn `HistGradientBoostingRegressor`: 400 trees, depth 4, learning rate
0.08) that maps **climate → annual soil temperature**.

**How it was trained.**
- **Target:** the measured `AnnualTs` at the observed sites (the ~1,947 of 2,015
  that fall on land cells with climate coverage).
- **Inputs:** the 7 climate features listed in *All sites → about the data*,
  sampled from each site's 1988–2017 climate baseline.
- **Validation:** 5-fold cross-validation → **R² ≈ 0.90, RMSE ≈ 2 °C**.
- **Applied** cell-by-cell to the 0.5° land grid for every year to make the maps.

⚠️ **Don't over-read that R².** Soil temperature is physically just **air
temperature plus a small offset**, so *any* climate-based model predicts it well
— a high score here is **expected physics, not a surprising result**, and the
model leans heavily on air temperature by design. (Retraining without the direct
air-temperature columns still scores ≈ 0.89, because the remaining features —
temperature seasonality and PET — also encode temperature; you can't predict soil
temperature without temperature.) The model's real value in this app is
**spatial gap-filling** — turning 2,015 scattered measurements into a continuous
global grid — and feeding the **habitability index**, not the headline accuracy.

**The habitability** axis combines a heat term (the human-climate-niche limit,
~29 °C) with an aridity term (the UNCCD dryland threshold).

**Past vs future — and what actually goes into the model.** The model never sees
coordinates; it sees *climate*. For years up to %d we feed it the **actual
historical climate** of that year at each location (so the maps wiggle — real
interannual variability). For later years we **extrapolate each location's own
climate forward** (its recent warming rate raises temperature, precipitation
follows its trend, etc.) and run the **same model** on that projected climate.
The projection has **no physics** — it can't capture acceleration or uneven
(Arctic-amplified) warming the way scenario models (CMIP6) would, and it assumes
today's climate→soil relationship still holds.

**The two warming rates (the toggle).** We measure each location's warming rate
two ways and let you switch between them:
- **Recent 30-yr (1988–2017)** — the *headline*. Warming has been faster lately,
  so this trend is steeper.
- **Conservative 50-yr (1968–2017)** — a *lower bound*. Averaging over the longer,
  slower period gives a gentler trend.

**Both are almost certainly under-estimates.** A straight-line trend assumes
warming continues at a *constant* rate, but climate change is **compounding and
accelerating**: global emissions are still rising, and self-reinforcing feedbacks
pile extra warming on top — melting ice exposes darker ocean and land that absorb
more heat (ice–albedo), thawing permafrost releases CO₂ and methane, warmer
oceans absorb less CO₂, and drier, hotter forests burn and release stored carbon.
So the real future likely lands **at or above even the 30-yr line** — treat these
projections as a floor, not a ceiling.
""" % HIST_END

PROJECTION_SOURCES = [
    ("Climate data: UDel air temp & precip (NOAA PSL)",
     "https://psl.noaa.gov/data/gridded/data.UDel_AirT_Precip.html"),
    ("Human climate niche — Xu et al. 2020 (PNAS)",
     "https://www.pnas.org/doi/10.1073/pnas.1910114117"),
    ("Aridity index / drylands — UNCCD",
     "https://www.unccd.int/sites/default/files/2024-12/aridity_report.pdf"),
]

DATA_INFO = """
**The starting point — `dataset.csv` (the dots).**
2,015 global land locations, each with a longitude, latitude, and **`AnnualTs`** =
mean annual **soil** temperature in °C (roughly −9.6 to 34 °C). These coordinates
are the only *measured* data on the map and are always shown as dots.

**Climate data we pulled in — UDel / NOAA PSL.**
Gridded monthly **air temperature** and **precipitation**, 0.5° resolution,
1900–2017, land only. We use it three ways: (1) train the model that turns
climate into soil temperature, (2) measure each location's recent **warming
rate**, and (3) reconstruct the **actual year-by-year climate** for the historical
part of the animation.

**The 7 features the model uses** — all derived per location from that UDel
climate:

1. **Mean annual air temperature** (°C)
2. **Annual precipitation** (mm)
3. **Temperature seasonality** — standard deviation of the 12 monthly temperatures
4. **Warmest-month mean temperature** (°C)
5. **Coldest-month mean temperature** (°C)
6. **Potential evapotranspiration (PET)** — the atmosphere's "drying demand" (how
   much water could evaporate + transpire if available); estimated with the
   classic **Thornthwaite** method from monthly temperature and daylength
7. **Aridity index** — precipitation ÷ PET (the water side of habitability)

**Mean annual air temperature is the dominant predictor** — unsurprising, since
soil temperature is essentially air temperature plus a small offset, so the model
is more a re-expression of the climate than a surprising discovery. See *How the
prediction is made* below for training detail and the caveat about reading the R².

**Filling the map beyond the 2,015 sites.** The dots are the only measurements —
far too sparse to colour a whole globe. So we let the model do **spatial
gap-filling**: having learned the climate→soil-temperature relationship from the
measured sites, it is then run on the **UDel climate at every ~0.5° land cell
worldwide**, turning 2,015 scattered points into a continuous global surface.
That predicted surface is what the coloured **overlay** shows (and what the
habitability / heat-aridity scores are computed from). The animation's overlay is
thinned to ~1.5° so it stays smooth in the browser; the observed dots are always
drawn on top, unchanged.

**Concept thresholds behind the focuses.**
The ~29 °C habitability ceiling comes from the **human climate niche** (Xu et al.
2020); the dryland aridity cut-off (0.65) from the **UNCCD**; the 0 °C permafrost
indicator from standard permafrost references.
"""

DATA_SOURCES = [
    ("UDel air temp & precip (NOAA PSL)",
     "https://psl.noaa.gov/data/gridded/data.UDel_AirT_Precip.html"),
    ("Thornthwaite PET method — Wikipedia",
     "https://en.wikipedia.org/wiki/Potential_evaporation"),
    ("Human climate niche — Xu et al. 2020 (PNAS)",
     "https://www.pnas.org/doi/10.1073/pnas.1910114117"),
    ("Aridity / drylands — UNCCD",
     "https://www.unccd.int/sites/default/files/2024-12/aridity_report.pdf"),
    ("Permafrost — NSIDC",
     "https://nsidc.org/learn/parts-cryosphere/frozen-ground-permafrost"),
]

# The "All sites" tile explains the underlying data (like the other focus tiles
# explain their topic). Mutating the dict updates FOCUS_BY_KEY too (same object).
FOCUS_BY_KEY["all"]["explanation"] = DATA_INFO
FOCUS_BY_KEY["all"]["sources"] = DATA_SOURCES


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


OVERLAY, POINTS, METRICS = load_outputs()

if "active_focus" not in st.session_state:
    st.session_state.active_focus = "all"


def select_focus(focus):
    st.session_state.active_focus = focus["key"]


def _overlay_trace(dfy, focus, ov_cbar=None):
    """A transparent shaded FILL (density heatmap) for the focus, or None."""
    ov = focus["overlay"]
    rgb = ov.get("fill")
    if ov["kind"] == "threshold":
        # binary region where a column is below a threshold (uniform fill)
        sel = dfy[dfy[ov["col"]] < ov["thr"]]
        return go.Densitymap(
            lat=sel["lat"], lon=sel["lon"], z=[1] * len(sel),
            radius=18, zmin=0, zmax=1, showscale=False, opacity=0.55,
            colorscale=[[0, f"rgba({rgb},0)"], [1, f"rgba({rgb},1)"]],
            hoverinfo="skip", name=ov["name"],
        )
    if ov["kind"] == "gradient":
        # continuous fill graded by a value column (fades out at low values)
        return go.Densitymap(
            lat=dfy["lat"], lon=dfy["lon"], z=dfy[ov["col"]],
            radius=14, zmin=0, zmax=ov.get("zmax", 1.0), opacity=0.6,
            colorscale=[[0, f"rgba({rgb},0)"], [1, f"rgba({rgb},1)"]],
            showscale=ov_cbar is not None, colorbar=ov_cbar,
            hoverinfo="skip", name=ov["name"],
        )
    return None


def _points_trace(dfy, pts_cbar):
    """Observed-site dots — the only markers — coloured by soil temp for the year."""
    return go.Scattermap(
        lat=dfy["lat"], lon=dfy["lon"], mode="markers", name="observed sites",
        text=dfy["hover"], hovertemplate="%{text}<extra></extra>",
        marker=dict(size=6, color=dfy["soil_temp"], colorscale=TS_SCALE,
                    cmin=TS_MIN, cmax=TS_MAX, opacity=1.0,
                    showscale=True, colorbar=pts_cbar),
    )


def _frame_traces(ov_df, pt_df, focus, pts_cbar, ov_cbar):
    """Traces for one year: shaded fill (if any) below the observed dots."""
    fill = _overlay_trace(ov_df, focus, ov_cbar)
    return ([fill] if fill is not None else []) + [_points_trace(pt_df, pts_cbar)]


def build_animation(focus, overlay_df, points_df):
    """go.Figure with one frame per year: shaded fill + observed dots."""
    years = sorted(overlay_df["year"].unique())
    start = 2020 if 2020 in years else years[0]
    ov_by = {y: overlay_df[overlay_df["year"] == y] for y in years}
    pt_by = {y: points_df[points_df["year"] == y] for y in years}

    # Legends: the dots always carry the °C scale. A gradient overlay (habitability)
    # adds a second 0-1 colour bar placed SIDE BY SIDE with the °C bar.
    is_grad = focus["overlay"]["kind"] == "gradient"
    if is_grad:
        pts_cbar = dict(title="°C", x=1.01, xanchor="left", y=0.5, yanchor="middle",
                        len=0.92)
        ov_cbar = dict(title="Habitability", x=1.12, xanchor="left", y=0.5,
                       yanchor="middle", len=0.92,
                       tickmode="array", tickvals=[0, 0.25, 0.5, 0.75, 1.0])
    else:
        pts_cbar, ov_cbar = dict(title="°C"), None

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
                 data=_frame_traces(ov_by[y], pt_by[y], focus, pts_cbar, ov_cbar),
                 layout=go.Layout(annotations=[ann(y)]))
        for y in years
    ]
    fig = go.Figure(
        data=_frame_traces(ov_by[start], pt_by[start], focus, pts_cbar, ov_cbar),
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


def make_trend_chart(points_df):
    """Mean soil temp across the timeline: observed, plus BOTH projection rates
    (30-yr and 50-yr) as a shaded range so the future uncertainty is visible."""
    def mean_by_year(tag):
        sub = points_df[points_df["rate"] == tag]
        return sub.groupby("year")["soil_temp"].mean().sort_index()

    hist, r30, r50 = mean_by_year("hist"), mean_by_year("r30"), mean_by_year("r50")

    # bridge each projection back to the last observed point so lines connect
    if len(hist):
        anchor = pd.Series({hist.index.max(): hist.iloc[-1]})
        r30 = pd.concat([anchor, r30]).sort_index()
        r50 = pd.concat([anchor, r50]).sort_index()

    fig = go.Figure()
    # observed (solid blue)
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers",
                             name="observed", line=dict(color="#2166ac", width=2)))
    # shaded band between the two rates: lower (50-yr) first, then upper (30-yr) fills to it
    fig.add_trace(go.Scatter(x=r50.index, y=r50.values, mode="lines",
                             name="projected (50-yr, low)",
                             line=dict(color="#f4a582", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=r30.index, y=r30.values, mode="lines",
                             name="projected (30-yr, high)",
                             line=dict(color="#b2182b", width=1.5, dash="dash"),
                             fill="tonexty", fillcolor="rgba(178,24,43,0.15)"))
    fig.add_vline(x=HIST_END + 0.5, line=dict(color="gray", width=1, dash="dot"))
    fig.add_annotation(x=HIST_END + 0.5, yref="paper", y=1.02, showarrow=False,
                       text="real ◂ ▸ projected", font=dict(size=10, color="gray"))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Mean soil temp at observed sites", font=dict(size=13)),
        xaxis_title=None, yaxis_title="°C",
        legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
        showlegend=True,
    )
    return fig


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
    st.caption(f"Raw observed data from dataset.csv — {len(df):,} sites · "
               "annual soil temperature, no model or projection.")
    with st.expander("Show data table"):
        st.dataframe(df[["latitude", "longitude", VALUE_COLUMN]], width="stretch")


# ---------------------------------------------------------------------------
# 6. CONTROLS ROW: view switch + future warming rate
# ---------------------------------------------------------------------------
MODEL_VIEW = "Model & projection"
RAW_VIEW = "Raw dataset.csv only"
ctrl_view, ctrl_rate = st.columns([3, 2])
with ctrl_view:
    view = st.segmented_control("View", [MODEL_VIEW, RAW_VIEW], default=MODEL_VIEW,
                                key="view_mode")
with ctrl_rate:
    rate = st.radio("Future warming rate", ["Recent 30-yr", "Conservative 50-yr"],
                    horizontal=True, key="rate_mode",
                    help="How fast each place keeps warming after 2017. Recent "
                         "30-yr (steeper) is the headline; 50-yr is a gentler "
                         "lower bound. Both are likely under-estimates — warming "
                         "is accelerating via compounding feedbacks, so treat them "
                         "as a floor. Past years are real climate either way. "
                         "(Ignored in raw-data view.)")

if view == RAW_VIEW:
    render_raw_dataset()
    st.stop()

# ---------------------------------------------------------------------------
# 7. THE UNIFIED MAP + FOCUS TILES (model view)
# ---------------------------------------------------------------------------
if OVERLAY is None or POINTS is None:
    st.warning("Run `../datavis/bin/python train_soil_temp.py` first to generate "
               "the prediction overlay and the per-year point data.")
    st.stop()

active = FOCUS_BY_KEY[st.session_state.active_focus]
rate_tag = "r30" if rate.startswith("Recent") else "r50"
ov_r = OVERLAY[OVERLAY["rate"].isin(["hist", rate_tag])]
pt_r = POINTS[POINTS["rate"].isin(["hist", rate_tag])]

st.plotly_chart(build_animation(active, ov_r, pt_r), width="stretch")

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
            links = " · ".join(f"[{t}]({u})" for t, u in active["sources"])
            st.markdown(f"**Sources:** {links}")
    with img_col:
        if active["key"] == "all":
            # the dataset panel gets the soil-temp trend chart: observed + BOTH
            # projection rates as a shaded range (so pass all rates, not pt_r)
            st.plotly_chart(make_trend_chart(POINTS), width="stretch")
        for path, caption in active["detail_images"]:
            st.image(path, caption=caption, width="stretch")

# Method + model skill + sources (always shown)
with st.expander("How the prediction is made (and how good it is)"):
    st.markdown(PROJECTION_INFO)
    if METRICS:
        st.markdown(
            f"**Model skill (5-fold CV):** soil-temp R² "
            f"**{METRICS.get('soil_temp_cv_r2_with_airtemp')}** "
            f"(RMSE {METRICS.get('soil_temp_cv_rmse_with_airtemp')} °C); "
            f"still R² **{METRICS.get('soil_temp_cv_r2_without_airtemp')}** "
            f"without the air-temp feature. "
            f"Habitability check — farmland "
            f"**{METRICS.get('habitability_validation_mean_farmland')}** vs barren "
            f"**{METRICS.get('habitability_validation_mean_barren')}** "
            f"(separation +{METRICS.get('habitability_validation_separation')})."
        )
    links = " · ".join(f"[{t}]({u})" for t, u in PROJECTION_SOURCES)
    st.markdown(f"**Sources:** {links}")
