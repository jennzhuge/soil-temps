"""
Soil-temperature & habitability pipeline with future projection.

WHAT THIS DOES
  1. Loads 50 years (1968-2017) of monthly air-temperature and precipitation
     (UDel, 0.5 deg, land only; cached in data/).
  2. Derives bioclim-style annual features per year, including a Thornthwaite
     potential-evapotranspiration (PET) estimate and an aridity index AI = P/PET.
  3. Computes a present-day baseline (mean of 1988-2017) and a per-pixel warming
     *rate* (linear OLS slope) over two windows:
         - recent 30 yr (1988-2017)  -> headline (faster, recent rate)
         - full   50 yr (1968-2017)  -> conservative low estimate
     and projects each feature to ~2070 (trend extrapolation, see CAVEATS).
  4. Trains a gradient-boosting model to predict observed annual soil temperature
     (dataset.csv) from the climate features. Reports skill WITH and WITHOUT the
     air-temperature feature (the air<->soil temp link is near-circular).
  5. Builds a transparent heat + aridity HABITABILITY index (0-1) and validates
     it against known agricultural vs barren reference regions.
  6. Predicts soil temp & habitability on the global land grid for the present
     and the ~2070 projection, and writes outputs/ (parquet + csv + PNG maps).

CAVEATS (also surfaced in the app):
  - Trend extrapolation assumes the recent warming RATE simply continues. It has
    no physics; it cannot capture acceleration or uneven (Arctic-amplified)
    warming the way scenario models (CMIP6) would. The 30yr/50yr pair is shown
    as a low-high range rather than a single line.
  - The model learned today is assumed to still hold in the future (stationarity).
  - UDel is land-only at 0.5 deg; the soil-temp obs year/depth is unknown.

Run:  ../datavis/bin/python train_soil_temp.py
"""

import json
import numpy as np
import pandas as pd
import xarray as xr

from soil_lib import FEATURE_COLS, habitability, heat_aridity_suit  # shared with app.py

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AIR_NC = "data/air.mon.mean.v501.nc"
PRECIP_NC = "data/precip.mon.total.v501.nc"
OBS_CSV = "dataset.csv"
OUT_DIR = "outputs"

YEAR_START, YEAR_END = 1968, 2017      # 50-year window
BASELINE_START = 1988                  # recent 30-yr baseline (present)
FUTURE_YEAR = 2070
BASELINE_MID = (BASELINE_START + YEAR_END) / 2.0   # ~2002.5
DELTA_YEARS = FUTURE_YEAR - BASELINE_MID            # ~67.5

DAYS_IN_MONTH = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MONTH_MID_DOY = np.array([15, 45, 74, 105, 135, 162, 198, 228, 258, 288, 318, 344])


# ---------------------------------------------------------------------------
# 1. Load monthly climate (sliced & in-memory)
# ---------------------------------------------------------------------------
def load_monthly():
    air = xr.open_dataset(AIR_NC)["air"].sel(
        time=slice(f"{YEAR_START}", f"{YEAR_END}")).load()          # degC
    pre = xr.open_dataset(PRECIP_NC)["precip"].sel(
        time=slice(f"{YEAR_START}", f"{YEAR_END}")).load() * 10.0    # cm -> mm
    lat = air["lat"].values
    lon = air["lon"].values
    nyear = YEAR_END - YEAR_START + 1
    # (nyear, 12, lat, lon)
    T = air.values.reshape(nyear, 12, lat.size, lon.size)
    P = pre.values.reshape(nyear, 12, lat.size, lon.size)
    return T, P, lat, lon


# ---------------------------------------------------------------------------
# 2. Thornthwaite PET (mm/yr), vectorised over (lat, lon) for one year
# ---------------------------------------------------------------------------
def daylight_hours(lat):
    """Mean daylight hours per month -> (12, nlat)."""
    lat_rad = np.radians(lat)[None, :]                       # (1, nlat)
    decl = 0.409 * np.sin(2 * np.pi * MONTH_MID_DOY / 365.0 - 1.39)[:, None]
    x = np.clip(-np.tan(lat_rad) * np.tan(decl), -1, 1)
    ws = np.arccos(x)                                        # sunset hour angle
    return 24.0 / np.pi * ws                                 # (12, nlat)


def pet_year(Tm, daylen):
    """
    Tm: (12, nlat, nlon) monthly mean temp. daylen: (12, nlat).
    Returns annual PET (nlat, nlon) in mm.
    """
    Tpos = np.clip(Tm, 0, None)
    heat_index = np.sum((Tpos / 5.0) ** 1.514, axis=0)       # (nlat, nlon)
    I = heat_index
    a = (6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239)
    Isafe = np.where(I <= 0, np.nan, I)

    daylen3 = daylen[:, :, None]                             # (12, nlat, 1)
    corr = (daylen3 / 12.0) * (DAYS_IN_MONTH[:, None, None] / 30.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        unadj = 16.0 * (10.0 * Tm / Isafe[None]) ** a[None]  # standard 0<T<26.5
    hot = -415.85 + 32.24 * Tm - 0.43 * Tm**2                # T >= 26.5 plateau
    monthly = np.where(Tm <= 0, 0.0, np.where(Tm < 26.5, unadj, hot))
    monthly = np.where(np.isfinite(monthly), monthly, 0.0)
    pet = np.sum(monthly * corr, axis=0)                     # (nlat, nlon)
    pet = np.where(I <= 0, 0.0, pet)
    return pet


# ---------------------------------------------------------------------------
# 3. Build per-year feature stack -> dict of (nyear, nlat, nlon)
# ---------------------------------------------------------------------------
def build_features(T, P, lat):
    nyear = T.shape[0]
    daylen = daylight_hours(lat)
    feats = {k: np.full((nyear, lat.size, T.shape[3]), np.nan, dtype="float32")
             for k in ["tmean", "pann", "tseason", "twarm", "tcold", "pet", "ai"]}
    for y in range(nyear):
        Tm, Pm = T[y], P[y]                                  # (12, nlat, nlon)
        feats["tmean"][y] = Tm.mean(0)
        feats["pann"][y] = Pm.sum(0)
        feats["tseason"][y] = Tm.std(0)
        feats["twarm"][y] = Tm.max(0)
        feats["tcold"][y] = Tm.min(0)
        pet = pet_year(Tm, daylen)
        feats["pet"][y] = pet
        with np.errstate(invalid="ignore", divide="ignore"):
            ai = Pm.sum(0) / np.where(pet > 0, pet, np.nan)
        feats["ai"][y] = np.clip(ai, 0, 3.0)
    return feats


def ols_slope(stack, years):
    """Per-pixel linear slope (units/yr). stack (nyear, nlat, nlon)."""
    ny, nlat, nlon = stack.shape
    Y = stack.reshape(ny, -1)                                # (nyear, npix)
    x = years.astype("float64")
    valid = np.isfinite(Y).all(axis=0)
    slope = np.full(Y.shape[1], np.nan)
    if valid.any():
        coef = np.polyfit(x, Y[:, valid].astype("float64"), 1)  # [slope, intercept]
        slope[valid] = coef[0]
    return slope.reshape(nlat, nlon)


# ---------------------------------------------------------------------------
# 4. Reference-region validation (real-world ground truth, no big download)
#    (the habitability index itself lives in soil_lib.py, shared with the app)
# ---------------------------------------------------------------------------
# (lat, lon, label, expect_high)  -- expect_high True = productive farmland
REFERENCE_REGIONS = [
    (42.0, -93.0, "Iowa Corn Belt (US)", True),
    (49.0, 32.0, "Ukraine wheat belt", True),
    (52.0, 0.5, "England (UK)", True),
    (-34.0, -60.0, "Argentine Pampas", True),
    (28.0, 77.0, "Indo-Gangetic Plain", True),
    (45.0, 5.0, "France (Rhone)", True),
    (23.0, 13.0, "Central Sahara", False),
    (20.0, 45.0, "Rub al Khali (Arabia)", False),
    (42.0, 105.0, "Gobi Desert", False),
    (-24.0, 124.0, "Australian interior", False),
    (76.0, -42.0, "Greenland interior", False),
    (-23.0, -68.0, "Atacama Desert", False),
]


def sample_grid(field, lat, lon, plat, plon):
    """Nearest-cell sample of a (nlat, nlon) field at point arrays."""
    plon360 = np.mod(plon, 360.0)
    ilat = np.clip(np.round((plat - lat[0]) / (lat[1] - lat[0])).astype(int), 0, lat.size - 1)
    ilon = np.clip(np.round((plon360 - lon[0]) / (lon[1] - lon[0])).astype(int), 0, lon.size - 1)
    return field[ilat, ilon]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading monthly climate (1968-2017)...")
    T, P, lat, lon = load_monthly()
    years = np.arange(YEAR_START, YEAR_END + 1)

    print("Building annual features + Thornthwaite PET...")
    feats = build_features(T, P, lat)

    # Present baseline = mean over recent 30 yr (1988-2017). Ocean cells are all
    # NaN -> the empty-slice warning is expected and harmless, so silence it.
    import warnings
    bmask = years >= BASELINE_START
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        base = {k: np.nanmean(v[bmask], axis=0) for k, v in feats.items()}

    # Warming/feature rates (per yr): recent 30yr (headline) and full 50yr (low)
    print("Fitting per-pixel trends (30 yr headline, 50 yr conservative)...")
    slope30 = {k: ols_slope(feats[k][bmask], years[bmask]) for k in feats}
    slope50 = {k: ols_slope(feats[k], years) for k in feats}

    def project(slope):
        fut = {k: base[k] + slope[k] * DELTA_YEARS for k in base}
        # AI must stay physical
        fut["ai"] = np.clip(fut["ai"], 0, 3.0)
        fut["pann"] = np.clip(fut["pann"], 0, None)
        return fut
    fut30, fut50 = project(slope30), project(slope50)

    # ---- Train soil-temp model on observed points ----
    print("Training soil-temperature model...")
    obs = pd.read_csv(OBS_CSV)
    obs["AnnualTs"] = pd.to_numeric(obs["AnnualTs"], errors="coerce")
    obs = obs.dropna(subset=["longitude", "latitude", "AnnualTs"]).reset_index(drop=True)
    Xobs = {k: sample_grid(base[k], lat, lon, obs["latitude"].values,
                           obs["longitude"].values) for k in FEATURE_COLS}
    Xobs = pd.DataFrame(Xobs)
    y = obs["AnnualTs"].to_numpy(dtype=float)
    keep = np.asarray(Xobs.notna().all(axis=1))
    Xobs, y = Xobs[keep], y[keep]
    print(f"  {len(y)} of {len(obs)} obs points have valid climate features.")

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import r2_score, mean_squared_error

    def evaluate(cols, tag):
        model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.08,
                                              max_iter=400, random_state=0)
        cv = KFold(5, shuffle=True, random_state=0)
        pred = cross_val_predict(model, Xobs[cols], y, cv=cv)
        r2 = r2_score(y, pred)
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        print(f"  [{tag}] 5-fold CV  R2={r2:.3f}  RMSE={rmse:.2f} C")
        return r2, rmse

    r2_full, rmse_full = evaluate(FEATURE_COLS, "with air temp")
    cols_noair = [c for c in FEATURE_COLS if c not in ("tmean", "twarm", "tcold")]
    r2_noair, rmse_noair = evaluate(cols_noair, "without air temp")

    # Final model on all data, applied to the grid (fit on plain arrays so grid
    # prediction with numpy doesn't warn about missing feature names)
    Xobs_arr = Xobs[FEATURE_COLS].to_numpy()
    final = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.08,
                                          max_iter=400, random_state=0).fit(Xobs_arr, y)

    def predict_grid(fdict):
        stack = np.stack([fdict[k].ravel() for k in FEATURE_COLS], axis=1)
        ok = np.isfinite(stack).all(axis=1)
        out = np.full(stack.shape[0], np.nan, dtype="float32")
        out[ok] = final.predict(stack[ok])
        return out.reshape(base["tmean"].shape)

    print("Predicting soil temperature (present, future)...")
    st_present = predict_grid(base)
    st_future = predict_grid(fut30)
    st_future_low = predict_grid(fut50)

    # ---- Habitability (present & future) ----
    print("Computing habitability index + validating...")
    hab_present = habitability(base["tmean"], base["ai"])
    hab_future = habitability(fut30["tmean"], fut30["ai"])
    hab_future_low = habitability(fut50["tmean"], fut50["ai"])

    # Reference-region validation
    val_rows = []
    for rlat, rlon, label, hi in REFERENCE_REGIONS:
        v = float(sample_grid(hab_present, lat, lon,
                              np.array([rlat]), np.array([rlon]))[0])
        val_rows.append((label, hi, v))
    hi_vals = [v for _, hi, v in val_rows if hi]
    lo_vals = [v for _, hi, v in val_rows if not hi]
    sep = float(np.nanmean(hi_vals) - np.nanmean(lo_vals))
    print("  Reference-region habitability (1=best):")
    for label, hi, v in val_rows:
        print(f"    {'FARM ' if hi else 'BARREN'}  {v:0.2f}  {label}")
    print(f"  Mean(farmland)={np.nanmean(hi_vals):.2f}  "
          f"Mean(barren)={np.nanmean(lo_vals):.2f}  separation={sep:+.2f}")

    # ---- Save outputs ----
    print("Writing outputs/ ...")
    lon180 = ((lon + 180) % 360) - 180
    order = np.argsort(lon180)
    lon_sorted = lon180[order]
    LON, LAT = np.meshgrid(lon_sorted, lat)

    def col(field):
        return field[:, order].ravel()

    grid = pd.DataFrame({
        "lat": LAT.ravel(), "lon": LON.ravel(),
        "soil_temp_present": col(st_present),
        "soil_temp_future": col(st_future),
        "soil_temp_future_low": col(st_future_low),
        "habitability_present": col(hab_present),
        "habitability_future": col(hab_future),
        "habitability_future_low": col(hab_future_low),
    }).dropna(subset=["soil_temp_present"]).reset_index(drop=True)
    grid.to_parquet(f"{OUT_DIR}/predictions_grid.parquet", index=False)
    print(f"  predictions_grid.parquet  ({len(grid):,} land cells)")

    # -----------------------------------------------------------------------
    # ANIMATED OUTPUTS for the app: per-year overlay grid + observed-point
    # time series. Past years (<= YEAR_END) use the ACTUAL historical climate
    # of that year (real data, so they wiggle); future years are projected under
    # BOTH the recent 30-yr rate ("r30") and the full 50-yr rate ("r50") so the
    # app can offer a warming-rate toggle. Historical rows are tagged "hist".
    # -----------------------------------------------------------------------
    print("Building animated per-year outputs (real history + 30yr/50yr projections)...")
    HIST_YEARS = list(range(1970, YEAR_END + 1, 5))      # real climate
    FUT_YEARS = list(range(2020, 2081, 5))               # projected
    RATES = {"r30": slope30, "r50": slope50}

    # jobs: (year, rate_tag, slope_dict). Historical years computed once.
    jobs = [(y, "hist", slope30) for y in HIST_YEARS]
    jobs += [(y, tag, sl) for y in FUT_YEARS for tag, sl in RATES.items()]

    def features_for(year, slope):
        """Full-grid feature dict: real climate if historical, else trend."""
        if year <= YEAR_END:
            idx = year - YEAR_START
            return {k: feats[k][idx] for k in FEATURE_COLS}
        dt = year - BASELINE_MID
        fd = {k: base[k] + slope[k] * dt for k in FEATURE_COLS}
        fd["ai"] = np.clip(fd["ai"], 0, 3.0)
        fd["pann"] = np.clip(fd["pann"], 0, None)
        return fd

    # Thinned overlay cells (~1.5 deg) so the browser animation stays smooth.
    flat_lat, flat_lon = LAT.ravel(), LON.ravel()
    thin = (np.round(flat_lat / 1.5) * 1.5) * 1000 + np.round(flat_lon / 1.5) * 1.5
    base_t_flat = col(base["tmean"])
    keepdf = pd.DataFrame({"k": thin, "i": np.arange(flat_lat.size),
                           "ok": np.isfinite(base_t_flat)})
    keep_idx = keepdf[keepdf["ok"]].groupby("k")["i"].first().to_numpy()

    overlay_rows = []
    for yr, tag, sl in jobs:
        fd = features_for(yr, sl)
        st_y = col(predict_grid(fd))[keep_idx]
        hb_y = col(habitability(fd["tmean"], fd["ai"]))[keep_idx]
        ha_y = col(heat_aridity_suit(fd["tmean"], fd["ai"]))[keep_idx]
        overlay_rows.append(pd.DataFrame({
            "lat": flat_lat[keep_idx], "lon": flat_lon[keep_idx], "year": yr,
            "rate": tag, "soil_temp": st_y, "habitability": hb_y,
            "heat_aridity": ha_y, "is_future": yr > YEAR_END,
        }))
    overlay = pd.concat(overlay_rows, ignore_index=True).dropna(subset=["soil_temp"])
    overlay.to_parquet(f"{OUT_DIR}/overlay_grid.parquet", index=False)
    print(f"  overlay_grid.parquet  ({len(keep_idx):,} cells x {len(jobs)} year/rate frames)")

    # Observed-point soil temp per year: anchor to the REAL measured value and
    # add the model's change relative to baseline (real-climate change for the
    # past, trend change for the future) -> at baseline it equals the data.
    plat, plon = obs["latitude"].to_numpy(), obs["longitude"].to_numpy()
    obs_ts = obs["AnnualTs"].to_numpy()
    base_pt = {k: sample_grid(base[k], lat, lon, plat, plon) for k in FEATURE_COLS}
    slope_pt = {tag: {k: sample_grid(sl[k], lat, lon, plat, plon) for k in FEATURE_COLS}
                for tag, sl in RATES.items()}
    modeled_base = final.predict(np.column_stack([base_pt[k] for k in FEATURE_COLS]))
    point_rows = []
    for yr, tag, sl in jobs:
        if yr <= YEAR_END:
            idx = yr - YEAR_START
            fpt = {k: sample_grid(feats[k][idx], lat, lon, plat, plon) for k in FEATURE_COLS}
        else:
            dt = yr - BASELINE_MID
            spt = slope_pt[tag]
            fpt = {k: base_pt[k] + spt[k] * dt for k in FEATURE_COLS}
            fpt["ai"] = np.clip(fpt["ai"], 0, 3.0)
            fpt["pann"] = np.clip(fpt["pann"], 0, None)
        modeled = final.predict(np.column_stack([fpt[k] for k in FEATURE_COLS]))
        point_rows.append(pd.DataFrame({
            "lat": plat, "lon": plon, "year": yr, "rate": tag,
            "soil_temp": obs_ts + (modeled - modeled_base),
            "is_future": yr > YEAR_END,
        }))
    points_ts = pd.concat(point_rows, ignore_index=True)
    points_ts.to_parquet(f"{OUT_DIR}/points_timeseries.parquet", index=False)
    print(f"  points_timeseries.parquet  ({len(obs_ts):,} points x {len(jobs)} frames)")

    obs_out = obs.iloc[np.flatnonzero(keep)].copy()
    obs_out["soil_temp_pred_present"] = final.predict(Xobs_arr)
    obs_out.to_csv(f"{OUT_DIR}/points_predictions.csv", index=False)

    metrics = {
        "soil_temp_cv_r2_with_airtemp": round(r2_full, 3),
        "soil_temp_cv_rmse_with_airtemp": round(rmse_full, 2),
        "soil_temp_cv_r2_without_airtemp": round(r2_noair, 3),
        "soil_temp_cv_rmse_without_airtemp": round(rmse_noair, 2),
        "habitability_validation_mean_farmland": round(float(np.nanmean(hi_vals)), 3),
        "habitability_validation_mean_barren": round(float(np.nanmean(lo_vals)), 3),
        "habitability_validation_separation": round(sep, 3),
        "future_year": FUTURE_YEAR,
        "baseline_period": f"{BASELINE_START}-{YEAR_END}",
        "n_obs_used": int(len(y)),
    }
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- PNG maps ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_map(field, title, fname, cmap, vmin, vmax):
        fig, ax = plt.subplots(figsize=(11, 5))
        m = ax.pcolormesh(lon_sorted, lat, field[:, order], cmap=cmap,
                          vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(title); ax.set_xlabel("lon"); ax.set_ylabel("lat")
        fig.colorbar(m, ax=ax, shrink=0.8)
        fig.tight_layout(); fig.savefig(f"{OUT_DIR}/{fname}", dpi=110)
        plt.close(fig)

    save_map(st_present, "Predicted soil temperature - present (C)",
             "soiltemp_present.png", "RdYlBu_r", -10, 35)
    save_map(st_future, f"Predicted soil temperature - ~{FUTURE_YEAR} (C)",
             "soiltemp_future.png", "RdYlBu_r", -10, 35)
    save_map(st_future - st_present, f"Soil temp change by ~{FUTURE_YEAR} (C)",
             "soiltemp_delta.png", "Reds", 0, 4)
    save_map(hab_present, "Habitability index - present (1=best)",
             "habitability_present.png", "YlGn", 0, 1)
    save_map(hab_future, f"Habitability index - ~{FUTURE_YEAR}",
             "habitability_future.png", "YlGn", 0, 1)
    save_map(hab_future - hab_present, f"Habitability change by ~{FUTURE_YEAR}",
             "habitability_delta.png", "RdBu", -0.5, 0.5)

    print("Done. Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
