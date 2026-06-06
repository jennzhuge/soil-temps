"""
Soil-temperature pipeline — real past (NCEP) + supervised future projection.

  1. Loads NCEP/NCAR Reanalysis monthly fields (all ~1.9° global, 1948-2025,
     same grid): soil temperature (0-10 cm), 2 m air temperature, precipitation.
  2. Derives bioclim-style FEATURES from air + precip (the model inputs) and uses
     the real annual-mean SOIL temperature as the label.
  3. Trains an XGBoost model on the WHOLE land grid x all years (1948-2025),
     validated by held-out years (GroupKFold by year — honest given that
     neighboring cells/years are correlated).
  4. PAST years (1950-2025) show the *real* NCEP soil temperature.
  5. FUTURE years (2030-2080) extrapolate each feature's recent trend forward and
     run the model on it (no anchoring — a small step at the real->predicted
     switch is expected and left as-is).
  6. Writes outputs/ (overlay grid + observed-point series + metrics) on the
     native NCEP grid.

Run:  ../datavis/bin/python train_soil_temp.py
"""

import json
import os
import numpy as np
import pandas as pd
import xarray as xr

from soil_lib import FEATURE_COLS, habitability, heat_aridity_suit

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SOIL_NC = "data/tmp.0-10cm.mon.mean.nc"     # label: 0-10 cm soil temperature (K)
AIR_NC = "data/air.2m.mon.mean.nc"          # feature: 2 m air temperature (K)
PRATE_NC = "data/prate.sfc.mon.mean.nc"     # feature: precip rate (kg/m^2/s)
LAND_NC = "data/land.sfc.gauss.nc"          # NCEP land/sea mask (1=land, 0=ocean)
OBS_CSV = "dataset.csv"                      # Restor sites (coordinates only)
OUT_DIR = "outputs"

YEAR_START, YEAR_END = 1948, 2025            # full-year training/real range
BASELINE_START = 1996                        # recent 30-yr window -> "present" ~2025
BASELINE_MID = (BASELINE_START + YEAR_END) / 2.0    # 2010.5
RATE50_START = 1976                          # full 50-yr window

HIST_YEARS = list(range(1950, YEAR_END + 1, 5))      # real soil temp
FUT_YEARS = list(range(2030, 2081, 5))               # projected

DAYS_IN_MONTH = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MONTH_MID_DOY = np.array([15, 45, 74, 105, 135, 162, 198, 228, 258, 288, 318, 344])


# ---------------------------------------------------------------------------
# 1. Load monthly NCEP fields, sliced & unit-converted, as (nyear, 12, lat, lon)
# ---------------------------------------------------------------------------
def load_monthly():
    sl = slice(f"{YEAR_START}", f"{YEAR_END}")
    soil = xr.open_dataset(SOIL_NC)["tmp"].sel(time=sl).load() - 273.15      # degC
    air = xr.open_dataset(AIR_NC)["air"].sel(time=sl).load() - 273.15        # degC
    prate = xr.open_dataset(PRATE_NC)["prate"].sel(time=sl).load()           # kg/m2/s
    lat, lon = soil["lat"].values, soil["lon"].values
    nyear = YEAR_END - YEAR_START + 1
    shp = (nyear, 12, lat.size, lon.size)
    T = air.values.reshape(shp)
    Ts = soil.values.reshape(shp)
    # precip rate -> monthly total mm (1 kg/m2 == 1 mm); rate * seconds in month
    secs = (DAYS_IN_MONTH * 86400.0)[None, :, None, None]
    P = prate.values.reshape(shp) * secs
    return T, P, Ts, lat, lon


# ---------------------------------------------------------------------------
# 2. Thornthwaite PET (mm/yr) for one year, vectorised over (lat, lon)
# ---------------------------------------------------------------------------
def daylight_hours(lat):
    lat_rad = np.radians(lat)[None, :]
    decl = 0.409 * np.sin(2 * np.pi * MONTH_MID_DOY / 365.0 - 1.39)[:, None]
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(decl), -1, 1))
    return 24.0 / np.pi * ws                                # (12, nlat)


def pet_year(Tm, daylen):
    Tpos = np.clip(Tm, 0, None)
    I = np.sum((Tpos / 5.0) ** 1.514, axis=0)
    a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239
    Isafe = np.where(I <= 0, np.nan, I)
    corr = (daylen[:, :, None] / 12.0) * (DAYS_IN_MONTH[:, None, None] / 30.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        unadj = 16.0 * (10.0 * Tm / Isafe[None]) ** a[None]
    hot = -415.85 + 32.24 * Tm - 0.43 * Tm**2
    monthly = np.where(Tm <= 0, 0.0, np.where(Tm < 26.5, unadj, hot))
    monthly = np.where(np.isfinite(monthly), monthly, 0.0)
    pet = np.sum(monthly * corr, axis=0)
    return np.where(I <= 0, 0.0, pet)


# ---------------------------------------------------------------------------
# 3. Per-year features (nyear, nlat, nlon) from air + precip
# ---------------------------------------------------------------------------
def build_features(T, P, lat):
    nyear = T.shape[0]
    daylen = daylight_hours(lat)
    feats = {k: np.full((nyear, lat.size, T.shape[3]), np.nan, dtype="float32")
             for k in FEATURE_COLS}
    for y in range(nyear):
        Tm, Pm = T[y], P[y]
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
    """Per-pixel linear slope (units/yr) over the given years."""
    ny = stack.shape[0]
    Y = stack.reshape(ny, -1)
    valid = np.isfinite(Y).all(axis=0)
    slope = np.full(Y.shape[1], np.nan)
    if valid.any():
        slope[valid] = np.polyfit(years.astype("float64"),
                                  Y[:, valid].astype("float64"), 1)[0]
    return slope.reshape(stack.shape[1:])


def nearest_idx(lat, lon, plat, plon):
    """Nearest NCEP cell indices for point arrays (lat is Gaussian, lon 0-360)."""
    ilat = np.abs(lat[None, :] - plat[:, None]).argmin(1)
    ilon = np.abs(lon[None, :] - np.mod(plon, 360.0)[:, None]).argmin(1)
    return ilat, ilon


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading NCEP monthly fields (1948-2025)...")
    T, P, Ts, lat, lon = load_monthly()
    years = np.arange(YEAR_START, YEAR_END + 1)

    print("Building features + annual soil temperature...")
    feats = build_features(T, P, lat)
    soil = np.nanmean(Ts, axis=1)                       # (nyear, nlat, nlon) degC
    # Land mask = NCEP land/sea mask AND a finite soil value every year. The
    # land/sea mask is essential: NCEP reports a "soil temperature" over Arctic
    # sea-ice, so finiteness alone would wrongly keep Arctic Ocean cells.
    land = xr.open_dataset(LAND_NC)["land"].values.squeeze() > 0.5
    landmask = np.isfinite(soil).all(axis=0) & land
    print(f"  {int(landmask.sum())} land cells of {landmask.size}")

    # ---- Training table: land cells x years ----
    feat_stack = np.stack([feats[k] for k in FEATURE_COLS], axis=-1)  # (ny,lat,lon,7)
    yr_grid = np.repeat(years, int(landmask.sum()))     # group label per row
    X = feat_stack[:, landmask, :].reshape(-1, len(FEATURE_COLS))
    y = soil[:, landmask].reshape(-1)
    ok = np.isfinite(y)                                 # all land/years are finite
    X, y, groups = X[ok], y[ok], yr_grid[ok]
    print(f"  training rows: {len(y):,}")

    from xgboost import XGBRegressor
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import r2_score, mean_squared_error

    def make_model():
        return XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", n_jobs=-1, random_state=0)

    def evaluate(cols_idx, tag):
        gkf = GroupKFold(n_splits=5)
        pred = cross_val_predict(make_model(), X[:, cols_idx], y,
                                 cv=gkf, groups=groups)
        r2 = r2_score(y, pred)
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        print(f"  [{tag}] held-out-year CV  R2={r2:.3f}  RMSE={rmse:.2f} C")
        return round(r2, 3), round(rmse, 2)

    print("Validating (GroupKFold by year)...")
    all_idx = list(range(len(FEATURE_COLS)))
    r2_full, rmse_full = evaluate(all_idx, "with air temp")
    noair = [i for i, k in enumerate(FEATURE_COLS) if k not in ("tmean", "twarm", "tcold")]
    r2_noair, rmse_noair = evaluate(noair, "without air temp")

    print("Fitting final model on all rows...")
    model = make_model().fit(X, y)

    # ---- Feature projection (per-cell trend, recent 30-yr and full 50-yr) ----
    print("Computing feature trends + projections...")
    base = {k: np.nanmean(feats[k][years >= BASELINE_START], axis=0) for k in FEATURE_COLS}
    s30 = {k: ols_slope(feats[k][years >= BASELINE_START], years[years >= BASELINE_START])
           for k in FEATURE_COLS}
    s50 = {k: ols_slope(feats[k][years >= RATE50_START], years[years >= RATE50_START])
           for k in FEATURE_COLS}
    RATES = {"r30": s30, "r50": s50}

    def project(slope, year):
        dt = year - BASELINE_MID
        fd = {k: base[k] + slope[k] * dt for k in FEATURE_COLS}
        fd["ai"] = np.clip(fd["ai"], 0, 3.0)
        fd["pann"] = np.clip(fd["pann"], 0, None)
        return fd

    def predict_field(fdict):
        stack = np.stack([fdict[k][landmask] for k in FEATURE_COLS], axis=-1)
        out = np.full(landmask.shape, np.nan, dtype="float32")
        out[landmask] = model.predict(stack)
        return out

    # ---- Build animation frames on the land grid ----
    ii, jj = np.where(landmask)
    lon180 = ((lon + 180) % 360) - 180
    cell_lat, cell_lon = lat[ii], lon180[jj]

    def frame_rows(year, rate, soil_field, feat_for_overlays, is_future):
        st = soil_field[ii, jj]
        tm = feat_for_overlays["tmean"][ii, jj]
        ai = feat_for_overlays["ai"][ii, jj]
        return pd.DataFrame({
            "lat": cell_lat, "lon": cell_lon, "year": year, "rate": rate,
            "soil_temp": st,
            "habitability": habitability(tm, ai),
            "heat_aridity": heat_aridity_suit(tm, ai),
            "is_future": is_future,
        })

    print("Assembling overlay grid (real past + projected future)...")
    rows = []
    for yr in HIST_YEARS:                                # real soil temp
        idx = yr - YEAR_START
        feat_yr = {k: feats[k][idx] for k in FEATURE_COLS}
        rows.append(frame_rows(yr, "hist", soil[idx], feat_yr, False))
    for yr in FUT_YEARS:                                 # modelled
        for tag, slope in RATES.items():
            fd = project(slope, yr)
            rows.append(frame_rows(yr, tag, predict_field(fd), fd, True))
    overlay = pd.concat(rows, ignore_index=True).dropna(subset=["soil_temp"])
    overlay.to_parquet(f"{OUT_DIR}/overlay_grid.parquet", index=False)
    print(f"  overlay_grid.parquet  ({len(ii):,} land cells x "
          f"{len(HIST_YEARS) + 2 * len(FUT_YEARS)} frames)")

    # ---- Observed Restor points -> nearest NCEP cell ----
    print("Mapping Restor points to nearest cells...")
    obs = pd.read_csv(OBS_CSV).dropna(subset=["longitude", "latitude"]).reset_index(drop=True)
    pilat, pilon = nearest_idx(lat, lon, obs["latitude"].to_numpy(), obs["longitude"].to_numpy())
    prows = []
    for yr in HIST_YEARS:
        st = soil[yr - YEAR_START][pilat, pilon]
        prows.append(pd.DataFrame({"lat": obs["latitude"], "lon": obs["longitude"],
                                   "year": yr, "rate": "hist", "soil_temp": st,
                                   "is_future": False}))
    for yr in FUT_YEARS:
        for tag, slope in RATES.items():
            field = predict_field(project(slope, yr))
            st = field[pilat, pilon]
            prows.append(pd.DataFrame({"lat": obs["latitude"], "lon": obs["longitude"],
                                       "year": yr, "rate": tag, "soil_temp": st,
                                       "is_future": True}))
    points = pd.concat(prows, ignore_index=True).dropna(subset=["soil_temp"])
    points.to_parquet(f"{OUT_DIR}/points_timeseries.parquet", index=False)
    print(f"  points_timeseries.parquet  ({len(obs):,} points x frames)")

    metrics = {
        "soil_temp_cv_r2_with_airtemp": r2_full,
        "soil_temp_cv_rmse_with_airtemp": rmse_full,
        "soil_temp_cv_r2_without_airtemp": r2_noair,
        "soil_temp_cv_rmse_without_airtemp": rmse_noair,
        "n_train_rows": int(len(y)),
        "train_years": f"{YEAR_START}-{YEAR_END}",
        "grid": "NCEP ~1.9 deg",
    }
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Done. Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
