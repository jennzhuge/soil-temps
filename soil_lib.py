"""
Shared logic used by BOTH train_soil_temp.py (the pipeline) and app.py (the UI),
so the year slider in the app computes exactly what the pipeline trained on.
"""

import numpy as np

# Climate feature columns fed to the soil-temperature model, in fixed order.
FEATURE_COLS = ["tmean", "pann", "tseason", "twarm", "tcold", "pet", "ai"]

# The present baseline is the mean of 1988-2017, i.e. centred on this year.
# A slider year of BASELINE_MID reproduces "present"; later years extrapolate.
BASELINE_START = 1988
YEAR_END = 2017
BASELINE_MID = (BASELINE_START + YEAR_END) / 2.0   # 2002.5


def project_features(base, slope, year):
    """
    Project each feature to `year` by linear extrapolation of its warming rate:
        value(year) = baseline + slope * (year - BASELINE_MID)
    `base` and `slope` are dicts (or DataFrame-like) of arrays keyed by FEATURE_COLS.
    """
    dt = year - BASELINE_MID
    out = {k: base[k] + slope[k] * dt for k in FEATURE_COLS}
    out["ai"] = np.clip(out["ai"], 0, 3.0)            # aridity index stays physical
    out["pann"] = np.clip(out["pann"], 0, None)       # precipitation >= 0
    return out


def _cold_suit(tmean):
    """Too-cold-to-farm penalty: ~0 below freezing, ~1 above ~5 C."""
    return 1.0 / (1.0 + np.exp(-(tmean - 2.0) / 2.0))


def _hot_suit(tmean):
    """Too-hot penalty: ~1 below ~27 C, ~0 above ~31 C (human-climate-niche
    upper limit ~29 C, Xu et al. 2020)."""
    return 1.0 / (1.0 + np.exp((tmean - 29.0) / 2.0))


def _water_suit(ai):
    """Aridity: 0 hyperarid (AI<=0.05) -> 1 sub-humid (AI>=0.65; UNCCD dryland
    boundary)."""
    return np.clip((ai - 0.05) / (0.65 - 0.05), 0, 1)


def heat_aridity_suit(tmean, ai):
    """
    Heat + aridity suitability only (ignores cold). This is the 'cropland heat
    limit' / desertification axis: it falls as land gets too hot OR too dry, and
    so GROWS more unfarmable as the climate warms. Low values = heat/aridity-
    limited land.
    """
    return _hot_suit(tmean) * _water_suit(ai)


def habitability(tmean, ai):
    """
    Full habitability, 0 (unliveable) to 1 (ideal): cold x heat x water. Used for
    the general index; for the heat-driven 'cropland heat limit' overlay use
    heat_aridity_suit() instead (which excludes the cold penalty).
    """
    return _cold_suit(tmean) * heat_aridity_suit(tmean, ai)
