"""
Generate synthetic RTS-GMLC-style time series data.

Mirrors the format of GridMod/RTS-GMLC RTS_Data/timeseries_data_files, but:
  - Only SourceData is taken verbatim from RTS-GMLC (copied in separately by the user).
  - No FormattedData is produced (Prescient builds that from SourceData + timeseries
    at run time).
  - timeseries_data_files are regenerated as *synthetic* profiles (not copied from
    RTS-GMLC) and truncated to a 48-hour window: one simulated PCM day, plus the
    look-ahead day the UC needs.

Layout produced (relative to this script's ../RTS_Data/timeseries_data_files):
  Load/DAY_AHEAD_regional_Load.csv        (hourly,  Period 1-24 x 2 days)
  Load/REAL_TIME_regional_Load.csv        (5-min,   Period 1-288 x 2 days)
  PV/DAY_AHEAD_pv.csv, PV/REAL_TIME_pv.csv
  WIND/DAY_AHEAD_wind.csv, WIND/REAL_TIME_wind.csv
  RTPV/DAY_AHEAD_rtpv.csv, RTPV/REAL_TIME_rtpv.csv
  Hydro/DAY_AHEAD_hydro.csv, Hydro/REAL_TIME_hydro.csv
  Hydro/DAY_AHEAD_hydro_inflow.csv, Hydro/REAL_TIME_hydro_inflow.csv

Column headers (generator/zone IDs) match RTS-GMLC's SourceData exactly, so the
synthetic timeseries lines up with the RTS-GMLC bus_generator.csv / bus.csv that
is copied in separately.
"""

import json
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

START_YEAR = 2020
START_MONTH = 1
START_DAY = 1
N_DAYS = 2  # 1 PCM day + 1 UC look-ahead day = 48 hours total

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "RTS_Data", "timeseries_data_files")
PROFILES_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "synthetic_profiles.json")

# ---------------------------------------------------------------------------
# Generator / zone ID lists, taken from RTS-GMLC's timeseries_data_files headers
# (these names must match the SourceData bus_generator.csv the user copies in).
# ---------------------------------------------------------------------------

LOAD_ZONES = ["1", "2", "3"]

PV_GENS = [
    "320_PV_1", "314_PV_1", "314_PV_2", "313_PV_1", "314_PV_3", "314_PV_4",
    "313_PV_2", "310_PV_1", "324_PV_1", "312_PV_1", "310_PV_2", "324_PV_2",
    "324_PV_3", "113_PV_1", "319_PV_1", "215_PV_1", "102_PV_1", "101_PV_1",
    "102_PV_2", "104_PV_1", "101_PV_2", "101_PV_3", "101_PV_4", "103_PV_1",
    "119_PV_1",
]
# nameplate capacity (MW) per PV unit -- rough RTS-GMLC-like values
PV_CAP = {g: c for g, c in zip(PV_GENS, RNG.uniform(2, 75, size=len(PV_GENS)))}

RTPV_GENS = [
    "308_RTPV_1", "313_RTPV_1", "313_RTPV_2", "313_RTPV_3", "313_RTPV_4",
    "313_RTPV_5", "313_RTPV_6", "313_RTPV_7", "313_RTPV_8", "313_RTPV_9",
    "313_RTPV_10", "313_RTPV_11", "313_RTPV_12", "320_RTPV_1", "320_RTPV_2",
    "320_RTPV_3", "313_RTPV_13", "320_RTPV_4", "320_RTPV_5", "118_RTPV_1",
    "118_RTPV_2", "118_RTPV_3", "118_RTPV_4", "118_RTPV_5", "118_RTPV_6",
    "320_RTPV_6", "118_RTPV_7", "118_RTPV_8", "118_RTPV_9", "118_RTPV_10",
    "213_RTPV_1",
]
RTPV_CAP = {g: c for g, c in zip(RTPV_GENS, RNG.uniform(0.5, 5, size=len(RTPV_GENS)))}

WIND_GENS = ["309_WIND_1", "317_WIND_1", "303_WIND_1", "122_WIND_1"]
WIND_CAP = {"309_WIND_1": 148.3, "317_WIND_1": 799.1, "303_WIND_1": 847.0, "122_WIND_1": 713.5}

HYDRO_GENS = [
    "122_HYDRO_1", "122_HYDRO_2", "122_HYDRO_3", "122_HYDRO_4", "122_HYDRO_5",
    "122_HYDRO_6", "201_HYDRO_4", "215_HYDRO_1", "215_HYDRO_2", "215_HYDRO_3",
    "222_HYDRO_1", "222_HYDRO_2", "222_HYDRO_3", "222_HYDRO_4", "222_HYDRO_5",
    "222_HYDRO_6", "322_HYDRO_1", "322_HYDRO_2", "322_HYDRO_3", "322_HYDRO_4",
]
HYDRO_CAP = {g: c for g, c in zip(HYDRO_GENS, RNG.uniform(4, 22, size=len(HYDRO_GENS)))}


def date_period_columns(n_days, periods_per_day):
    """Return Year, Month, Day, Period arrays for n_days starting START_*."""
    dates = pd.date_range(
        start=f"{START_YEAR}-{START_MONTH:02d}-{START_DAY:02d}", periods=n_days, freq="D"
    )
    years, months, days, periods = [], [], [], []
    for d in dates:
        for p in range(1, periods_per_day + 1):
            years.append(d.year)
            months.append(d.month)
            days.append(d.day)
            periods.append(p)
    return years, months, days, periods


def diurnal_load_shape(hours):
    """Double-hump daily load shape, normalized to peak 1.0."""
    morning = np.exp(-((hours - 8.5) ** 2) / (2 * 2.2 ** 2))
    evening = np.exp(-((hours - 19.5) ** 2) / (2 * 2.6 ** 2))
    base = 0.55
    shape = base + 0.45 * morning + 0.55 * evening
    return shape / shape.max()

def diurnal_solar_shape(hours):
    """Daylight bell curve, zero outside ~6:00-19:00."""
    shape = np.exp(-((hours - 12.5) ** 2) / (2 * 3.0 ** 2))
    shape[(hours < 6) | (hours > 19)] = 0.0
    return shape / shape.max()


def build_series(n_days, periods_per_day, hour_of_period):
    """hour_of_period: fractional hour-of-day for each period, length n_days*periods_per_day."""
    years, months, days, periods = date_period_columns(n_days, periods_per_day)
    return years, months, days, periods


def make_frame(years, months, days, periods, data_cols):
    df = pd.DataFrame({"Year": years, "Month": months, "Day": days, "Period": periods})
    for name, vals in data_cols.items():
        df[name] = vals
    return df


def hours_array(n_days, periods_per_day):
    return np.tile(np.linspace(0, 24, periods_per_day, endpoint=False), n_days)


def gen_load(periods_per_day, noise_scale):
    n = N_DAYS * periods_per_day
    hours = hours_array(N_DAYS, periods_per_day)
    shape = diurnal_load_shape(hours)
    zone_peak = {"1": 1000.0, "2": 1200.0, "3": 1350.0}
    cols = {}
    for z, peak in zone_peak.items():
        noise = RNG.normal(0, noise_scale, size=n)
        cols[z] = np.clip(peak * shape * (1 + noise), 0, None)
    return cols, hours


def gen_solar(gens, cap, periods_per_day, noise_scale):
    n = N_DAYS * periods_per_day
    hours = hours_array(N_DAYS, periods_per_day)
    shape = diurnal_solar_shape(hours)
    cols = {}
    for g in gens:
        noise = RNG.normal(0, noise_scale, size=n)
        cols[g] = np.clip(cap[g] * shape * (1 + noise), 0, cap[g])
    return cols


def gen_wind(gens, cap, periods_per_day, noise_scale):
    n = N_DAYS * periods_per_day
    cols = {}
    for g in gens:
        walk = RNG.normal(0, noise_scale, size=n).cumsum()
        walk -= walk.min()
        walk = walk / (walk.max() + 1e-9)
        base_level = RNG.uniform(0.25, 0.55)
        profile = np.clip(base_level + 0.5 * walk + RNG.normal(0, 0.03, size=n), 0.02, 1.0)
        cols[g] = cap[g] * profile
    return cols


def gen_hydro(gens, cap, periods_per_day, noise_scale):
    n = N_DAYS * periods_per_day
    cols = {}
    for g in gens:
        base = RNG.uniform(0.5, 0.9) * cap[g]
        noise = RNG.normal(0, noise_scale, size=n)
        cols[g] = np.clip(base * (1 + noise), 0, cap[g])
    return cols


def write_pair(subdir, base_name, gens_or_zones, cap_or_none, gen_func, kwargs_da, kwargs_rt):
    da_dir = os.path.join(OUT_ROOT, subdir)
    os.makedirs(da_dir, exist_ok=True)

    # DAY_AHEAD: hourly, 24 periods/day
    years, months, days, periods = date_period_columns(N_DAYS, 24)
    cols = gen_func(**kwargs_da)
    if isinstance(cols, tuple):
        cols = cols[0]
    df_da = make_frame(years, months, days, periods, cols)
    df_da.to_csv(os.path.join(da_dir, f"DAY_AHEAD_{base_name}.csv"), index=False)
    day_ahead_cols = cols

    # REAL_TIME: 5-minute, 288 periods/day
    years, months, days, periods = date_period_columns(N_DAYS, 288)
    cols = gen_func(**kwargs_rt)
    if isinstance(cols, tuple):
        cols = cols[0]
    df_rt = make_frame(years, months, days, periods, cols)
    df_rt.to_csv(os.path.join(da_dir, f"REAL_TIME_{base_name}.csv"), index=False)

    return day_ahead_cols


def main():
    profiles = {}

    # Load
    profiles["Load"] = write_pair(
        "Load", "regional_Load", LOAD_ZONES, None, gen_load,
        kwargs_da=dict(periods_per_day=24, noise_scale=0.02),
        kwargs_rt=dict(periods_per_day=288, noise_scale=0.03),
    )

    # PV
    profiles["PV"] = write_pair(
        "PV", "pv", PV_GENS, PV_CAP, gen_solar,
        kwargs_da=dict(gens=PV_GENS, cap=PV_CAP, periods_per_day=24, noise_scale=0.05),
        kwargs_rt=dict(gens=PV_GENS, cap=PV_CAP, periods_per_day=288, noise_scale=0.08),
    )

    # RTPV
    profiles["RTPV"] = write_pair(
        "RTPV", "rtpv", RTPV_GENS, RTPV_CAP, gen_solar,
        kwargs_da=dict(gens=RTPV_GENS, cap=RTPV_CAP, periods_per_day=24, noise_scale=0.05),
        kwargs_rt=dict(gens=RTPV_GENS, cap=RTPV_CAP, periods_per_day=288, noise_scale=0.08),
    )

    # WIND
    profiles["WIND"] = write_pair(
        "WIND", "wind", WIND_GENS, WIND_CAP, gen_wind,
        kwargs_da=dict(gens=WIND_GENS, cap=WIND_CAP, periods_per_day=24, noise_scale=0.06),
        kwargs_rt=dict(gens=WIND_GENS, cap=WIND_CAP, periods_per_day=288, noise_scale=0.10),
    )

    # Hydro generation
    profiles["Hydro"] = write_pair(
        "Hydro", "hydro", HYDRO_GENS, HYDRO_CAP, gen_hydro,
        kwargs_da=dict(gens=HYDRO_GENS, cap=HYDRO_CAP, periods_per_day=24, noise_scale=0.03),
        kwargs_rt=dict(gens=HYDRO_GENS, cap=HYDRO_CAP, periods_per_day=288, noise_scale=0.04),
    )

    # Hydro inflow (reservoir head) -- same generators, "_RESERVOIR_head" suffix,
    # RTS-GMLC keeps this numerically identical to the hydro generation series.
    hydro_inflow_gens = [f"{g}_RESERVOIR_head" for g in HYDRO_GENS]
    inflow_cap = {f"{g}_RESERVOIR_head": HYDRO_CAP[g] for g in HYDRO_GENS}
    write_pair(
        "Hydro", "hydro_inflow", hydro_inflow_gens, inflow_cap, gen_hydro,
        kwargs_da=dict(gens=hydro_inflow_gens, cap=inflow_cap, periods_per_day=24, noise_scale=0.03),
        kwargs_rt=dict(gens=hydro_inflow_gens, cap=inflow_cap, periods_per_day=288, noise_scale=0.04),
    )

    # Dump the DAY_AHEAD profiles as JSON, in the same
    # {"Category": {"gen_name": [hourly values]}} format that update_gmlc.py consumes.
    json_profiles = {
        category: {name: np.asarray(series).tolist() for name, series in cols.items()}
        for category, cols in profiles.items()
    }
    with open(PROFILES_JSON_PATH, "w") as f:
        json.dump(json_profiles, f, indent=2)

    print(f"Synthetic timeseries written under: {os.path.abspath(OUT_ROOT)}")
    print(f"Synthetic profiles JSON written to: {os.path.abspath(PROFILES_JSON_PATH)}")


if __name__ == "__main__":
    main()
