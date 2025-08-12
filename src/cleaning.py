"""Data collection and cleaning pipeline for FastF1 telemetry.

This module:
- Pulls session schedules and telemetry via FastF1.
- Merges lap telemetry with weather data.
- Encodes categorical columns into global integer IDs (for later one-hot).
- Resamples each lap to a fixed number of distance-based steps.
- Saves per-driver lap cubes (.npy) and per-lap summary vectors (.json).
- Tracks which sessions have already been cleaned (to resume work safely).

Outputs
-------
data/fastf1cache/       : FastF1 HTTP cache (managed by FastF1)
data/raw/{year/gp/session}/{driver}.npy
                        : Stack of resampled laps for a driver (shape: L x T x F_raw)
data/processed/{year/gp/session}/{driver}.json
                        : List of per-lap summary vectors (mean of columns)
data/metadata/cleaned_sessions.json
                        : Bookkeeping of sessions already processed
data/metadata/global_encoders.json
                        : Integer label encoders for categorical columns
"""

from __future__ import annotations

import json
from pathlib import Path
from time import sleep
from typing import Dict, List

import fastf1
import numpy as np
import pandas as pd
from fastf1 import get_session
from fastf1.core import Laps
from fastf1.events import get_event_schedule

# -----------------------------------------------------------------------------
# FastF1 cache
# -----------------------------------------------------------------------------
cache_dir = Path(__file__).parent.parent / "data" / "fastf1cache"
cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir.resolve()))

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
SESSION_TYPES = ["Q", "R"]
YEARS = list(range(2019, 2024))

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

STATE_FILE = Path(__file__).parent.parent / "data" / "metadata" / "cleaned_sessions.json"
ENCODER_PATH = Path(__file__).parent.parent / "data" / "metadata" / "global_encoders.json"

ENCODE_COLUMNS = ["Compound", "TrackStatus", "Team", "Driver"]

# Column ordering used both for summary vectors and for resampled matrices
COLUMN_NAMES = [
    "Speed",         # km/h
    "Throttle",      # 0–100 %
    "Brake",         # 0 or 1
    "nGear",         # int (1–8)
    "RPM",           # engine revolutions per minute
    "DRS",           # 0 = off, 1 = on, 2 = active
    "Distance",      # meters

    "AirTemp",       # °C
    "Humidity",      # %
    "Pressure",      # hPa
    "Rainfall",      # mm/hr
    "TrackTemp",     # °C
    "WindDirection", # degrees
    "WindSpeed",     # km/h

    "LapNumber",     # int
    "Stint",         # int
    "Compound",      # string (e.g., "SOFT")
    "TrackStatus",   # int or flag code
    "Team",          # string
    "Driver",        # string
    "LapTime",       # seconds (float)
]


# -----------------------------------------------------------------------------
# Helpers: session tracking and schedules
# -----------------------------------------------------------------------------
def get_schedule(year: int) -> pd.DataFrame:
    """Return the FastF1 event schedule for a year.

    Parameters
    ----------
    year : int
        Championship year.

    Returns
    -------
    pd.DataFrame
        Event schedule. Empty on failure.
    """
    try:
        return get_event_schedule(year)
    except Exception as e:
        print(f"Failed to load schedule for {year}: {e}")
        return pd.DataFrame()


def load_cleaned_state() -> Dict[str, Dict[str, List[str]]]:
    """Load bookkeeping for already-cleaned sessions.

    Returns
    -------
    dict
        Nested mapping: {year: {gp_name: [session_type, ...]}}.
    """
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cleaned_state(state: dict) -> None:
    """Persist bookkeeping for already-cleaned sessions."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def was_cleaned(state: dict, year: int, gp_name: str, session_type: str) -> bool:
    """Check whether a specific session has been cleaned already."""
    return (
        str(year) in state
        and gp_name in state[str(year)]
        and session_type in state[str(year)][gp_name]
    )


def mark_cleaned(state: dict, year: int, gp_name: str, session_type: str) -> None:
    """Mark a session as cleaned in the state mapping."""
    state.setdefault(str(year), {}).setdefault(gp_name, []).append(session_type)


# -----------------------------------------------------------------------------
# Global encoder logic (categoricals → integer IDs)
# -----------------------------------------------------------------------------
def load_encoders() -> dict:
    """Load global encoders for categorical columns."""
    if ENCODER_PATH.exists():
        with open(ENCODER_PATH, "r") as f:
            return json.load(f)
    return {}


def save_encoders(encoders: dict) -> None:
    """Persist global encoders for categorical columns."""
    ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ENCODER_PATH, "w") as f:
        json.dump(encoders, f, indent=2)


def encode_value(encoders: dict, col_name: str, val) -> int:
    """Map a categorical value to a stable integer ID (creating if new).

    Parameters
    ----------
    encoders : dict
        Global mapping of {column: {label: id}}.
    col_name : str
        Column to encode.
    val : Any
        Original categorical value.

    Returns
    -------
    int
        Integer ID for the given value.
    """
    val = str(val)
    if col_name not in encoders:
        encoders[col_name] = {}
        save_encoders(encoders)
    if val not in encoders[col_name]:
        encoders[col_name][val] = len(encoders[col_name])
        save_encoders(encoders)
    return encoders[col_name][val]


# -----------------------------------------------------------------------------
# Resampling: lap → fixed number of distance steps
# -----------------------------------------------------------------------------
def resample_lap_by_distance(df: pd.DataFrame, steps: int = 150) -> np.ndarray:
    """Resample a lap to a fixed number of distance-normalized steps.

    Interpolates each numeric column against normalized distance in [0, 1],
    returning a (steps, features) matrix aligned to equal-distance increments.

    Parameters
    ----------
    df : pd.DataFrame
        Telemetry and enriched data for a single lap, sorted by time.
    steps : int, optional
        Number of distance steps to interpolate to (default: 150).

    Returns
    -------
    np.ndarray
        Resampled lap matrix of shape (steps, features), dtype float32.

    Raises
    ------
    ValueError
        If 'Distance' is missing or zero.
    """
    max_distance = df["Distance"].max()
    if max_distance == 0 or pd.isnull(max_distance):
        raise ValueError("Invalid or missing 'Distance' column")

    df = df.copy()
    df["NormDistance"] = df["Distance"] / max_distance

    target_distances = np.linspace(0, 1, num=steps)

    # ---- Exclude non-interpolated columns ---- #
    exclude_cols = ["Time", "Distance", "NormDistance"]
    interp_cols = [col for col in df.columns if col not in exclude_cols]

    resampled = {}
    for col in interp_cols:
        try:
            resampled[col] = np.interp(target_distances, df["NormDistance"], df[col])
        except Exception as e:
            print(f"⚠️ Interpolation failed for column '{col}': {e}")
            resampled[col] = np.full(steps, np.nan)

    resampled_array = np.column_stack([resampled[col] for col in interp_cols])
    return resampled_array.astype(np.float32)


# -----------------------------------------------------------------------------
# Saving: raw cubes (.npy) and per-lap summaries (.json)
# -----------------------------------------------------------------------------
def save_driver_cube(driver_cube: np.ndarray, year: int, gp: str, session_type: str, driver: str) -> None:
    """Save a driver's stack of laps as a .npy cube (shape: L x T x F_raw)."""
    out_dir = RAW_DATA_DIR / str(year) / gp.replace(" ", "_") / session_type
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{driver}.npy"
    np.save(file_path, driver_cube)
    print(f"💾 Saved raw cube: {file_path}")


def save_summary_vector(
    summary_vector: List[float],
    column_names: List[str],
    year: int,
    gp: str,
    session_type: str,
    driver: str,
    lap_number: int,
    encoders: dict,
) -> None:
    """Append a single-lap summary vector to the driver's JSON log.

    The summary is the mean across telemetry rows for each numeric column;
    categorical integer IDs are converted back to human-readable labels.

    Parameters
    ----------
    summary_vector : list[float]
        Mean of each column for the lap (aligned to COLUMN_NAMES).
    column_names : list[str]
        Names corresponding to `summary_vector`.
    year, gp, session_type, driver, lap_number : Various
        Metadata identifying where to store the summary.
    encoders : dict
        Global categorical encoders used to reverse-map labels.
    """
    file_path = (
        Path(__file__).parent.parent
        / "data"
        / "processed"
        / str(year)
        / gp.replace(" ", "_")
        / session_type
        / f"{driver}.json"
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)

    summary_dict = {
        "lap_number": lap_number,
        "summary": dict(zip(column_names, summary_vector)),
    }

    # ---- Remove fields that don't belong in the summary payload ---- #
    del summary_dict["summary"]["LapNumber"]

    # ---- Reverse-translate encoded categoricals for readability ---- #
    for key in summary_dict["summary"]:
        if key in ENCODE_COLUMNS:
            val = summary_dict["summary"][key]
            reverse_lookup = {v: k for k, v in encoders[key].items()}
            summary_dict["summary"][key] = reverse_lookup.get(int(val), f"UNKNOWN({val})")

    # ---- Read/append/write ---- #
    if file_path.exists():
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(summary_dict)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"💾 Saved summary vector: {file_path} (lap {lap_number})")


# -----------------------------------------------------------------------------
# Core cleaning for a session
# -----------------------------------------------------------------------------
def try_clean_session(year: int, gp_name: str, session_type: str, encoders: dict) -> bool:
    """Load a session and run cleaning; return success flag."""
    try:
        session = get_session(year, gp_name, session_type)
        session.load()
        print(f"✅ Loaded {year} {gp_name} {session_type}")
        clean(session, year, gp_name, session_type, encoders)
        return True
    except Exception as e:
        print(f"❌ Failed {year} {gp_name} {session_type}: {e}")
        return False


def clean(session, year: int, gp_name: str, session_type: str, encoders: dict) -> dict:
    """Clean a FastF1 session and persist outputs.

    Steps
    -----
    1) Get and sort weather data for timestamp alignment.
    2) Pick quick laps only.
    3) For each driver and lap:
       - Fetch telemetry and add distance.
       - Merge with weather (nearest-asof within 60s).
       - Append lap-level metadata to each row.
       - Encode categoricals to global integer IDs.
       - Save lap summary vector.
       - Resample the lap to fixed distance steps; collect per-driver laps.
    4) Save stacked per-driver cubes (.npy).

    Returns
    -------
    dict
        Mapping of driver → cube (shape: L x T x F_raw). Empty if no valid laps.
    """
    # ---- 1) Weather data for time sync ---- #
    weather_data = session.weather_data[
        ["Time", "AirTemp", "Humidity", "Pressure", "Rainfall", "TrackTemp", "WindDirection", "WindSpeed"]
    ].copy()
    weather_data.sort_values("Time", inplace=True)

    # ---- 2) Quick laps only (exclude in/out laps etc.) ---- #
    laps: Laps = session.laps.pick_quicklaps()
    if laps.empty:
        print(f"No valid laps in session {session}")
        return {}

    driver_laps = {drv: laps[laps["Driver"] == drv] for drv in laps["Driver"].unique()}
    driver_cubes = {}

    for driver, drv_laps in driver_laps.items():
        lap_matrices = []

        for _, lap in drv_laps.iterrows():
            try:
                telemetry = lap.get_telemetry().add_distance()
            except Exception as e:
                print(f"❌ Telemetry fail — Lap {lap['LapNumber']}, Driver {driver}: {e}")
                continue

            if telemetry.empty:
                print(f"⚠️ Empty telemetry — Lap {lap['LapNumber']}, Driver {driver}")
                continue

            telemetry = telemetry[["Time", "Speed", "Throttle", "Brake", "nGear", "RPM", "DRS", "Distance"]].copy()
            telemetry.sort_values("Time", inplace=True)

            # ---- 3) Merge-asof: telemetry ↔ weather ---- #
            enriched = pd.merge_asof(
                telemetry, weather_data, on="Time", direction="nearest", tolerance=pd.Timedelta("60s")
            )

            if enriched.isnull().any().any():
                print(f"⚠️ Missing synced weather — Lap {lap['LapNumber']}, Driver {driver}")
                continue

            # ---- Lap-level metadata for each row ---- #
            enriched["LapNumber"] = lap["LapNumber"]
            enriched["Stint"] = lap["Stint"]
            enriched["Compound"] = lap["Compound"]
            enriched["TrackStatus"] = lap["TrackStatus"]
            enriched["Team"] = lap["Team"]
            enriched["Driver"] = lap["Driver"]
            enriched["LapTime"] = lap["LapTime"].total_seconds()

            if enriched.isnull().any().any():
                print(f"⚠️ NaNs after enrichment — Lap {lap['LapNumber']}, Driver {driver}")
                continue

            # ---- Encode categoricals (shared/global encoders) ---- #
            for col in ENCODE_COLUMNS:
                if col in enriched.columns:
                    encoded_val = encode_value(encoders, col, enriched[col].iloc[0])
                    enriched[col] = encoded_val

            # ---- Per-lap summary (means over rows) ---- #
            summary_vector = enriched.mean(numeric_only=True).to_list()
            save_summary_vector(
                summary_vector,
                COLUMN_NAMES,
                year,
                gp_name,
                session_type,
                driver,
                lap["LapNumber"],
                encoders,
            )

            # ---- Resample lap to fixed steps and collect ---- #
            lap_matrix = resample_lap_by_distance(enriched, steps=150)
            lap_matrices.append(lap_matrix)

        if lap_matrices:
            try:
                driver_cube = np.stack(lap_matrices, axis=0)
                driver_cubes[driver] = driver_cube
                save_driver_cube(driver_cube, year, gp_name, session_type, driver)
            except ValueError as e:
                print(f"❌ Shape mismatch stacking laps for {driver}: {e}")
                continue

    return driver_cubes


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    """Entry point: iterate seasons/events/sessions and clean unseen sessions."""
    encoders = load_encoders()
    cleaned = load_cleaned_state()

    for year in YEARS:
        schedule = get_schedule(year)

        for _, event in schedule.iterrows():
            gp_name = event["EventName"]

            for session_type in SESSION_TYPES:
                if was_cleaned(cleaned, year, gp_name, session_type):
                    print(f"⏭️ Skipping {year} {gp_name} {session_type}")
                    continue

                print(f"🔄 Trying {year} {gp_name} {session_type}")
                success = try_clean_session(year, gp_name, session_type, encoders)

                if success:
                    mark_cleaned(cleaned, year, gp_name, session_type)
                    save_cleaned_state(cleaned)
                    sleep(1)  # avoid rate-limiting
                

if __name__ == "__main__":
    main()
