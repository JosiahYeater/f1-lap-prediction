"""Dataset preparation utilities for F1 lap-time prediction.

This module:
- Aggregates per-lap summary metadata and tracks preprocessing status.
- Computes normalization statistics from raw driver cubes.
- Builds deterministic one-hot vector encoders from global integer encoders.
- Preprocesses individual laps (normalize numeric features, one-hot categoricals).
- Saves preprocessed laps into train/val/test splits as .npy files.

Directory conventions
---------------------
data/raw/{year/gp/session}/{driver}.npy
    Driver-specific cube of laps (L x T x F_raw) from the cleaning pipeline.

data/processed/{year/gp/session}/{driver}.json
    Per-lap summaries (used to enumerate laps and build metadata keys).

data/preprocessed/{split}/lapXXXXX.npy
    Final model-ready laps (T x F_processed), with target LapTime in the last column.

data/metadata/preprocessing_metadata.json
    Key-value status map of laps: "unprocessed" | "train" | "val" | "test" | "skipped".

data/metadata/global_encoders.json
    Integer ID encoders for raw categoricals (e.g., {"Driver": {"VER": 0, ...}}).

data/metadata/vector_encoders.json
    One-hot encoders derived from global encoders (e.g., {"Driver": {"VER": [1,0,...], ...}}).

data/metadata/normalization_stats.json
    Mean/std stats for numeric columns (used for z-score normalization).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PREPROCESSED_DIR = ROOT_DIR / "data" / "preprocessed"
METADATA_PATH = ROOT_DIR / "data" / "metadata" / "preprocessing_metadata.json"
ENCODER_PATH = ROOT_DIR / "data" / "metadata" / "global_encoders.json"
NORMSTATS_PATH = ROOT_DIR / "data" / "metadata" / "normalization_stats.json"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
VECTOR_ENCODER_PATH = ROOT_DIR / "data" / "metadata" / "vector_encoders.json"

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# Column order for raw matrices produced by the cleaning pipeline
COLUMN_NAMES = [
    "Speed",           # km/h
    "Throttle",        # 0–100 %
    "Brake",           # 0 or 1
    "nGear",           # int (1–8), may need encoding
    "RPM",             # engine revolutions per minute
    "DRS",             # 0 = off, 1 = on, 2 = active

    "AirTemp",         # °C
    "Humidity",        # %
    "Pressure",        # hPa
    "Rainfall",        # mm/hr
    "TrackTemp",       # °C
    "WindDirection",   # degrees
    "WindSpeed",       # km/h

    "LapNumber",       # int
    "Stint",           # int

    "Compound",        # string (e.g. "SOFT") → encoded integer in raw
    "TrackStatus",     # int or flag code
    "Team",            # string → encoded integer in raw
    "Driver",          # string → encoded integer in raw
    "LapTime",         # seconds (float) — same across all rows in a lap
]

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for Python and NumPy to ensure reproducible results.

    Args:
        seed (int): The seed value to use for reproducibility. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)

# -----------------------------------------------------------------------------
# Normalization statistics
# -----------------------------------------------------------------------------
def load_normstats() -> Dict[str, Dict[str, float]]:
    """Load normalization stats (mean/std per numeric column) if present."""
    if NORMSTATS_PATH.exists():
        with open(NORMSTATS_PATH, "r") as f:
            return json.load(f)
    return {}


def save_normstats(normstats: Dict[str, Dict[str, float]]) -> None:
    """Persist normalization stats to disk."""
    NORMSTATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NORMSTATS_PATH, "w") as f:
        json.dump(normstats, f, indent=2)


def calculate_norm_stats(
    raw_dir: Path = RAW_DATA_DIR,
    column_names: List[str] = COLUMN_NAMES,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Compute mean/std for numeric columns from raw driver cubes.

    Scans all .npy cubes under `raw_dir`, vertically stacks laps across drivers,
    and computes column-wise mean and std. Non-numeric/categorical columns are
    dropped from the resulting statistics.

    Parameters
    ----------
    raw_dir : Path
        Root path to the raw driver cubes.
    column_names : list[str]
        Column names aligned to raw matrices.

    Returns
    -------
    dict | None
        Mapping {column: {"mean": m, "std": s}}, or None if no data found.
    """
    if NORMSTATS_PATH.exists():
        print("✅ Norm stats already exist, loading...")
        return load_normstats()

    all_values = []

    for file in raw_dir.rglob("*.npy"):
        try:
            cube = np.load(file)
            if cube.ndim != 3:
                print(f"⚠️ Skipping non-cube file: {file}")
                continue

            for lap in cube:
                all_values.append(lap)
        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")

    if not all_values:
        print("❌ No valid lap data found.")
        return None

    # ---- shape: (total_timesteps_across_laps, num_features_raw) ---- #
    stacked = np.vstack(all_values)
    means = stacked.mean(axis=0)
    stds = stacked.std(axis=0)

    norm_stats = {
        col: {"mean": float(m), "std": float(s) if s > 0 else 1.0}
        for col, m, s in zip(column_names, means, stds)
    }

    # ---- Drop non-numeric/categorical fields; these are handled via one-hot later ---- #
    del norm_stats["Compound"]
    del norm_stats["TrackStatus"]
    del norm_stats["Team"]
    del norm_stats["Driver"]
    del norm_stats["LapTime"]

    save_normstats(norm_stats)
    print(f"✅ Saved norm stats to {NORMSTATS_PATH}")
    return norm_stats


# -----------------------------------------------------------------------------
# One-hot vector encoders (derived from global integer encoders)
# -----------------------------------------------------------------------------
def load_encoders() -> Dict[str, Dict[str, int]]:
    """Load global integer encoders (e.g., {'Driver': {'VER': 0, ...}})."""
    if ENCODER_PATH.exists():
        with open(ENCODER_PATH, "r") as f:
            return json.load(f)
    return {}


def build_vector_encoders(encoders: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, List[int]]]:
    """Build deterministic one-hot encoders from global integer encoders.

    For each category (e.g., 'Driver'), construct a mapping from label to a
    one-hot vector based on the integer ID assignment. The insertion order of
    the resulting dict follows the integer index (0..N-1) for stability.

    Returns existing encoders from disk if already built.
    """
    if VECTOR_ENCODER_PATH.exists():
        print("✅ Vector encoders already exist, loading...")
        return load_vector_encoders()

    vector_encoders: Dict[str, Dict[str, List[int]]] = {}

    for category, mapping in encoders.items():
        size = len(mapping)
        reverse_mapping = {v: k for k, v in mapping.items()}  # {idx: label}
        vector_encoders[category] = {}

        for idx in range(size):
            one_hot = [0] * size
            one_hot[idx] = 1
            label = reverse_mapping[idx]
            vector_encoders[category][label] = one_hot

    save_vector_encoders(vector_encoders)
    print(f"✅ Saved vector encoders to {VECTOR_ENCODER_PATH}")
    return vector_encoders


def save_vector_encoders(vector_encoders: Dict[str, Dict[str, List[int]]]) -> None:
    """Persist one-hot vector encoders to disk."""
    VECTOR_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VECTOR_ENCODER_PATH, "w") as f:
        json.dump(vector_encoders, f, indent=2)


def load_vector_encoders() -> Dict[str, Dict[str, List[int]]]:
    """Load previously built one-hot vector encoders if present."""
    if VECTOR_ENCODER_PATH.exists():
        with open(VECTOR_ENCODER_PATH, "r") as f:
            return json.load(f)
    return {}


# -----------------------------------------------------------------------------
# Metadata management
# -----------------------------------------------------------------------------
def load_metadata() -> Dict[str, Dict[str, str]]:
    """Load preprocessing metadata (lap status map) if present."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return {}


def save_metadata(metadata: Dict[str, Dict[str, str]]) -> None:
    """Persist preprocessing metadata to disk."""
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)


def update_lap_metadata() -> Dict[str, Dict[str, str]]:
    """Ensure all laps found in processed summaries exist in the metadata map.

    Scans the processed summaries to create or fill entries like:
        "2019/Australian_Grand_Prix/Q/VER(5)": {"status": "unprocessed"}

    Returns
    -------
    dict
        Updated metadata dictionary.
    """
    metadata = load_metadata()
    summary_files = list(PROCESSED_DATA_DIR.glob("*/*/*/*.json"))

    for file in summary_files:
        with open(file, "r") as f:
            laps = json.load(f)

        # ---- Parse path components: year/gp/session/driver.json ---- # 
        parts = file.relative_to(PROCESSED_DATA_DIR).parts
        year, gp, session, driver_json = parts
        driver = driver_json.replace(".json", "")

        for lap in laps:
            lap_number = int(lap["lap_number"])
            key = f"{year}/{gp}/{session}/{driver}({lap_number})"
            if key not in metadata:
                metadata[key] = {"status": "unprocessed"}

    save_metadata(metadata)
    print(f"✅ Metadata updated with {len(metadata)} total laps")
    return metadata


# -----------------------------------------------------------------------------
# Lap selection
# -----------------------------------------------------------------------------
def select_unprocessed_lap(metadata: Dict[str, Dict[str, str]]) -> Optional[str]:
    """Randomly choose an unprocessed lap key from metadata."""
    unprocessed = [k for k, v in metadata.items() if v["status"] == "unprocessed"]
    if not unprocessed:
        return None
    return random.choice(unprocessed)


# -----------------------------------------------------------------------------
# Preprocessing: per-lap normalization and one-hot encoding
# -----------------------------------------------------------------------------
def preprocess_lap(
    lap_matrix: np.ndarray,
    column_names: List[str],
    norm_stats: Dict[str, Dict[str, float]],
    vector_encoders: Dict[str, Dict[str, List[int]]],
) -> np.ndarray:
    """Preprocess a single lap matrix.

    For each row:
      - Z-score normalize numeric columns using `norm_stats`.
      - Replace categorical integer IDs with their one-hot vectors from
        `vector_encoders`.
      - Append the (constant) LapTime value at the end of the row.

    Parameters
    ----------
    lap_matrix : np.ndarray
        Raw lap (T x F_raw) from the cleaning pipeline.
    column_names : list[str]
        Column names for the raw lap.
    norm_stats : dict
        Mapping {column: {"mean": m, "std": s}} for numeric fields.
    vector_encoders : dict
        Mapping {category: {label: one_hot}} derived from global encoders.

    Returns
    -------
    np.ndarray
        Processed lap (T x F_processed), dtype float32.
    """
    processed_rows: List[List[float]] = []

    for row in lap_matrix:
        new_row: List[float] = []

        for i, col_name in enumerate(column_names):
            if col_name == "LapTime":
                continue  # LapTime is appended once at the end of each row

            val = row[i]

            if col_name in norm_stats:
                mean = norm_stats[col_name]["mean"]
                std = norm_stats[col_name]["std"]
                normalized = (val - mean) / std
                new_row.append(normalized)

            elif col_name in vector_encoders:
                # Convert integer ID back to label order used in vector encoders
                try:
                    label = list(vector_encoders[col_name].keys())[int(val)]
                    one_hot = vector_encoders[col_name][label]
                    new_row.extend(one_hot)
                except Exception as e:
                    print(f"⚠️ Vector encoding failed for {col_name}={val}: {e}")
                    continue

            else:
                # If a column isn't in norm_stats or vector_encoders, keep raw value
                new_row.append(val)

        # Append constant LapTime target for this row
        lap_time = row[column_names.index("LapTime")]
        new_row.append(lap_time)

        processed_rows.append(new_row)

    return np.array(processed_rows, dtype=np.float32)


# -----------------------------------------------------------------------------
# Save logic
# -----------------------------------------------------------------------------
def save_preprocessed_lap(lap_id: str, split: str, processed_lap: np.ndarray) -> None:
    """Save a preprocessed lap array under the given split and ID."""
    out_path = PREPROCESSED_DIR / split
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / f"{lap_id}.npy", processed_lap)
    print(f"✅ Saved: {split}/{lap_id}.npy")


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main() -> None:
    """Enumerate laps, preprocess them, and save into train/val/test splits."""
    counter = 0

    metadata = update_lap_metadata()

    global_encoders = load_encoders()
    vector_encoders = build_vector_encoders(global_encoders)

    norm_stats = calculate_norm_stats(RAW_DATA_DIR, COLUMN_NAMES)

    while True:
        lap_key = select_unprocessed_lap(metadata)
        if not lap_key:
            print("✅ All laps processed.")
            break

        print(f"🔄 Processing {lap_key}")
        year, gp, session_type, driver_lap = lap_key.split("/")
        driver, lap_number = driver_lap.split("(")
        lap_number = int(lap_number[:-1])

        lap_path = PROCESSED_DATA_DIR / year / gp / session_type / f"{driver}.json"

        with open(lap_path, "r") as f:
            lap_data = json.load(f)

        lap_numbers = [lap["lap_number"] for lap in lap_data]
        if lap_number not in lap_numbers:
            print(f"⚠️ Lap number {lap_number} not found in {lap_path}")
            metadata[lap_key] = {"status": "skipped"}
            save_metadata(metadata)
            continue

        lap_index = lap_numbers.index(lap_number)

        cube_path = RAW_DATA_DIR / year / gp / session_type / f"{driver}.npy"
        if not cube_path.exists():
            print(f"⚠️ Missing lap file: {cube_path}")
            metadata[lap_key] = {"status": "skipped"}
            save_metadata(metadata)
            continue

        try:
            cube = np.load(cube_path)
            lap_matrix = cube[lap_index]  # Extract single lap (T x F_raw)

            processed = preprocess_lap(lap_matrix, COLUMN_NAMES, norm_stats or {}, vector_encoders)

            # ---- Assign lap_id and split ---- #
            lap_id = f"lap{counter:05d}"
            split = random.choices(
                population=["train", "val", "test"],
                weights=[SPLIT_RATIOS["train"], SPLIT_RATIOS["val"], SPLIT_RATIOS["test"]],
                k=1,
            )[0]

            save_preprocessed_lap(lap_id, split, processed)
            metadata[lap_key] = {"lap_id": lap_id, "status": split}
            save_metadata(metadata)
            counter += 1

        except Exception as e:
            print(f"❌ Failed to process {lap_key}: {e}")


if __name__ == "__main__":
    set_seed(42) # Ensure reproducibility
    main()
