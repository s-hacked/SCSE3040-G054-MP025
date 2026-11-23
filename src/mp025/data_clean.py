"""
Data cleaning and validation module for MP025 Fishing Boat Catch dataset.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import yaml

DEFAULT_BOAT_MAPPING = {
    "country_boat": "country_boat",
    " country_boat": "country_boat",
    "country_boat ": "country_boat",
    "COUNTRY_BOAT": "country_boat",
    "Country_Boat": "country_boat",
    "country boat": "country_boat",
    "cou.": "country_boat",
    "motorised": "motorised",
    " motorised": "motorised",
    "motorised ": "motorised",
    "MOTORISED": "motorised",
    "Motorised": "motorised",
    "mot.": "motorised",
    "trawler": "trawler",
    " trawler": "trawler",
    "trawler ": "trawler",
    "TRAWLER": "trawler",
    "Trawler": "trawler",
    "tra.": "trawler",
}

CANONICAL_BOAT_CATEGORIES = {"country_boat", "motorised", "trawler"}


def load_config(config_path: str = "params.yaml") -> dict:
    """Load configuration parameters from YAML file."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load dataset from CSV file into a pandas DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """Validate that required columns are present in DataFrame."""
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Schema validation failed. Missing columns: {missing}")
    return True


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove identical duplicate records from DataFrame.
    Preserves first occurrence.
    """
    initial_count = len(df)
    cleaned_df = df.drop_duplicates().copy()
    removed_count = initial_count - len(cleaned_df)
    return cleaned_df


def correct_unit_mixup(
    df: pd.DataFrame,
    harbour: str = "North Jetty",
    column: str = "trip_hours",
    conversion_factor: float = 60.0,
) -> pd.DataFrame:
    """
    Correct unit mixup where trip_hours were recorded in minutes for a specific harbour.
    Divides affected values by conversion_factor (60.0).
    """
    cleaned_df = df.copy()
    if column in cleaned_df.columns and "harbour" in cleaned_df.columns:
        mask = cleaned_df["harbour"] == harbour
        cleaned_df.loc[mask, column] = cleaned_df.loc[mask, column] / conversion_factor
    return cleaned_df


def normalize_boat_categories(
    df: pd.DataFrame,
    mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Normalize inconsistent spelling variants of boat types into canonical categories:
    ['country_boat', 'motorised', 'trawler'].
    """
    cleaned_df = df.copy()
    if "boat" not in cleaned_df.columns:
        return cleaned_df

    boat_map = mapping if mapping is not None else DEFAULT_BOAT_MAPPING

    # Clean leading/trailing spaces and map
    cleaned_df["boat"] = cleaned_df["boat"].apply(
        lambda x: boat_map.get(str(x), str(x).strip().lower())
    )

    # Secondary fallback for partial matches
    def canonical_fallback(val: str) -> str:
        v = str(val).strip().lower().replace(" ", "_")
        if v in CANONICAL_BOAT_CATEGORIES:
            return v
        if "cou" in v:
            return "country_boat"
        if "mot" in v:
            return "motorised"
        if "tra" in v:
            return "trawler"
        return v

    cleaned_df["boat"] = cleaned_df["boat"].apply(canonical_fallback)

    unrecognized = set(cleaned_df["boat"]) - CANONICAL_BOAT_CATEGORIES
    if unrecognized:
        raise ValueError(f"Unrecognized boat categories after normalization: {unrecognized}")

    return cleaned_df


def drop_leakage_columns(
    df: pd.DataFrame,
    columns_to_drop: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Drop post-event / leakage columns such as kg_sold_at_auction.
    """
    if columns_to_drop is None:
        columns_to_drop = ["kg_sold_at_auction"]
    cleaned_df = df.drop(columns=columns_to_drop, errors="ignore").copy()
    return cleaned_df


def validate_cleaned_data(df: pd.DataFrame) -> None:
    """Validate data integrity and sanity checks on cleaned dataset."""
    # 1. Null check
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        raise ValueError(f"Cleaned dataset contains unexpected null values: {null_counts[null_counts > 0].to_dict()}")

    # 2. Categorical integrity
    if "boat" in df.columns:
        invalid_boats = set(df["boat"]) - CANONICAL_BOAT_CATEGORIES
        if invalid_boats:
            raise ValueError(f"Invalid boat categories found: {invalid_boats}")

    # 3. Numeric range checks
    if "trip_hours" in df.columns:
        if (df["trip_hours"] <= 0).any():
            raise ValueError("Found non-positive values in trip_hours.")
        if (df["trip_hours"] > 100).any():
            raise ValueError("trip_hours values exceed expected operational limits (>100 hrs).")

    if "distance_from_shore_km" in df.columns:
        if (df["distance_from_shore_km"] < 0).any():
            raise ValueError("Negative values found in distance_from_shore_km.")

    if "wave_height_m" in df.columns:
        if (df["wave_height_m"] < 0).any():
            raise ValueError("Negative values found in wave_height_m.")

    if "catch_kg" in df.columns:
        if (df["catch_kg"] < 0).any():
            raise ValueError("Negative values found in target catch_kg.")


def clean_dataset(
    input_path_or_df: Union[str, Path, pd.DataFrame],
    config_path: str = "params.yaml",
    save_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Full data cleaning pipeline:
    1. Load data
    2. Remove duplicates
    3. Correct North Jetty trip_hours unit mixup
    4. Normalize boat spelling variants
    5. Drop leakage columns (kg_sold_at_auction)
    6. Validate cleaned dataset
    7. Optionally save cleaned dataset to separate path
    """
    config = load_config(config_path)

    # 1. Load data
    if isinstance(input_path_or_df, pd.DataFrame):
        df = input_path_or_df.copy()
    else:
        df = load_data(input_path_or_df)

    cleaning_cfg = config.get("cleaning", {})
    unit_cfg = cleaning_cfg.get("unit_mixup", {})
    target_harbour = unit_cfg.get("target_harbour", "North Jetty")
    target_col = unit_cfg.get("target_column", "trip_hours")
    factor = unit_cfg.get("conversion_factor", 60.0)
    leakage_cols = cleaning_cfg.get("leakage_columns", ["kg_sold_at_auction"])
    boat_map = cleaning_cfg.get("boat_mapping", DEFAULT_BOAT_MAPPING)

    # 2. Remove duplicates
    if cleaning_cfg.get("remove_duplicates", True):
        df = remove_duplicates(df)

    # 3. Correct unit mixup
    df = correct_unit_mixup(
        df,
        harbour=target_harbour,
        column=target_col,
        conversion_factor=factor,
    )

    # 4. Normalize boat categories
    df = normalize_boat_categories(df, mapping=boat_map)

    # 5. Drop leakage columns
    df = drop_leakage_columns(df, columns_to_drop=leakage_cols)

    # 6. Validate cleaned data
    validate_cleaned_data(df)

    # 7. Optionally save output
    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)

    return df


if __name__ == "__main__":
    cfg = load_config("params.yaml")
    raw_path = cfg.get("data", {}).get("raw_train_path", "data/train.csv")
    cleaned_path = cfg.get("data", {}).get("cleaned_train_path", "data/processed/train_cleaned.csv")
    print(f"Cleaning raw data from {raw_path}...")
    cleaned_data = clean_dataset(raw_path, config_path="params.yaml", save_path=cleaned_path)
    print(f"Successfully cleaned data. Rows: {len(cleaned_data)}, Columns: {list(cleaned_data.columns)}")
    print(f"Saved to: {cleaned_path}")
