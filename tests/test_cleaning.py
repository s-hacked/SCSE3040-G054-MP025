"""
Synthetic-data unit tests for data cleaning module (src/mp025/data_clean.py).
Tests run entirely on synthetic mock data without requiring raw CSV files.
"""

import pandas as pd
import pytest
from src.mp025.data_clean import (
    remove_duplicates,
    correct_unit_mixup,
    normalize_boat_categories,
    drop_leakage_columns,
    validate_cleaned_data,
    clean_dataset,
    CANONICAL_BOAT_CATEGORIES,
)


@pytest.fixture
def sample_synthetic_raw_data() -> pd.DataFrame:
    """Create synthetic raw data mimicking the real dataset issues."""
    data = [
        # Normal row
        {
            "record_id": "R001",
            "date": "2026-01-01",
            "trip_hours": 12.0,
            "distance_from_shore_km": 25.0,
            "crew": 4,
            "wave_height_m": 1.5,
            "water_temp_c": 26.0,
            "net_length_m": 800,
            "boat": "country_boat",
            "harbour": "South Jetty",
            "kg_sold_at_auction": 450.0,
            "catch_kg": 455.0,
        },
        # Duplicate of R001
        {
            "record_id": "R001",
            "date": "2026-01-01",
            "trip_hours": 12.0,
            "distance_from_shore_km": 25.0,
            "crew": 4,
            "wave_height_m": 1.5,
            "water_temp_c": 26.0,
            "net_length_m": 800,
            "boat": "country_boat",
            "harbour": "South Jetty",
            "kg_sold_at_auction": 450.0,
            "catch_kg": 455.0,
        },
        # North Jetty row with trip_hours in minutes (unit mixup: 600 mins = 10 hrs)
        {
            "record_id": "R002",
            "date": "2026-01-02",
            "trip_hours": 600.0,
            "distance_from_shore_km": 30.0,
            "crew": 6,
            "wave_height_m": 2.0,
            "water_temp_c": 25.0,
            "net_length_m": 950,
            "boat": " cou. ",
            "harbour": "North Jetty",
            "kg_sold_at_auction": 500.0,
            "catch_kg": 505.0,
        },
        # Boat spelling variant: MOTORISED
        {
            "record_id": "R003",
            "date": "2026-01-03",
            "trip_hours": 8.0,
            "distance_from_shore_km": 15.0,
            "crew": 3,
            "wave_height_m": 1.0,
            "water_temp_c": 27.0,
            "net_length_m": 600,
            "boat": "MOTORISED",
            "harbour": "Lagoon",
            "kg_sold_at_auction": 280.0,
            "catch_kg": 282.0,
        },
        # Boat spelling variant: tra.
        {
            "record_id": "R004",
            "date": "2026-01-04",
            "trip_hours": 15.0,
            "distance_from_shore_km": 40.0,
            "crew": 8,
            "wave_height_m": 2.5,
            "water_temp_c": 24.0,
            "net_length_m": 1200,
            "boat": "tra.",
            "harbour": "Lagoon",
            "kg_sold_at_auction": 620.0,
            "catch_kg": 618.0,
        },
    ]
    return pd.DataFrame(data)


def test_remove_duplicates(sample_synthetic_raw_data):
    """Verify duplicate removal drops duplicate rows correctly."""
    assert len(sample_synthetic_raw_data) == 5
    cleaned = remove_duplicates(sample_synthetic_raw_data)
    assert len(cleaned) == 4
    assert cleaned["record_id"].tolist() == ["R001", "R002", "R003", "R004"]


def test_correct_unit_mixup(sample_synthetic_raw_data):
    """Verify North Jetty trip_hours are divided by 60 while other harbours are untouched."""
    corrected = correct_unit_mixup(
        sample_synthetic_raw_data,
        harbour="North Jetty",
        column="trip_hours",
        conversion_factor=60.0,
    )
    # North Jetty record (originally 600.0) should now be 10.0
    nj_hours = corrected.loc[corrected["harbour"] == "North Jetty", "trip_hours"].iloc[0]
    assert nj_hours == pytest.approx(10.0)

    # South Jetty record (originally 12.0) should remain 12.0
    sj_hours = corrected.loc[corrected["harbour"] == "South Jetty", "trip_hours"].iloc[0]
    assert sj_hours == pytest.approx(12.0)


def test_boat_spelling_normalization():
    """Verify all observed variants normalize into canonical levels."""
    variants = [
        ("cou.", "country_boat"),
        (" cou. ", "country_boat"),
        ("country boat", "country_boat"),
        ("COUNTRY_BOAT", "country_boat"),
        ("Country_Boat", "country_boat"),
        (" country_boat", "country_boat"),
        ("country_boat ", "country_boat"),
        ("mot.", "motorised"),
        ("MOTORISED", "motorised"),
        ("Motorised", "motorised"),
        (" motorised ", "motorised"),
        ("tra.", "trawler"),
        ("TRAWLER", "trawler"),
        ("Trawler", "trawler"),
        (" trawler ", "trawler"),
    ]
    df = pd.DataFrame({"boat": [v[0] for v in variants]})
    normalized = normalize_boat_categories(df)

    for i, (_, expected) in enumerate(variants):
        assert normalized["boat"].iloc[i] == expected
    assert set(normalized["boat"]).issubset(CANONICAL_BOAT_CATEGORIES)


def test_drop_leakage_columns(sample_synthetic_raw_data):
    """Verify leakage column kg_sold_at_auction is removed while catch_kg is preserved."""
    assert "kg_sold_at_auction" in sample_synthetic_raw_data.columns
    assert "catch_kg" in sample_synthetic_raw_data.columns

    cleaned = drop_leakage_columns(sample_synthetic_raw_data, ["kg_sold_at_auction"])
    assert "kg_sold_at_auction" not in cleaned.columns
    assert "catch_kg" in cleaned.columns


def test_validate_cleaned_data_detects_invalid():
    """Verify validate_cleaned_data raises ValueError on invalid values."""
    # Negative trip hours
    invalid_df = pd.DataFrame({
        "trip_hours": [-5.0],
        "distance_from_shore_km": [10.0],
        "wave_height_m": [1.0],
        "boat": ["trawler"],
        "catch_kg": [100.0],
    })
    with pytest.raises(ValueError, match="non-positive"):
        validate_cleaned_data(invalid_df)

    # Invalid boat category
    invalid_boat_df = pd.DataFrame({
        "trip_hours": [5.0],
        "distance_from_shore_km": [10.0],
        "wave_height_m": [1.0],
        "boat": ["spaceship"],
        "catch_kg": [100.0],
    })
    with pytest.raises(ValueError, match="Invalid boat categories"):
        validate_cleaned_data(invalid_boat_df)


def test_clean_dataset_end_to_end(sample_synthetic_raw_data):
    """Verify the full clean_dataset pipeline on synthetic data."""
    cleaned = clean_dataset(sample_synthetic_raw_data)
    # Duplicate dropped: 5 -> 4
    assert len(cleaned) == 4
    # Leakage column dropped
    assert "kg_sold_at_auction" not in cleaned.columns
    # Target preserved
    assert "catch_kg" in cleaned.columns
    # North Jetty unit fixed
    assert cleaned.loc[cleaned["harbour"] == "North Jetty", "trip_hours"].iloc[0] == pytest.approx(10.0)
    # Boat normalized
    assert set(cleaned["boat"]).issubset(CANONICAL_BOAT_CATEGORIES)
