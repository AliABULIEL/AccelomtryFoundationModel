"""
Unit tests for NHANES minute-level processing.
"""

import numpy as np
import pandas as pd
import pytest

from src.dataio.nhanes.parse_minute import (
    create_minute_windows,
    compute_wear_stats
)


def create_synthetic_minute_dataframe(n_participants=2, minutes_per_participant=2000):
    """Create synthetic minute-level dataframe for testing."""
    np.random.seed(42)

    data_list = []

    for pid in range(n_participants):
        timestamps = pd.date_range(
            '2020-01-01',
            periods=minutes_per_participant,
            freq='1min'
        )

        axis_summaries = np.random.exponential(scale=50, size=minutes_per_participant)
        wear_flag = np.random.rand(minutes_per_participant) > 0.1  # 90% wear

        participant_df = pd.DataFrame({
            'participant_id': str(pid),
            'timestamp': timestamps,
            'axis_summaries': axis_summaries,
            'wear_flag': wear_flag,
            'cycle': '2011-2012'
        })

        data_list.append(participant_df)

    return pd.concat(data_list, ignore_index=True)


def test_create_minute_windows_shapes():
    """Test that minute windows have correct shapes."""
    # Create synthetic data
    minute_df = create_synthetic_minute_dataframe(n_participants=2, minutes_per_participant=2000)

    # Create windows
    context_length = 128  # Smaller for testing
    prediction_length = 32

    X_context, Y_future, windows_meta = create_minute_windows(
        minute_df,
        context_length=context_length,
        prediction_length=prediction_length,
        min_wear_fraction=0.8
    )

    # Check shapes
    assert X_context.shape[1:] == (1, context_length), \
        f"Expected X shape (B, 1, {context_length}), got {X_context.shape}"

    assert Y_future.shape[1:] == (1, prediction_length), \
        f"Expected Y shape (B, 1, {prediction_length}), got {Y_future.shape}"

    assert X_context.shape[0] == Y_future.shape[0], \
        "X and Y must have same batch size"

    assert len(windows_meta) == X_context.shape[0], \
        "Metadata length must match batch size"


def test_create_minute_windows_nonempty():
    """Test that windows are created successfully."""
    # Create data with high wear
    minute_df = create_synthetic_minute_dataframe(n_participants=2, minutes_per_participant=2000)

    # Force high wear
    minute_df['wear_flag'] = True

    # Create windows
    X_context, Y_future, windows_meta = create_minute_windows(
        minute_df,
        context_length=128,
        prediction_length=32,
        min_wear_fraction=0.8
    )

    # Should have many windows
    assert len(X_context) > 0, "No windows created"


def test_create_minute_windows_wear_filtering():
    """Test that windows with low wear are filtered out."""
    # Create data
    minute_df = create_synthetic_minute_dataframe(n_participants=1, minutes_per_participant=500)

    # Set low wear
    minute_df['wear_flag'] = False

    # Create windows
    X_context, Y_future, windows_meta = create_minute_windows(
        minute_df,
        context_length=100,
        prediction_length=20,
        min_wear_fraction=0.8
    )

    # Should have zero windows due to low wear
    assert len(X_context) == 0, "Windows should be filtered due to low wear"


def test_create_minute_windows_metadata():
    """Test metadata columns."""
    minute_df = create_synthetic_minute_dataframe(n_participants=2, minutes_per_participant=500)
    minute_df['wear_flag'] = True

    X_context, Y_future, windows_meta = create_minute_windows(
        minute_df,
        context_length=100,
        prediction_length=20,
        min_wear_fraction=0.8
    )

    # Check metadata columns
    assert 'participant_id' in windows_meta.columns
    assert 'start_timestamp' in windows_meta.columns

    # Check participant IDs are valid
    valid_pids = {'0', '1'}
    assert all(windows_meta['participant_id'].isin(valid_pids))


def test_compute_wear_stats():
    """Test wear statistics computation."""
    minute_df = create_synthetic_minute_dataframe(n_participants=2, minutes_per_participant=1000)

    wear_stats = compute_wear_stats(minute_df)

    # Check columns
    assert 'participant_id' in wear_stats.columns
    assert 'total_minutes' in wear_stats.columns
    assert 'wear_minutes' in wear_stats.columns
    assert 'wear_fraction' in wear_stats.columns

    # Check values
    assert len(wear_stats) == 2  # 2 participants

    # Each participant has 1000 minutes
    assert all(wear_stats['total_minutes'] == 1000)

    # Wear fraction should be between 0 and 1
    assert all((wear_stats['wear_fraction'] >= 0) & (wear_stats['wear_fraction'] <= 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
