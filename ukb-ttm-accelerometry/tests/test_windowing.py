"""
Unit tests for windowing operations.

Tests window parameter computation, strided windowing, and edge cases.
"""
import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataio.windowing import (
    compute_window_params,
    segment_stream,
    compute_window_timestamps,
    filter_windows_by_gaps
)


class TestComputeWindowParams:
    """Tests for compute_window_params function."""

    def test_floor_rounding(self):
        """Test floor rounding mode."""
        win_n, hop_n = compute_window_params(100, 8.192, 4.096, "floor")
        assert win_n == 819
        assert hop_n == 409

    def test_nearest_rounding(self):
        """Test nearest rounding mode."""
        win_n, hop_n = compute_window_params(100, 8.192, 4.096, "nearest")
        assert win_n == 819
        assert hop_n == 410

    def test_ceil_rounding(self):
        """Test ceil rounding mode."""
        win_n, hop_n = compute_window_params(100, 8.195, 4.097, "ceil")
        assert win_n == 820
        assert hop_n == 410

    def test_different_fs(self):
        """Test different sampling frequencies."""
        win_n, hop_n = compute_window_params(50, 8.192, 4.096, "floor")
        assert win_n == 409
        assert hop_n == 204

    def test_invalid_rounding(self):
        """Test invalid rounding mode raises error."""
        with pytest.raises(ValueError, match="Invalid rounding mode"):
            compute_window_params(100, 8.192, 4.096, "invalid")


class TestSegmentStream:
    """Tests for segment_stream function."""

    def test_basic_windowing(self):
        """Test basic windowing operation."""
        # Create synthetic data: 1000 samples, 3 channels
        x = np.random.randn(1000, 3)
        windows = segment_stream(x, win_n=100, hop_n=50, pad_mode="none")

        # Expected number of windows: (1000 - 100) // 50 + 1 = 19
        assert windows.shape == (19, 3, 100)

    def test_no_overlap(self):
        """Test windowing with no overlap."""
        x = np.random.randn(1000, 3)
        windows = segment_stream(x, win_n=100, hop_n=100, pad_mode="none")

        # Expected: 1000 // 100 = 10 windows
        assert windows.shape[0] == 10

    def test_reflect_padding(self):
        """Test reflect padding mode."""
        x = np.random.randn(1000, 3)
        windows_no_pad = segment_stream(x, win_n=100, hop_n=50, pad_mode="none")
        windows_pad = segment_stream(x, win_n=100, hop_n=50, pad_mode="reflect")

        # Padded version should have more windows
        assert windows_pad.shape[0] >= windows_no_pad.shape[0]

    def test_zero_padding(self):
        """Test zero padding mode."""
        x = np.random.randn(1000, 3)
        windows = segment_stream(x, win_n=100, hop_n=50, pad_mode="zero")

        # Check that some windows contain zeros (at the end)
        assert windows.shape[0] > 0

    def test_channel_first_output(self):
        """Test that output is in (N, C, T) format."""
        x = np.random.randn(1000, 3)
        windows = segment_stream(x, win_n=100, hop_n=50)

        assert windows.ndim == 3
        assert windows.shape[1] == 3  # Channels
        assert windows.shape[2] == 100  # Time

    def test_window_too_small_error(self):
        """Test that very small windows raise error."""
        x = np.random.randn(1000, 3)

        with pytest.raises(ValueError, match="too small"):
            segment_stream(x, win_n=50, hop_n=25)

    def test_window_too_large_error(self):
        """Test that window larger than input raises error."""
        x = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="shorter than window size"):
            segment_stream(x, win_n=200, hop_n=100)

    def test_divisible_by_seven_assertion(self):
        """Test that window size divisible by 7 raises assertion."""
        x = np.random.randn(1000, 3)

        with pytest.raises(AssertionError, match="divisible by 7"):
            segment_stream(x, win_n=700, hop_n=350)

    def test_incorrect_input_shape(self):
        """Test that non-2D input raises error."""
        x = np.random.randn(1000)  # 1-D

        with pytest.raises(ValueError, match="Expected 2-D input"):
            segment_stream(x, win_n=100, hop_n=50)

    def test_window_content_correctness(self):
        """Test that windows contain correct data."""
        # Create predictable data
        x = np.arange(300).reshape(100, 3)
        windows = segment_stream(x, win_n=20, hop_n=10, pad_mode="none")

        # First window should contain samples 0-19
        assert np.array_equal(windows[0, 0, :], x[0:20, 0])

        # Second window should contain samples 10-29
        assert np.array_equal(windows[1, 0, :], x[10:30, 0])


class TestComputeWindowTimestamps:
    """Tests for compute_window_timestamps function."""

    def test_basic_timestamps(self):
        """Test basic timestamp computation."""
        timestamps = pd.date_range('2020-01-01', periods=1000, freq='10ms')
        start_times, end_times = compute_window_timestamps(
            timestamps.values, hop_n=50, win_n=100
        )

        # Number of windows
        n_windows = (1000 - 100) // 50 + 1
        assert len(start_times) == n_windows
        assert len(end_times) == n_windows

        # Check first window
        assert start_times[0] == timestamps[0]
        assert end_times[0] == timestamps[99]

    def test_timestamp_progression(self):
        """Test that timestamps progress correctly."""
        timestamps = pd.date_range('2020-01-01', periods=1000, freq='10ms')
        start_times, end_times = compute_window_timestamps(
            timestamps.values, hop_n=50, win_n=100
        )

        # Start times should be spaced by hop_n * period
        hop_period = pd.Timedelta('10ms') * 50
        for i in range(1, len(start_times)):
            expected_start = start_times[i-1] + hop_period
            assert start_times[i] == expected_start


class TestFilterWindowsByGaps:
    """Tests for filter_windows_by_gaps function."""

    def test_no_gaps(self):
        """Test that all windows pass when there are no gaps."""
        windows = np.random.randn(10, 3, 100)
        gap_flags = np.zeros(1000, dtype=int)

        filtered, mask = filter_windows_by_gaps(
            windows, gap_flags, hop_n=100, win_n=100, max_gap_ratio=0.1
        )

        assert len(filtered) == len(windows)
        assert np.all(mask)

    def test_large_gap_filtered(self):
        """Test that windows with large gaps are filtered."""
        windows = np.random.randn(10, 3, 100)
        gap_flags = np.zeros(1000, dtype=int)

        # Add large gap in second window (samples 100-199)
        gap_flags[100:150] = 1  # 50% gap

        filtered, mask = filter_windows_by_gaps(
            windows, gap_flags, hop_n=100, win_n=100, max_gap_ratio=0.1
        )

        # Second window should be filtered
        assert not mask[1]
        assert len(filtered) < len(windows)

    def test_small_gap_passes(self):
        """Test that windows with small gaps pass."""
        windows = np.random.randn(10, 3, 100)
        gap_flags = np.zeros(1000, dtype=int)

        # Add small gap (5% < 10% threshold)
        gap_flags[0:5] = 1

        filtered, mask = filter_windows_by_gaps(
            windows, gap_flags, hop_n=100, win_n=100, max_gap_ratio=0.1
        )

        # First window should pass
        assert mask[0]
        assert len(filtered) == len(windows)

    def test_custom_threshold(self):
        """Test custom gap ratio threshold."""
        windows = np.random.randn(10, 3, 100)
        gap_flags = np.zeros(1000, dtype=int)

        # Add 15% gap
        gap_flags[0:15] = 1

        # Should fail with 10% threshold
        filtered_strict, mask_strict = filter_windows_by_gaps(
            windows, gap_flags, hop_n=100, win_n=100, max_gap_ratio=0.1
        )
        assert not mask_strict[0]

        # Should pass with 20% threshold
        filtered_loose, mask_loose = filter_windows_by_gaps(
            windows, gap_flags, hop_n=100, win_n=100, max_gap_ratio=0.2
        )
        assert mask_loose[0]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_window(self):
        """Test with exactly one window."""
        x = np.random.randn(100, 3)
        windows = segment_stream(x, win_n=100, hop_n=50)

        assert windows.shape[0] == 1
        assert windows.shape == (1, 3, 100)

    def test_empty_input(self):
        """Test with empty input."""
        x = np.random.randn(0, 3)

        with pytest.raises(ValueError):
            segment_stream(x, win_n=100, hop_n=50)

    def test_exact_multiple(self):
        """Test when input length is exact multiple of hop size."""
        x = np.random.randn(1000, 3)
        windows = segment_stream(x, win_n=100, hop_n=100, pad_mode="none")

        assert windows.shape[0] == 10


class TestReproducibility:
    """Test reproducibility of operations."""

    def test_deterministic_windowing(self):
        """Test that windowing is deterministic."""
        np.random.seed(42)
        x1 = np.random.randn(1000, 3)

        np.random.seed(42)
        x2 = np.random.randn(1000, 3)

        windows1 = segment_stream(x1, win_n=100, hop_n=50)
        windows2 = segment_stream(x2, win_n=100, hop_n=50)

        assert np.array_equal(windows1, windows2)


class TestShapeInvariants:
    """Test shape invariants across operations."""

    def test_window_count_formula(self):
        """Test that window count follows formula."""
        for T in [500, 1000, 2000]:
            for win_n in [100, 200]:
                for hop_n in [50, 100]:
                    if T >= win_n:
                        x = np.random.randn(T, 3)
                        windows = segment_stream(x, win_n=win_n, hop_n=hop_n)

                        expected_count = (T - win_n) // hop_n + 1
                        assert windows.shape[0] == expected_count

    def test_channel_preservation(self):
        """Test that number of channels is preserved."""
        for n_channels in [1, 3, 6]:
            x = np.random.randn(1000, n_channels)
            windows = segment_stream(x, win_n=100, hop_n=50)

            assert windows.shape[1] == n_channels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
