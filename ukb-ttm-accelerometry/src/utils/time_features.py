"""
Time-based feature engineering for accelerometry data.

Extracts cyclical and categorical temporal features useful for downstream tasks.
"""
from typing import Dict, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd


def build_time_features(
    timestamps: Union[np.ndarray, pd.DatetimeIndex],
    include_holiday: bool = False,
    country: str = "UK"
) -> Dict[str, np.ndarray]:
    """
    Build comprehensive time features from timestamps.

    Extracts cyclical encodings (sin/cos) for periodic features and categorical features
    for day of week, weekend, etc.

    Args:
        timestamps: Array of timestamps (datetime64 or DatetimeIndex)
        include_holiday: Whether to include holiday flag (requires holidays package)
        country: Country code for holiday calendar (default: "UK")

    Returns:
        Dictionary containing:
        - "hour_sin": Sine encoding of hour (24-hour period)
        - "hour_cos": Cosine encoding of hour (24-hour period)
        - "minute_sin": Sine encoding of minute within hour
        - "minute_cos": Cosine encoding of minute within hour
        - "day_of_week_sin": Sine encoding of day of week (7-day period)
        - "day_of_week_cos": Cosine encoding of day of week (7-day period)
        - "day_of_week": Day of week as integer (0=Monday, 6=Sunday)
        - "is_weekend": Binary flag for weekend (Sat/Sun)
        - "month_sin": Sine encoding of month (12-month period)
        - "month_cos": Cosine encoding of month (12-month period)
        - "is_holiday": Binary holiday flag (if include_holiday=True)

    Examples:
        >>> timestamps = pd.date_range('2020-01-01', periods=100, freq='H')
        >>> features = build_time_features(timestamps)
        >>> features.keys()
        dict_keys(['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', ...])
        >>> features['hour_sin'].shape
        (100,)
    """
    # Convert to pandas DatetimeIndex for easier feature extraction
    if isinstance(timestamps, np.ndarray):
        if np.issubdtype(timestamps.dtype, np.datetime64):
            dt_index = pd.DatetimeIndex(timestamps)
        else:
            dt_index = pd.to_datetime(timestamps)
    else:
        dt_index = timestamps

    features = {}

    # Extract base temporal components
    hours = dt_index.hour
    minutes = dt_index.minute
    day_of_week = dt_index.dayofweek  # 0=Monday, 6=Sunday
    month = dt_index.month
    day_of_year = dt_index.dayofyear

    # Cyclical encoding of hour (24-hour period)
    # Maps 0-23 hours to [0, 2π]
    hour_angle = 2 * np.pi * hours / 24.0
    features["hour_sin"] = np.sin(hour_angle)
    features["hour_cos"] = np.cos(hour_angle)

    # Cyclical encoding of minute (60-minute period)
    minute_angle = 2 * np.pi * minutes / 60.0
    features["minute_sin"] = np.sin(minute_angle)
    features["minute_cos"] = np.cos(minute_angle)

    # Cyclical encoding of day of week (7-day period)
    dow_angle = 2 * np.pi * day_of_week / 7.0
    features["day_of_week_sin"] = np.sin(dow_angle)
    features["day_of_week_cos"] = np.cos(dow_angle)

    # Cyclical encoding of month (12-month period)
    month_angle = 2 * np.pi * (month - 1) / 12.0  # month is 1-12, normalize to 0-11
    features["month_sin"] = np.sin(month_angle)
    features["month_cos"] = np.cos(month_angle)

    # Cyclical encoding of day of year (365-day period, approximate)
    doy_angle = 2 * np.pi * (day_of_year - 1) / 365.25
    features["day_of_year_sin"] = np.sin(doy_angle)
    features["day_of_year_cos"] = np.cos(doy_angle)

    # Categorical features
    features["day_of_week"] = day_of_week.values.astype(np.int32)
    features["is_weekend"] = (day_of_week >= 5).astype(np.int32)  # Saturday=5, Sunday=6

    # Hour categories for coarser time-of-day grouping
    # Morning (6-12), Afternoon (12-18), Evening (18-22), Night (22-6)
    hour_category = np.zeros(len(hours), dtype=np.int32)
    hour_category[(hours >= 6) & (hours < 12)] = 0   # Morning
    hour_category[(hours >= 12) & (hours < 18)] = 1  # Afternoon
    hour_category[(hours >= 18) & (hours < 22)] = 2  # Evening
    hour_category[(hours >= 22) | (hours < 6)] = 3   # Night
    features["hour_category"] = hour_category

    # Holiday feature (optional)
    if include_holiday:
        features["is_holiday"] = _get_holiday_flags(dt_index, country)

    return features


def _get_holiday_flags(dt_index: pd.DatetimeIndex, country: str = "UK") -> np.ndarray:
    """
    Get binary holiday flags for dates.

    Args:
        dt_index: DatetimeIndex
        country: Country code for holiday calendar

    Returns:
        Binary array where 1 indicates holiday
    """
    try:
        import holidays
    except ImportError:
        raise ImportError(
            "holidays package required for holiday features. "
            "Install with: pip install holidays"
        )

    # Get country holiday calendar
    if country.upper() == "UK":
        country_holidays = holidays.UK()
    elif country.upper() == "US":
        country_holidays = holidays.US()
    else:
        raise ValueError(f"Country {country} not supported. Use 'UK' or 'US'.")

    # Check each date
    is_holiday = np.array([date.date() in country_holidays for date in dt_index], dtype=np.int32)

    return is_holiday


def build_time_features_for_windows(
    window_start_times: np.ndarray,
    window_end_times: Optional[np.ndarray] = None,
    use_midpoint: bool = True,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Build time features for windowed data.

    Args:
        window_start_times: Array of window start timestamps
        window_end_times: Optional array of window end timestamps
        use_midpoint: If True and end_times provided, use midpoint of window
        **kwargs: Additional arguments passed to build_time_features

    Returns:
        Dictionary of time features, one value per window

    Examples:
        >>> starts = pd.date_range('2020-01-01', periods=100, freq='8.192s')
        >>> features = build_time_features_for_windows(starts)
        >>> features['hour_sin'].shape
        (100,)
    """
    if use_midpoint and window_end_times is not None:
        # Use midpoint of window
        if isinstance(window_start_times, pd.DatetimeIndex):
            start_ns = window_start_times.values.astype('int64')
            end_ns = pd.DatetimeIndex(window_end_times).values.astype('int64')
        else:
            start_ns = window_start_times.astype('int64')
            end_ns = window_end_times.astype('int64')

        midpoint_ns = (start_ns + end_ns) // 2
        timestamps = pd.to_datetime(midpoint_ns)
    else:
        timestamps = window_start_times

    return build_time_features(timestamps, **kwargs)


def aggregate_time_features(
    timestamps: np.ndarray,
    aggregation: str = "mean"
) -> np.ndarray:
    """
    Aggregate time features over a window (for variable-length sequences).

    Args:
        timestamps: Array of timestamps within a window
        aggregation: Aggregation method ("mean", "first", "last")

    Returns:
        Aggregated feature vector

    Examples:
        >>> timestamps = pd.date_range('2020-01-01 08:00', periods=819, freq='10ms')
        >>> features = aggregate_time_features(timestamps, aggregation='mean')
    """
    features = build_time_features(timestamps)

    if aggregation == "mean":
        # Average cyclical features (preserves circular properties)
        aggregated = np.array([
            features["hour_sin"].mean(),
            features["hour_cos"].mean(),
            features["day_of_week_sin"].mean(),
            features["day_of_week_cos"].mean(),
            features["is_weekend"].mean(),
        ])
    elif aggregation == "first":
        aggregated = np.array([
            features["hour_sin"][0],
            features["hour_cos"][0],
            features["day_of_week_sin"][0],
            features["day_of_week_cos"][0],
            features["is_weekend"][0],
        ])
    elif aggregation == "last":
        aggregated = np.array([
            features["hour_sin"][-1],
            features["hour_cos"][-1],
            features["day_of_week_sin"][-1],
            features["day_of_week_cos"][-1],
            features["is_weekend"][-1],
        ])
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    return aggregated
