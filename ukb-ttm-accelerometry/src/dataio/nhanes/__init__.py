"""
NHANES accelerometry data I/O module.

Supports two pathways:
- 80 Hz raw data (2011-2014) for seconds-scale modeling
- Minute-level summaries (2003-2014) for minute/hour/day modeling
"""

__version__ = "0.1.0"

from .parse_80hz import (
    compute_window_params,
    to_windows,
    standardize_windows,
    process_participant_80hz
)

from .parse_minute import (
    download_minute_data,
    parse_minute_summaries,
    create_minute_windows
)

from .merge_clinical import (
    download_clinical_data,
    parse_xpt_file,
    normalize_clinical_variables,
    merge_with_accelerometry
)

from .nonwear import (
    choi_algorithm,
    troiano_algorithm,
    clip_outliers,
    highpass_filter,
    mark_nonwear_windows_from_spans
)

from .datasets_nhanes import (
    NHANES80HzForecastDataset,
    NHANES80HzLabelDataset,
    NHANESMinuteDataset
)

__all__ = [
    # 80 Hz parsing
    'compute_window_params',
    'to_windows',
    'standardize_windows',
    'process_participant_80hz',

    # Minute-level parsing
    'download_minute_data',
    'parse_minute_summaries',
    'create_minute_windows',

    # Clinical data
    'download_clinical_data',
    'parse_xpt_file',
    'normalize_clinical_variables',
    'merge_with_accelerometry',

    # Non-wear detection
    'choi_algorithm',
    'troiano_algorithm',
    'clip_outliers',
    'highpass_filter',
    'mark_nonwear_windows_from_spans',

    # Datasets
    'NHANES80HzForecastDataset',
    'NHANES80HzLabelDataset',
    'NHANESMinuteDataset'
]
