# FILE: ukb_ttm_accel/data/clinical_metadata.py

"""
Clinical metadata handling for UKB accelerometry data.

Functions to:
- Load clinical data from CSV
- Map participant IDs to labels and metadata
- Create train/val/test splits
- Handle BMI classification and other clinical labels
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Any, Dict
from sklearn.model_selection import train_test_split


def load_clinical_data(
    csv_path: str,
    required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load clinical metadata from CSV file.

    Expected minimum columns: eid (participant ID)
    Common additional columns: age, sex, bmi, height, weight, etc.

    Args:
        csv_path: Path to clinical CSV file
        required_columns: List of required column names (default: ['eid'])

    Returns:
        DataFrame with clinical data

    Example:
        >>> clinical_df = load_clinical_data("clinical.csv")
        >>> print(clinical_df.columns)
        Index(['eid', 'age', 'sex', 'bmi', 'height', 'weight', ...])
    """
    if required_columns is None:
        required_columns = ['eid']

    df = pd.read_csv(csv_path)

    # Validate required columns exist
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Clinical CSV missing required columns: {missing_cols}")

    # Set eid as index for easy lookup
    if 'eid' in df.columns:
        df.set_index('eid', inplace=True)

    return df


def build_label_and_metadata(
    eid: int,
    clinical_df: pd.DataFrame,
    label_type: str,
    metadata_columns: List[str]
) -> Tuple[Any, np.ndarray]:
    """
    Extract label and metadata for a participant.

    Args:
        eid: Participant ID
        clinical_df: Clinical DataFrame (indexed by eid)
        label_type: Type of label to extract:
                   - "bmi_class": BMI classification (0=underweight, 1=normal, 2=overweight, 3=obese)
                   - "bmi": Raw BMI value (regression)
                   - "age_group": Age classification (0=young, 1=middle, 2=old)
                   - "sex": Binary sex (0=female, 1=male, -1=other/unknown)
                   - "hypertension": Binary (0=no, 1=yes)
                   - "diabetes": Binary (0=no, 1=yes)
        metadata_columns: List of metadata column names to extract

    Returns:
        Tuple of (label, metadata_vector)
        - label: int/float/str depending on label_type
        - metadata_vector: np.ndarray of float values

    Example:
        >>> clinical_df = load_clinical_data("clinical.csv")
        >>> label, meta = build_label_and_metadata(
        ...     eid=1001,
        ...     clinical_df=clinical_df,
        ...     label_type="bmi_class",
        ...     metadata_columns=["age", "sex", "bmi"]
        ... )
        >>> print(label, meta)
        2 [45.0, 1.0, 27.3]
    """
    if eid not in clinical_df.index:
        raise ValueError(f"Participant {eid} not found in clinical data")

    participant_data = clinical_df.loc[eid]

    # Extract label based on label_type
    label = _extract_label(participant_data, label_type)

    # Extract metadata
    metadata = []
    for col in metadata_columns:
        if col not in participant_data:
            raise ValueError(f"Metadata column '{col}' not found for participant {eid}")

        value = participant_data[col]

        # Handle missing values
        if pd.isna(value):
            value = 0.0  # Assumption: fill missing with 0 (can be improved with imputation)

        # Convert to float
        try:
            value = float(value)
        except (ValueError, TypeError):
            # Handle categorical values (e.g., sex as 'M'/'F')
            value = _encode_categorical(col, value)

        metadata.append(value)

    metadata_vector = np.array(metadata, dtype=np.float32)

    return label, metadata_vector


def _extract_label(participant_data: pd.Series, label_type: str) -> Any:
    """
    Extract label from participant data based on label type.

    Args:
        participant_data: Series with participant's clinical data
        label_type: Type of label to extract

    Returns:
        Label value (type depends on label_type)
    """
    if label_type == "bmi_class":
        # BMI classification
        # Assumption: Standard WHO BMI categories
        # < 18.5: underweight (0)
        # 18.5-24.9: normal (1)
        # 25-29.9: overweight (2)
        # >= 30: obese (3)
        if 'bmi' not in participant_data:
            raise ValueError("BMI column required for bmi_class label")

        bmi = participant_data['bmi']

        if pd.isna(bmi):
            return -1  # Missing label

        if bmi < 18.5:
            return 0
        elif bmi < 25.0:
            return 1
        elif bmi < 30.0:
            return 2
        else:
            return 3

    elif label_type == "bmi":
        # Raw BMI value for regression
        bmi = participant_data.get('bmi', np.nan)
        return float(bmi) if not pd.isna(bmi) else -1.0

    elif label_type == "age_group":
        # Age classification
        # Assumption: < 40: young (0), 40-60: middle (1), > 60: old (2)
        if 'age' not in participant_data:
            raise ValueError("Age column required for age_group label")

        age = participant_data['age']

        if pd.isna(age):
            return -1

        if age < 40:
            return 0
        elif age < 60:
            return 1
        else:
            return 2

    elif label_type == "sex":
        # Binary sex classification
        # Assumption: 0=female, 1=male, -1=other/unknown
        sex = participant_data.get('sex', -1)

        if pd.isna(sex):
            return -1

        # Handle various encodings
        if isinstance(sex, str):
            sex_lower = sex.lower()
            if sex_lower in ['f', 'female', '0']:
                return 0
            elif sex_lower in ['m', 'male', '1']:
                return 1
            else:
                return -1
        else:
            # Numeric encoding
            return int(sex) if sex in [0, 1] else -1

    elif label_type == "hypertension":
        # Binary hypertension label
        # Assumption: Column named 'hypertension' with 0/1 or boolean
        hypertension = participant_data.get('hypertension', -1)

        if pd.isna(hypertension):
            return -1

        return int(bool(hypertension))

    elif label_type == "diabetes":
        # Binary diabetes label
        # Assumption: Column named 'diabetes' with 0/1 or boolean
        diabetes = participant_data.get('diabetes', -1)

        if pd.isna(diabetes):
            return -1

        return int(bool(diabetes))

    else:
        raise ValueError(f"Unsupported label_type: {label_type}")


def _encode_categorical(column_name: str, value: Any) -> float:
    """
    Encode categorical values to numeric.

    Args:
        column_name: Name of the column
        value: Categorical value

    Returns:
        Numeric encoding
    """
    # Handle sex encoding
    if column_name.lower() == 'sex':
        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower in ['f', 'female']:
                return 0.0
            elif value_lower in ['m', 'male']:
                return 1.0

    # Default: try to convert to float, return 0.0 if fails
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def create_train_val_test_split(
    participant_ids: List[int],
    clinical_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_column: Optional[str] = None,
    random_state: int = 42
) -> Dict[str, List[int]]:
    """
    Create train/val/test split of participant IDs.

    Args:
        participant_ids: List of participant IDs
        clinical_df: Clinical DataFrame (for stratification)
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        stratify_column: Optional column name for stratified splitting
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to participant ID lists

    Example:
        >>> participant_ids = [1001, 1002, 1003, ..., 2000]
        >>> splits = create_train_val_test_split(
        ...     participant_ids,
        ...     clinical_df,
        ...     stratify_column='sex'
        ... )
        >>> print(len(splits['train']), len(splits['val']), len(splits['test']))
        700 150 150
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Prepare stratification labels if requested
    stratify = None
    if stratify_column is not None:
        stratify = [
            clinical_df.loc[pid, stratify_column]
            if pid in clinical_df.index else -1
            for pid in participant_ids
        ]

    # First split: train vs (val + test)
    train_ids, temp_ids = train_test_split(
        participant_ids,
        test_size=(val_ratio + test_ratio),
        stratify=stratify,
        random_state=random_state
    )

    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)

    if stratify_column is not None:
        stratify_temp = [
            clinical_df.loc[pid, stratify_column]
            if pid in clinical_df.index else -1
            for pid in temp_ids
        ]
    else:
        stratify_temp = None

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=(1 - val_ratio_adjusted),
        stratify=stratify_temp,
        random_state=random_state
    )

    return {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
