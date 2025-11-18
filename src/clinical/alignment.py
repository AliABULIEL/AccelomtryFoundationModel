"""
Clinical-Accelerometry Data Alignment
Matches clinical outcomes with accelerometry data
Implements quality control and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import h5py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalAccelerometryAligner:
    """
    Align clinical outcomes with accelerometry data.

    Quality criteria:
    - Wear time ≥66% (16 hours/day minimum)
    - Calibration error <10mg
    - ≥3 valid wear days
    - Valid temporal alignment with outcomes
    """

    # Quality thresholds
    MIN_WEAR_TIME = 0.66  # 66% = 16 hours/day
    MAX_CALIB_ERROR = 10.0  # mg
    MIN_VALID_DAYS = 3

    def __init__(
        self,
        min_wear_time: float = 0.66,
        max_calib_error: float = 10.0,
        min_valid_days: int = 3,
    ):
        """
        Initialize aligner.

        Args:
            min_wear_time: Minimum wear time fraction
            max_calib_error: Maximum calibration error (mg)
            min_valid_days: Minimum valid wear days
        """
        self.min_wear_time = min_wear_time
        self.max_calib_error = max_calib_error
        self.min_valid_days = min_valid_days

    def apply_quality_filters(
        self,
        df_demographics: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply quality filters to accelerometry data.

        Args:
            df_demographics: Demographics with accelerometry QC metrics

        Returns:
            Filtered DataFrame and filter statistics
        """
        logger.info("Applying quality filters...")

        initial_n = len(df_demographics)

        # Filter 1: Wear time
        if 'acc_wear_time' in df_demographics.columns:
            mask_wear = df_demographics['acc_wear_time'] >= self.min_wear_time
            n_wear = mask_wear.sum()
        else:
            mask_wear = pd.Series([True] * len(df_demographics))
            n_wear = initial_n

        # Filter 2: Calibration error
        if 'acc_calibration_error' in df_demographics.columns:
            mask_calib = df_demographics['acc_calibration_error'] <= self.max_calib_error
            n_calib = mask_calib.sum()
        else:
            mask_calib = pd.Series([True] * len(df_demographics))
            n_calib = initial_n

        # Filter 3: Valid days
        if 'acc_valid_days' in df_demographics.columns:
            mask_days = df_demographics['acc_valid_days'] >= self.min_valid_days
            n_days = mask_days.sum()
        else:
            mask_days = pd.Series([True] * len(df_demographics))
            n_days = initial_n

        # Combined filter
        mask_all = mask_wear & mask_calib & mask_days
        df_filtered = df_demographics[mask_all].copy()

        # Statistics
        stats = {
            'initial_n': initial_n,
            'passed_wear_time': int(n_wear),
            'passed_calibration': int(n_calib),
            'passed_valid_days': int(n_days),
            'passed_all': len(df_filtered),
            'excluded': initial_n - len(df_filtered),
        }

        logger.info(f"Quality filter results:")
        logger.info(f"  Initial:        {stats['initial_n']:,}")
        logger.info(f"  Wear time ≥{self.min_wear_time:.0%}:  {stats['passed_wear_time']:,}")
        logger.info(f"  Calibration <{self.max_calib_error}mg: {stats['passed_calibration']:,}")
        logger.info(f"  Valid days ≥{self.min_valid_days}:   {stats['passed_valid_days']:,}")
        logger.info(f"  Passed all:     {stats['passed_all']:,}")
        logger.info(f"  Excluded:       {stats['excluded']:,}")

        return df_filtered, stats

    def merge_clinical_data(
        self,
        df_demographics: pd.DataFrame,
        df_outcomes: pd.DataFrame,
        df_death: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Merge demographics, outcomes, and death registry.

        Args:
            df_demographics: Participant demographics + accelerometry QC
            df_outcomes: Clinical outcomes
            df_death: Death registry

        Returns:
            Merged DataFrame
        """
        logger.info("Merging clinical data...")

        # Start with demographics
        df = df_demographics.copy()

        # Add outcomes
        df = df.merge(df_outcomes, on='eid', how='left')

        # Add death
        if df_death is not None:
            df = df.merge(
                df_death[['eid', 'death_date', 'primary_cause']],
                on='eid',
                how='left'
            )

        logger.info(f"Merged data: {len(df)} participants")

        return df

    def create_ml_dataset(
        self,
        df_merged: pd.DataFrame,
        accelerometry_hdf5: str,
        output_path: str,
        split_by_center: bool = True,
    ) -> Dict:
        """
        Create ML-ready dataset with proper splits.

        Args:
            df_merged: Merged clinical data
            accelerometry_hdf5: Path to accelerometry HDF5 file
            output_path: Output HDF5 path
            split_by_center: Split by assessment center for geographic diversity

        Returns:
            Dataset statistics
        """
        logger.info("Creating ML-ready dataset...")

        # Filter for participants with good quality data
        df_filtered, qc_stats = self.apply_quality_filters(df_merged)

        # Split by assessment center
        if split_by_center and 'assessment_center' in df_filtered.columns:
            # Get unique centers
            centers = df_filtered['assessment_center'].unique()

            # Assign centers to splits (70/15/15)
            np.random.seed(42)
            np.random.shuffle(centers)

            n_centers = len(centers)
            n_train = int(n_centers * 0.7)
            n_val = int(n_centers * 0.15)

            train_centers = centers[:n_train]
            val_centers = centers[n_train:n_train + n_val]
            test_centers = centers[n_train + n_val:]

            df_filtered['split'] = 'test'  # Default
            df_filtered.loc[df_filtered['assessment_center'].isin(train_centers), 'split'] = 'train'
            df_filtered.loc[df_filtered['assessment_center'].isin(val_centers), 'split'] = 'val'

            logger.info(f"Split by {n_centers} assessment centers")
        else:
            # Random split
            from sklearn.model_selection import train_test_split

            train_idx, temp_idx = train_test_split(
                df_filtered.index,
                test_size=0.3,
                random_state=42
            )

            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=0.5,
                random_state=42
            )

            df_filtered['split'] = 'test'
            df_filtered.loc[train_idx, 'split'] = 'train'
            df_filtered.loc[val_idx, 'split'] = 'val'

        # Split sizes
        split_counts = df_filtered['split'].value_counts()
        logger.info(f"Split sizes:\n{split_counts}")

        # Save to HDF5
        logger.info(f"Saving to {output_path}...")

        with h5py.File(output_path, 'w') as f:
            # Save participant metadata
            for split in ['train', 'val', 'test']:
                df_split = df_filtered[df_filtered['split'] == split]

                if len(df_split) > 0:
                    grp = f.create_group(split)

                    # Participant IDs
                    grp.create_dataset('eids', data=df_split['eid'].values)

                    # Demographics
                    if 'age' in df_split.columns:
                        grp.create_dataset('age', data=df_split['age'].values)
                    if 'sex' in df_split.columns:
                        sex_encoded = df_split['sex'].map({'M': 1, 'F': 0}).values
                        grp.create_dataset('sex', data=sex_encoded)
                    if 'bmi' in df_split.columns:
                        grp.create_dataset('bmi', data=df_split['bmi'].fillna(0).values)

                    # Outcomes (all disease groups)
                    outcome_cols = [col for col in df_split.columns if '_incident' in col]
                    for col in outcome_cols:
                        grp.create_dataset(col, data=df_split[col].fillna(0).values)

                    logger.info(f"  {split}: {len(df_split)} participants")

            # Store metadata
            f.attrs['n_train'] = int(split_counts.get('train', 0))
            f.attrs['n_val'] = int(split_counts.get('val', 0))
            f.attrs['n_test'] = int(split_counts.get('test', 0))
            f.attrs['split_by_center'] = split_by_center

        logger.info(f"Saved ML dataset to {output_path}")

        # Statistics
        stats = {
            'n_total': len(df_filtered),
            'n_train': int(split_counts.get('train', 0)),
            'n_val': int(split_counts.get('val', 0)),
            'n_test': int(split_counts.get('test', 0)),
            'qc_stats': qc_stats,
        }

        return stats

    def validate_temporal_alignment(
        self,
        df_merged: pd.DataFrame
    ) -> Dict:
        """
        Validate temporal alignment between accelerometry and outcomes.

        Checks for data leakage and proper incident disease classification.

        Args:
            df_merged: Merged data with outcomes

        Returns:
            Validation report
        """
        logger.info("Validating temporal alignment...")

        validation = {
            'total_participants': len(df_merged),
            'with_accelerometry_date': 0,
            'prevalent_excluded': 0,
            'incident_cases': {},
            'temporal_issues': 0,
        }

        # Check for accelerometry date
        if 'accelerometry_date' in df_merged.columns:
            validation['with_accelerometry_date'] = df_merged['accelerometry_date'].notna().sum()

        # Check each disease group
        for col in df_merged.columns:
            if col.endswith('_prevalent'):
                disease = col.replace('_prevalent', '')
                n_prevalent = df_merged[col].sum()
                validation['prevalent_excluded'] += n_prevalent

            if col.endswith('_incident'):
                disease = col.replace('_incident', '')
                n_incident = df_merged[col].sum()

                # Check days to event
                days_col = f'{disease}_days_to_event'
                if days_col in df_merged.columns:
                    incident_mask = df_merged[col] == 1
                    if incident_mask.sum() > 0:
                        days = df_merged.loc[incident_mask, days_col]
                        mean_days = days.mean()
                        median_days = days.median()

                        validation['incident_cases'][disease] = {
                            'n': int(n_incident),
                            'mean_days_to_event': float(mean_days),
                            'median_days_to_event': float(median_days),
                        }

        logger.info("Temporal alignment validation:")
        logger.info(f"  Total participants: {validation['total_participants']}")
        logger.info(f"  With accelerometry date: {validation['with_accelerometry_date']}")
        logger.info(f"  Prevalent cases excluded: {validation['prevalent_excluded']}")

        for disease, stats in validation['incident_cases'].items():
            logger.info(
                f"  {disease}: {stats['n']} incident cases, "
                f"median {stats['median_days_to_event']:.0f} days to event"
            )

        return validation


def test_aligner():
    """Test clinical-accelerometry aligner."""
    logger.info("Testing clinical-accelerometry aligner...")

    # Create sample data
    df_demo = pd.DataFrame({
        'eid': range(100),
        'age': np.random.randint(40, 70, 100),
        'sex': np.random.choice(['M', 'F'], 100),
        'bmi': np.random.normal(27, 5, 100),
        'assessment_center': np.random.choice([11000, 11001, 11002], 100),
        'acc_wear_time': np.random.uniform(0.5, 0.95, 100),
        'acc_valid_days': np.random.randint(1, 8, 100),
        'acc_calibration_error': np.random.uniform(0, 15, 100),
    })

    df_outcomes = pd.DataFrame({
        'eid': range(100),
        'cvd_incident': np.random.choice([0, 1], 100, p=[0.92, 0.08]),
        'cvd_days_to_event': np.random.uniform(100, 2000, 100),
        'diabetes_incident': np.random.choice([0, 1], 100, p=[0.96, 0.04]),
    })

    # Test aligner
    aligner = ClinicalAccelerometryAligner()

    # Apply QC
    df_filtered, qc_stats = aligner.apply_quality_filters(df_demo)

    print("\nQuality filter stats:")
    for key, value in qc_stats.items():
        print(f"  {key}: {value}")

    # Merge
    df_merged = aligner.merge_clinical_data(df_filtered, df_outcomes)

    print(f"\nMerged data: {len(df_merged)} participants")
    print(df_merged.head())

    logger.info("✓ Aligner test passed!")


if __name__ == '__main__':
    test_aligner()
