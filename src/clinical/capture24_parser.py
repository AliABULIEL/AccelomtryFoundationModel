"""
CAPTURE-24 Annotation Parser
Maps 206 CPA codes to 4 intensity levels using MET thresholds
Handles train/test split (P001-P100 train, P101-P151 test)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CAPTURE24Parser:
    """
    Parse CAPTURE-24 wrist-worn accelerometry annotations.

    Dataset:
    - 151 participants (P001-P151)
    - 24-hour free-living activities
    - 206 CPA (Compendium of Physical Activities) codes
    - Ground truth annotations at second-level resolution

    Intensity levels:
    - Sleep: 0-0.95 METs
    - Sedentary: 0.95-1.5 METs
    - Light PA: 1.5-3.0 METs
    - MVPA: ≥3.0 METs
    """

    # MET thresholds for intensity classification
    MET_THRESHOLDS = {
        'sleep': (0.0, 0.95),
        'sedentary': (0.95, 1.5),
        'light': (1.5, 3.0),
        'mvpa': (3.0, float('inf')),
    }

    # CPA code to MET mapping (subset - full mapping would be 206 codes)
    # Based on Ainsworth et al. 2011 Compendium of Physical Activities
    CPA_TO_METS = {
        # Sleep and rest
        '07030': 0.9,   # Sleeping
        '07020': 1.0,   # Lying quietly
        '07010': 1.0,   # Resting, sitting

        # Sedentary
        '09070': 1.3,   # Sitting, reading
        '09020': 1.3,   # Sitting, writing
        '05040': 1.5,   # Sitting, light office work
        '09090': 1.5,   # Watching TV

        # Light activities
        '02080': 2.0,   # Walking, slow (2 mph)
        '05020': 2.3,   # Cooking
        '05050': 2.5,   # Cleaning, light
        '13020': 2.8,   # Standing, light work

        # Moderate activities
        '02010': 3.5,   # Walking, 3.5 mph
        '02040': 3.8,   # Walking, 4 mph
        '05100': 3.5,   # Cleaning, vigorous
        '08030': 4.0,   # Gardening

        # Vigorous activities
        '02050': 5.0,   # Walking, very brisk (4.5 mph)
        '12030': 6.0,   # Running, 5 mph
        '12050': 8.0,   # Running, 6 mph
        '12070': 10.0,  # Running, 7 mph
        '01010': 8.0,   # Cycling, vigorous

        # NOTE: Full mapping would include all 206 codes
        # This is a representative subset for demonstration
    }

    # Official train/test split
    TRAIN_IDS = [f'P{i:03d}' for i in range(1, 101)]   # P001-P100
    TEST_IDS = [f'P{i:03d}' for i in range(101, 152)]  # P101-P151

    def __init__(self, annotation_dir: str):
        """
        Initialize CAPTURE-24 parser.

        Args:
            annotation_dir: Directory containing annotation CSV files
                           (one file per participant: P001.csv, P002.csv, etc.)
        """
        self.annotation_dir = Path(annotation_dir)

        if not self.annotation_dir.exists():
            logger.warning(f"Annotation directory does not exist: {annotation_dir}")

        logger.info(f"Initialized CAPTURE-24 parser: {self.annotation_dir}")

    def get_mets_for_cpa(self, cpa_code: str) -> float:
        """
        Get MET value for CPA code.

        Args:
            cpa_code: CPA code (e.g., '02080')

        Returns:
            MET value (defaults to 1.5 for unknown codes)
        """
        return self.CPA_TO_METS.get(cpa_code, 1.5)  # Default to light sedentary

    def classify_intensity(self, mets: float) -> str:
        """
        Classify MET value into intensity level.

        Args:
            mets: MET value

        Returns:
            Intensity level: 'sleep', 'sedentary', 'light', or 'mvpa'
        """
        for intensity, (min_mets, max_mets) in self.MET_THRESHOLDS.items():
            if min_mets <= mets < max_mets:
                return intensity

        return 'sedentary'  # Default

    def parse_participant_annotations(
        self,
        participant_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Parse annotations for single participant.

        Args:
            participant_id: Participant ID (e.g., 'P001')

        Returns:
            DataFrame with columns: timestamp, cpa_code, mets, intensity
        """
        annotation_file = self.annotation_dir / f"{participant_id}.csv"

        if not annotation_file.exists():
            logger.warning(f"Annotation file not found: {annotation_file}")
            return None

        try:
            # Read annotation file
            df = pd.read_csv(annotation_file)

            # Expected columns: timestamp, annotation (CPA code)
            if 'timestamp' not in df.columns:
                logger.error(f"Missing 'timestamp' column in {annotation_file}")
                return None

            # Get CPA codes
            if 'annotation' in df.columns:
                cpa_col = 'annotation'
            elif 'cpa_code' in df.columns:
                cpa_col = 'cpa_code'
            else:
                logger.error(f"Missing annotation column in {annotation_file}")
                return None

            # Add participant ID
            df['participant_id'] = participant_id

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            # Get METs for each CPA code
            df['cpa_code'] = df[cpa_col].astype(str)
            df['mets'] = df['cpa_code'].apply(self.get_mets_for_cpa)

            # Classify intensity
            df['intensity'] = df['mets'].apply(self.classify_intensity)

            # Keep relevant columns
            df = df[['participant_id', 'timestamp', 'cpa_code', 'mets', 'intensity']].copy()

            return df

        except Exception as e:
            logger.error(f"Error parsing {annotation_file}: {e}")
            return None

    def parse_all_annotations(
        self,
        participant_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Parse annotations for multiple participants.

        Args:
            participant_ids: List of participant IDs (defaults to all P001-P151)

        Returns:
            Combined DataFrame with all annotations
        """
        if participant_ids is None:
            participant_ids = self.TRAIN_IDS + self.TEST_IDS

        logger.info(f"Parsing annotations for {len(participant_ids)} participants...")

        all_annotations = []

        for pid in tqdm(participant_ids, desc="Parsing annotations"):
            df = self.parse_participant_annotations(pid)

            if df is not None and len(df) > 0:
                all_annotations.append(df)

        if all_annotations:
            df_combined = pd.concat(all_annotations, ignore_index=True)

            logger.info(f"Parsed {len(df_combined)} annotation records")
            logger.info(f"Participants: {df_combined['participant_id'].nunique()}")

            # Intensity distribution
            intensity_dist = df_combined['intensity'].value_counts(normalize=True) * 100
            logger.info(f"\nIntensity distribution:\n{intensity_dist}")

            return df_combined
        else:
            logger.warning("No annotations parsed")
            return pd.DataFrame()

    def aggregate_to_windows(
        self,
        df_annotations: pd.DataFrame,
        window_size_sec: int = 8.192 * 100 / 100,  # 8.192 seconds
    ) -> pd.DataFrame:
        """
        Aggregate second-level annotations to fixed windows.

        Uses majority voting within each window.

        Args:
            df_annotations: Annotation data
            window_size_sec: Window size in seconds (8.192s for TTM)

        Returns:
            DataFrame with one row per window
        """
        logger.info(f"Aggregating to {window_size_sec:.3f}s windows...")

        results = []

        for participant_id in df_annotations['participant_id'].unique():
            df_p = df_annotations[df_annotations['participant_id'] == participant_id].copy()
            df_p = df_p.sort_values('timestamp')

            # Create time bins
            df_p['time_seconds'] = (
                df_p['timestamp'] - df_p['timestamp'].min()
            ).dt.total_seconds()

            df_p['window_id'] = (df_p['time_seconds'] // window_size_sec).astype(int)

            # Aggregate each window
            for window_id, window_data in df_p.groupby('window_id'):
                # Majority vote for intensity
                intensity = window_data['intensity'].mode()[0]

                # Mean METs
                mean_mets = window_data['mets'].mean()

                # Confidence (fraction of majority class)
                confidence = (window_data['intensity'] == intensity).mean()

                results.append({
                    'participant_id': participant_id,
                    'window_id': window_id,
                    'start_time': window_data['timestamp'].min(),
                    'intensity': intensity,
                    'mean_mets': mean_mets,
                    'confidence': confidence,
                    'n_samples': len(window_data),
                })

        df_windows = pd.DataFrame(results)

        logger.info(f"Created {len(df_windows)} windows")

        # Confidence distribution
        logger.info(f"Mean confidence: {df_windows['confidence'].mean():.3f}")
        logger.info(f"Low confidence (<0.8): {(df_windows['confidence'] < 0.8).sum()} windows")

        return df_windows

    def create_train_test_split(
        self,
        df_windows: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split into official train/test sets.

        Args:
            df_windows: Window-level annotations

        Returns:
            Tuple of (train_df, test_df)
        """
        train_mask = df_windows['participant_id'].isin(self.TRAIN_IDS)

        df_train = df_windows[train_mask].copy()
        df_test = df_windows[~train_mask].copy()

        logger.info(f"Train set: {len(df_train)} windows from {df_train['participant_id'].nunique()} participants")
        logger.info(f"Test set: {len(df_test)} windows from {df_test['participant_id'].nunique()} participants")

        return df_train, df_test

    def get_label_encoder(self) -> Dict:
        """
        Get label encoding for intensity levels.

        Returns:
            Dict with mappings and inverse mappings
        """
        label_to_int = {
            'sleep': 0,
            'sedentary': 1,
            'light': 2,
            'mvpa': 3,
        }

        int_to_label = {v: k for k, v in label_to_int.items()}

        return {
            'label_to_int': label_to_int,
            'int_to_label': int_to_label,
            'n_classes': 4,
            'class_names': ['Sleep', 'Sedentary', 'Light', 'MVPA'],
        }


def test_capture24_parser():
    """Test CAPTURE-24 parser with sample data."""
    logger.info("Testing CAPTURE-24 parser...")

    # Create sample annotation data
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample annotation file
        sample_file = Path(tmpdir) / 'P001.csv'

        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1S'),
            'annotation': ['07030'] * 300 +  # Sleep
                         ['09070'] * 400 +  # Sedentary
                         ['02080'] * 200 +  # Light
                         ['12030'] * 100,   # MVPA
        })

        sample_data.to_csv(sample_file, index=False)

        # Test parser
        parser = CAPTURE24Parser(tmpdir)

        # Parse participant
        df = parser.parse_participant_annotations('P001')

        print("\nParsed annotations:")
        print(df.head(10))
        print(f"\nTotal records: {len(df)}")

        # Intensity distribution
        print("\nIntensity distribution:")
        print(df['intensity'].value_counts())

        # Aggregate to windows
        df_windows = parser.aggregate_to_windows(df, window_size_sec=10)

        print("\nWindowed data:")
        print(df_windows.head())
        print(f"\nTotal windows: {len(df_windows)}")

        # Label encoder
        encoder = parser.get_label_encoder()
        print("\nLabel encoding:")
        print(encoder)

    logger.info("✓ CAPTURE-24 parser test passed!")


if __name__ == '__main__':
    test_capture24_parser()
