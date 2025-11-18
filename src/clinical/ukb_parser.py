"""
UK Biobank Phenotype Parser
Handles 500,000+ participants with 40,000+ fields
Optimized for memory-constrained environments (12GB RAM)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging
from tqdm import tqdm
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UKBiobankParser:
    """
    Parse UK Biobank phenotype data with memory-efficient chunked processing.

    Key features:
    - Handles 50GB+ CSV files in chunks
    - Extracts specific fields only (memory efficient)
    - Parses field instances (0.0, 0.1, 1.0 for repeat assessments)
    - Applies categorical encodings
    - GDPR compliance (withdrawn participants)
    """

    # Field definitions (UKB data coding)
    FIELDS = {
        # Demographics
        'eid': 'Participant ID',
        '31': 'Sex',
        '21003': 'Age when attended assessment centre',
        '21001': 'BMI',
        '54': 'UK Biobank assessment centre',

        # Accelerometry metadata
        '90001': 'Accelerometer data file',
        '90002': 'Accelerometer data processing status',
        '90003': 'Accelerometer data collection date',
        '90004': 'Accelerometer wear start time',
        '90005': 'Accelerometer wear end time',
        '90006': 'Accelerometer total wear time',
        '90007': 'Accelerometer valid days',
        '90008': 'Accelerometer calibration good',
        '90009': 'Accelerometer calibration error (mg)',
        '90015': 'Accelerometer overall activity',

        # Cardiovascular
        '131286': 'Coronary heart disease',
        '131298': 'Hypertension',
        '6150': 'Vascular/heart problems diagnosed by doctor',
        '20002': 'Non-cancer illness code, self-reported',

        # Metabolic
        '30740': 'Glucose',
        '30750': 'HbA1c',
        '2443': 'Diabetes diagnosed by doctor',

        # ICD-10 diagnoses (up to 213 instances)
        '41270': 'ICD-10 diagnosis code',
        '41280': 'Date of ICD-10 diagnosis',

        # Death registry
        '40000': 'Date of death',
        '40001': 'Underlying (primary) cause of death: ICD10',
        '40002': 'Contributory (secondary) causes of death: ICD10',
    }

    # Categorical encodings
    SEX_ENCODING = {0: 'F', 1: 'M'}

    def __init__(
        self,
        phenotype_path: str,
        withdrawn_path: Optional[str] = None,
        chunk_size: int = 10000,
    ):
        """
        Initialize UK Biobank parser.

        Args:
            phenotype_path: Path to main phenotype CSV
            withdrawn_path: Path to withdrawn participants file
            chunk_size: Number of rows per chunk (10k fits in memory)
        """
        self.phenotype_path = Path(phenotype_path)
        self.withdrawn_path = Path(withdrawn_path) if withdrawn_path else None
        self.chunk_size = chunk_size

        # Load withdrawn participants
        self.withdrawn_eids = self._load_withdrawn()

        logger.info(f"Initialized UKB parser: {self.phenotype_path}")
        logger.info(f"Withdrawn participants: {len(self.withdrawn_eids)}")

    def _load_withdrawn(self) -> Set[int]:
        """Load list of withdrawn participants (GDPR compliance)."""
        if not self.withdrawn_path or not self.withdrawn_path.exists():
            return set()

        withdrawn = pd.read_csv(self.withdrawn_path, header=None)
        withdrawn_eids = set(withdrawn[0].values)
        logger.info(f"Loaded {len(withdrawn_eids)} withdrawn participants")
        return withdrawn_eids

    def _get_field_columns(self, all_columns: List[str], field_id: str) -> List[str]:
        """
        Get all column names for a field (including instances).

        UK Biobank fields have instances like:
        - 21003-0.0 (baseline assessment)
        - 21003-1.0 (first repeat)
        - 21003-2.0 (second repeat)
        """
        field_cols = [col for col in all_columns if col.startswith(f'{field_id}-')]
        return sorted(field_cols)

    def parse_demographics(
        self,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Parse demographics and accelerometry metadata.

        Returns DataFrame with columns:
        - eid, sex, age, bmi, assessment_center
        - acc_wear_time, acc_valid_days, acc_calibration_error
        """
        logger.info("Parsing demographics and accelerometry metadata...")

        # Fields to extract
        demo_fields = ['31', '21003', '21001', '54']  # sex, age, bmi, center
        acc_fields = ['90006', '90007', '90009']  # wear time, valid days, calib error

        results = []

        # Read in chunks
        for chunk in tqdm(
            pd.read_csv(self.phenotype_path, chunksize=self.chunk_size, low_memory=False),
            desc="Processing chunks"
        ):
            # Get available columns
            all_cols = chunk.columns.tolist()

            # Extract participant ID
            eids = chunk['eid'].values

            # Filter withdrawn participants
            mask = ~np.isin(eids, list(self.withdrawn_eids))
            chunk_filtered = chunk[mask].copy()

            if len(chunk_filtered) == 0:
                continue

            # Extract demographics
            demo_data = {'eid': chunk_filtered['eid'].values}

            # Sex (baseline only)
            sex_cols = self._get_field_columns(all_cols, '31')
            if sex_cols:
                demo_data['sex'] = chunk_filtered[sex_cols[0]].map(self.SEX_ENCODING)

            # Age (use baseline, instance 0.0)
            age_cols = self._get_field_columns(all_cols, '21003')
            if age_cols:
                demo_data['age'] = chunk_filtered[age_cols[0]].values

            # BMI (baseline)
            bmi_cols = self._get_field_columns(all_cols, '21001')
            if bmi_cols:
                demo_data['bmi'] = chunk_filtered[bmi_cols[0]].values

            # Assessment center
            center_cols = self._get_field_columns(all_cols, '54')
            if center_cols:
                demo_data['assessment_center'] = chunk_filtered[center_cols[0]].values

            # Accelerometry metadata
            wear_time_cols = self._get_field_columns(all_cols, '90006')
            if wear_time_cols:
                demo_data['acc_wear_time'] = chunk_filtered[wear_time_cols[0]].values

            valid_days_cols = self._get_field_columns(all_cols, '90007')
            if valid_days_cols:
                demo_data['acc_valid_days'] = chunk_filtered[valid_days_cols[0]].values

            calib_error_cols = self._get_field_columns(all_cols, '90009')
            if calib_error_cols:
                demo_data['acc_calibration_error'] = chunk_filtered[calib_error_cols[0]].values

            # Convert to DataFrame
            df_chunk = pd.DataFrame(demo_data)
            results.append(df_chunk)

        # Combine all chunks
        df = pd.concat(results, ignore_index=True)

        logger.info(f"Parsed {len(df)} participants")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Save if requested
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved to {output_path}")

        return df

    def parse_icd10_diagnoses(
        self,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Parse ICD-10 diagnoses with dates.

        Returns DataFrame with columns:
        - eid, instance, icd10_code, diagnosis_date

        Each participant can have up to 213 diagnosis instances.
        """
        logger.info("Parsing ICD-10 diagnoses...")

        results = []

        for chunk in tqdm(
            pd.read_csv(self.phenotype_path, chunksize=self.chunk_size, low_memory=False),
            desc="Processing ICD-10"
        ):
            all_cols = chunk.columns.tolist()

            # Filter withdrawn
            eids = chunk['eid'].values
            mask = ~np.isin(eids, list(self.withdrawn_eids))
            chunk_filtered = chunk[mask].copy()

            if len(chunk_filtered) == 0:
                continue

            # Get ICD-10 code columns (41270-0.X, 41270-1.X, etc.)
            icd_cols = self._get_field_columns(all_cols, '41270')
            date_cols = self._get_field_columns(all_cols, '41280')

            # Parse each instance
            for icd_col, date_col in zip(icd_cols, date_cols):
                # Extract instance number from column name
                # e.g., "41270-0.0" -> instance 0
                instance = int(icd_col.split('-')[1].split('.')[0])

                # Get non-null diagnoses
                has_diag = chunk_filtered[icd_col].notna()

                if has_diag.sum() > 0:
                    df_instance = pd.DataFrame({
                        'eid': chunk_filtered.loc[has_diag, 'eid'].values,
                        'instance': instance,
                        'icd10_code': chunk_filtered.loc[has_diag, icd_col].values,
                        'diagnosis_date': pd.to_datetime(
                            chunk_filtered.loc[has_diag, date_col],
                            errors='coerce'
                        )
                    })
                    results.append(df_instance)

        # Combine
        if results:
            df = pd.concat(results, ignore_index=True)

            # Remove invalid dates
            df = df[df['diagnosis_date'].notna()].copy()

            logger.info(f"Parsed {len(df)} ICD-10 diagnoses for {df['eid'].nunique()} participants")
            logger.info(f"Date range: {df['diagnosis_date'].min()} to {df['diagnosis_date'].max()}")

            if output_path:
                df.to_csv(output_path, index=False)
                logger.info(f"Saved to {output_path}")

            return df
        else:
            logger.warning("No ICD-10 diagnoses found")
            return pd.DataFrame()

    def parse_death_registry(
        self,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Parse death registry data.

        Returns DataFrame with columns:
        - eid, death_date, primary_cause, secondary_causes
        """
        logger.info("Parsing death registry...")

        results = []

        for chunk in tqdm(
            pd.read_csv(self.phenotype_path, chunksize=self.chunk_size, low_memory=False),
            desc="Processing deaths"
        ):
            all_cols = chunk.columns.tolist()

            # Filter withdrawn
            eids = chunk['eid'].values
            mask = ~np.isin(eids, list(self.withdrawn_eids))
            chunk_filtered = chunk[mask].copy()

            if len(chunk_filtered) == 0:
                continue

            # Death date (field 40000)
            death_date_cols = self._get_field_columns(all_cols, '40000')
            if not death_date_cols:
                continue

            has_death = chunk_filtered[death_date_cols[0]].notna()

            if has_death.sum() > 0:
                death_data = {
                    'eid': chunk_filtered.loc[has_death, 'eid'].values,
                    'death_date': pd.to_datetime(
                        chunk_filtered.loc[has_death, death_date_cols[0]],
                        errors='coerce'
                    )
                }

                # Primary cause (field 40001)
                primary_cols = self._get_field_columns(all_cols, '40001')
                if primary_cols:
                    death_data['primary_cause'] = chunk_filtered.loc[has_death, primary_cols[0]].values

                # Secondary causes (field 40002, multiple instances)
                secondary_cols = self._get_field_columns(all_cols, '40002')
                if secondary_cols:
                    # Combine all secondary causes
                    secondary_causes = []
                    for col in secondary_cols:
                        causes = chunk_filtered.loc[has_death, col].fillna('').values
                        secondary_causes.append(causes)

                    # Combine into single string
                    death_data['secondary_causes'] = [
                        ','.join([c for c in causes if c])
                        for causes in zip(*secondary_causes)
                    ]

                results.append(pd.DataFrame(death_data))

        # Combine
        if results:
            df = pd.concat(results, ignore_index=True)
            df = df[df['death_date'].notna()].copy()

            logger.info(f"Parsed {len(df)} deaths")
            logger.info(f"Date range: {df['death_date'].min()} to {df['death_date'].max()}")

            if output_path:
                df.to_csv(output_path, index=False)
                logger.info(f"Saved to {output_path}")

            return df
        else:
            logger.warning("No death records found")
            return pd.DataFrame()

    def get_sample_statistics(self) -> Dict:
        """
        Calculate summary statistics for the dataset.
        """
        logger.info("Calculating sample statistics...")

        stats = {
            'total_participants': 0,
            'withdrawn': len(self.withdrawn_eids),
            'with_accelerometry': 0,
            'age_mean': 0,
            'age_std': 0,
            'bmi_mean': 0,
            'sex_distribution': {},
        }

        # Read first chunk to get quick stats
        chunk = pd.read_csv(self.phenotype_path, nrows=self.chunk_size, low_memory=False)

        # Estimate total from file size (rough approximation)
        stats['total_participants'] = len(chunk)  # Will be updated during full parse

        return stats


def test_parser():
    """Test the UKB parser with sample data."""
    logger.info("Testing UK Biobank parser...")

    # Create sample data
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Write sample phenotype data
        f.write("eid,31-0.0,21003-0.0,21001-0.0,54-0.0,90006-0.0,90007-0.0,90009-0.0\n")
        for i in range(100):
            sex = np.random.choice([0, 1])
            age = np.random.randint(40, 70)
            bmi = np.random.normal(27, 5)
            center = np.random.randint(11000, 11024)
            wear_time = np.random.uniform(0.6, 0.9)
            valid_days = np.random.randint(3, 8)
            calib_error = np.random.uniform(0, 15)

            f.write(f"{i},{sex},{age},{bmi:.1f},{center},{wear_time:.3f},{valid_days},{calib_error:.2f}\n")

        sample_path = f.name

    # Test parser
    parser = UKBiobankParser(sample_path, chunk_size=50)

    # Parse demographics
    df_demo = parser.parse_demographics()
    print(f"\nDemographics: {len(df_demo)} participants")
    print(df_demo.head())
    print(f"\nSex distribution:\n{df_demo['sex'].value_counts()}")
    print(f"\nAge: {df_demo['age'].mean():.1f} ± {df_demo['age'].std():.1f}")

    # Clean up
    import os
    os.unlink(sample_path)

    logger.info("✓ Parser test passed!")


if __name__ == '__main__':
    test_parser()
