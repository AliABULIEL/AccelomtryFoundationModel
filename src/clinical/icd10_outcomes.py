"""
ICD-10 Outcome Processing for UK Biobank
Handles disease classification, temporal alignment, and competing risks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICD10OutcomeProcessor:
    """
    Process ICD-10 diagnosis codes into clinical outcomes.

    Features:
    - Disease group classification (CVD, diabetes, dementia)
    - Temporal alignment with accelerometry dates
    - Incident vs prevalent disease
    - Competing risks (death before outcome)
    - Quality validation
    """

    # Disease group definitions (ICD-10 codes)
    DISEASE_GROUPS = {
        'cvd': {
            'CHD': ['I20', 'I21', 'I22', 'I23', 'I24', 'I25'],  # Coronary heart disease
            'stroke': ['I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69'],
            'heart_failure': ['I50'],
            'atrial_fib': ['I48'],
        },
        'diabetes': {
            'type1': ['E10'],
            'type2': ['E11'],
            'other': ['E12', 'E13', 'E14'],
        },
        'dementia': {
            'alzheimers': ['G30', 'F00'],
            'vascular_dementia': ['F01'],
            'other_dementia': ['F02', 'F03'],
        },
        'cancer': {
            'lung': ['C34'],
            'breast': ['C50'],
            'prostate': ['C61'],
            'colorectal': ['C18', 'C19', 'C20'],
        },
        'copd': {
            'copd': ['J44'],
            'emphysema': ['J43'],
        }
    }

    # Minimum days between accelerometry and diagnosis for incident disease
    INCIDENT_MIN_DAYS = 30

    def __init__(self, min_incident_days: int = 30):
        """
        Initialize ICD-10 processor.

        Args:
            min_incident_days: Minimum days between accelerometry and diagnosis
                              for incident disease classification
        """
        self.min_incident_days = min_incident_days

        # Create flat lookup for quick matching
        self.icd_lookup = self._create_icd_lookup()

    def _create_icd_lookup(self) -> Dict[str, Tuple[str, str]]:
        """
        Create lookup dict mapping ICD codes to disease groups.

        Returns:
            Dict: {icd_code: (group, subgroup)}
        """
        lookup = {}

        for group_name, subgroups in self.DISEASE_GROUPS.items():
            for subgroup_name, icd_codes in subgroups.items():
                for icd_code in icd_codes:
                    lookup[icd_code] = (group_name, subgroup_name)

        logger.info(f"Created ICD-10 lookup with {len(lookup)} codes")
        return lookup

    def match_icd_code(self, icd_code: str) -> Optional[Tuple[str, str]]:
        """
        Match ICD-10 code to disease group.

        Args:
            icd_code: ICD-10 code (e.g., "I21.0")

        Returns:
            Tuple of (group, subgroup) or None
        """
        if not isinstance(icd_code, str):
            return None

        # Extract base code (first 3 characters)
        base_code = icd_code[:3]

        return self.icd_lookup.get(base_code)

    def classify_diagnoses(
        self,
        df_diagnoses: pd.DataFrame,
        accelerometry_dates: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Classify diagnoses into disease groups with temporal alignment.

        Args:
            df_diagnoses: DataFrame with columns [eid, icd10_code, diagnosis_date]
            accelerometry_dates: DataFrame with [eid, accelerometry_date]

        Returns:
            DataFrame with classified outcomes
        """
        logger.info(f"Classifying {len(df_diagnoses)} diagnoses...")

        # Add disease group classification
        classifications = []

        for _, row in df_diagnoses.iterrows():
            match = self.match_icd_code(row['icd10_code'])

            if match:
                group, subgroup = match
                classifications.append({
                    'eid': row['eid'],
                    'icd10_code': row['icd10_code'],
                    'diagnosis_date': row['diagnosis_date'],
                    'disease_group': group,
                    'disease_subgroup': subgroup,
                })

        df_classified = pd.DataFrame(classifications)

        logger.info(f"Classified {len(df_classified)} diagnoses into disease groups")

        # Group distribution
        if len(df_classified) > 0:
            group_counts = df_classified['disease_group'].value_counts()
            logger.info(f"\nDisease group distribution:\n{group_counts}")

        # Add temporal alignment if accelerometry dates provided
        if accelerometry_dates is not None and len(df_classified) > 0:
            df_classified = self._add_temporal_alignment(
                df_classified,
                accelerometry_dates
            )

        return df_classified

    def _add_temporal_alignment(
        self,
        df_diagnoses: pd.DataFrame,
        df_accelerometry: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add temporal alignment between diagnoses and accelerometry.

        Classifies each diagnosis as:
        - prevalent: diagnosis before accelerometry
        - incident: diagnosis >30 days after accelerometry
        - acute: diagnosis within 30 days of accelerometry (excluded)
        """
        logger.info("Adding temporal alignment...")

        # Merge with accelerometry dates
        df_merged = df_diagnoses.merge(
            df_accelerometry[['eid', 'accelerometry_date']],
            on='eid',
            how='left'
        )

        # Calculate days between accelerometry and diagnosis
        df_merged['days_to_diagnosis'] = (
            df_merged['diagnosis_date'] - df_merged['accelerometry_date']
        ).dt.days

        # Classify temporal relationship
        def classify_temporal(days):
            if pd.isna(days):
                return 'unknown'
            elif days < 0:
                return 'prevalent'
            elif days >= self.min_incident_days:
                return 'incident'
            else:
                return 'acute'  # Too close to accelerometry

        df_merged['temporal_class'] = df_merged['days_to_diagnosis'].apply(classify_temporal)

        # Distribution
        temporal_counts = df_merged['temporal_class'].value_counts()
        logger.info(f"\nTemporal classification:\n{temporal_counts}")

        return df_merged

    def create_outcome_summary(
        self,
        df_diagnoses: pd.DataFrame,
        include_death: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create per-participant outcome summary.

        Args:
            df_diagnoses: Classified diagnoses with temporal alignment
            include_death: Death registry data

        Returns:
            DataFrame with one row per participant showing all outcomes
        """
        logger.info("Creating outcome summary...")

        # Get unique participants
        eids = df_diagnoses['eid'].unique()

        outcomes = []

        for eid in eids:
            participant_data = df_diagnoses[df_diagnoses['eid'] == eid]

            outcome = {'eid': eid}

            # For each disease group
            for group_name in self.DISEASE_GROUPS.keys():
                group_data = participant_data[participant_data['disease_group'] == group_name]

                if len(group_data) == 0:
                    # No diagnosis
                    outcome[f'{group_name}_incident'] = 0
                    outcome[f'{group_name}_prevalent'] = 0
                    outcome[f'{group_name}_days_to_event'] = np.nan
                else:
                    # Check for incident cases
                    incident = group_data[group_data['temporal_class'] == 'incident']
                    prevalent = group_data[group_data['temporal_class'] == 'prevalent']

                    outcome[f'{group_name}_incident'] = int(len(incident) > 0)
                    outcome[f'{group_name}_prevalent'] = int(len(prevalent) > 0)

                    # Days to first incident event
                    if len(incident) > 0:
                        min_days = incident['days_to_diagnosis'].min()
                        outcome[f'{group_name}_days_to_event'] = min_days
                    else:
                        outcome[f'{group_name}_days_to_event'] = np.nan

            outcomes.append(outcome)

        df_outcomes = pd.DataFrame(outcomes)

        # Add death information if provided
        if include_death is not None:
            df_outcomes = df_outcomes.merge(
                include_death[['eid', 'death_date', 'primary_cause']],
                on='eid',
                how='left'
            )

            # Flag deaths
            df_outcomes['died'] = df_outcomes['death_date'].notna().astype(int)

        logger.info(f"Created outcomes for {len(df_outcomes)} participants")

        # Summary statistics
        for group in self.DISEASE_GROUPS.keys():
            incident_col = f'{group}_incident'
            if incident_col in df_outcomes.columns:
                rate = df_outcomes[incident_col].mean() * 100
                n = df_outcomes[incident_col].sum()
                logger.info(f"  {group}: {n} incident cases ({rate:.1f}%)")

        return df_outcomes

    def calculate_followup_time(
        self,
        df_accelerometry: pd.DataFrame,
        df_outcomes: pd.DataFrame,
        df_death: Optional[pd.DataFrame] = None,
        censor_date: str = '2023-12-31'
    ) -> pd.DataFrame:
        """
        Calculate follow-up time for survival analysis.

        Args:
            df_accelerometry: Accelerometry dates
            df_outcomes: Outcome data
            df_death: Death registry
            censor_date: Administrative censoring date

        Returns:
            DataFrame with follow-up time and event indicators
        """
        logger.info("Calculating follow-up time...")

        censor_date = pd.to_datetime(censor_date)

        # Merge data
        df = df_accelerometry[['eid', 'accelerometry_date']].copy()

        # Add outcomes
        for group in self.DISEASE_GROUPS.keys():
            event_col = f'{group}_incident'
            days_col = f'{group}_days_to_event'

            if event_col in df_outcomes.columns:
                df = df.merge(
                    df_outcomes[['eid', event_col, days_col]],
                    on='eid',
                    how='left'
                )

        # Add death
        if df_death is not None:
            df = df.merge(
                df_death[['eid', 'death_date']],
                on='eid',
                how='left'
            )
        else:
            df['death_date'] = pd.NaT

        # Calculate follow-up time for each outcome
        for group in self.DISEASE_GROUPS.keys():
            event_col = f'{group}_incident'
            days_col = f'{group}_days_to_event'
            followup_col = f'{group}_followup_years'

            if event_col in df.columns:
                # End date is earliest of: event, death, or censor
                def get_followup(row):
                    # Start from accelerometry
                    start = row['accelerometry_date']

                    # Event date
                    if row[event_col] == 1 and not pd.isna(row[days_col]):
                        event_date = start + timedelta(days=row[days_col])
                    else:
                        event_date = pd.NaT

                    # Death date
                    death_date = row['death_date']

                    # End date is earliest non-null
                    end_dates = [d for d in [event_date, death_date, censor_date] if pd.notna(d)]
                    if end_dates:
                        end_date = min(end_dates)
                    else:
                        end_date = censor_date

                    # Follow-up in years
                    followup_days = (end_date - start).days
                    return followup_days / 365.25

                df[followup_col] = df.apply(get_followup, axis=1)

        logger.info("Calculated follow-up time")

        # Summary
        for group in self.DISEASE_GROUPS.keys():
            followup_col = f'{group}_followup_years'
            if followup_col in df.columns:
                mean_followup = df[followup_col].mean()
                logger.info(f"  {group}: {mean_followup:.1f} years mean follow-up")

        return df

    def validate_outcomes(
        self,
        df_outcomes: pd.DataFrame,
        expected_rates: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict:
        """
        Validate outcome rates against expected prevalence.

        Args:
            df_outcomes: Outcome summary
            expected_rates: Dict of {disease: (min_rate, max_rate)} as percentages

        Returns:
            Validation report
        """
        if expected_rates is None:
            # Default expected rates (per 100 person-years for 5-year follow-up)
            expected_rates = {
                'cvd': (5, 10),        # 5-10% incident CVD
                'diabetes': (2, 5),     # 2-5% incident diabetes
                'dementia': (0.5, 2),   # 0.5-2% incident dementia
                'cancer': (1, 5),       # 1-5% incident cancer
            }

        logger.info("Validating outcome rates...")

        validation = {}

        for group, (min_rate, max_rate) in expected_rates.items():
            incident_col = f'{group}_incident'

            if incident_col in df_outcomes.columns:
                observed_rate = df_outcomes[incident_col].mean() * 100
                n_events = df_outcomes[incident_col].sum()

                is_valid = min_rate <= observed_rate <= max_rate

                validation[group] = {
                    'observed_rate_pct': observed_rate,
                    'n_events': int(n_events),
                    'expected_range': (min_rate, max_rate),
                    'valid': is_valid,
                }

                status = "✓" if is_valid else "✗"
                logger.info(
                    f"  {status} {group}: {observed_rate:.1f}% "
                    f"(expected {min_rate}-{max_rate}%, n={n_events})"
                )

        return validation


def test_icd10_processor():
    """Test ICD-10 processor."""
    logger.info("Testing ICD-10 outcome processor...")

    # Create sample diagnosis data
    sample_diagnoses = pd.DataFrame({
        'eid': [1, 1, 2, 2, 3, 3, 4],
        'icd10_code': ['I21.0', 'I50.0', 'E11.0', 'I63.0', 'G30.0', 'C34.0', 'I48.0'],
        'diagnosis_date': pd.to_datetime([
            '2015-06-01', '2016-03-15', '2014-12-01',
            '2017-08-20', '2016-11-10', '2015-03-01', '2016-07-14'
        ])
    })

    # Sample accelerometry dates
    sample_acc = pd.DataFrame({
        'eid': [1, 2, 3, 4],
        'accelerometry_date': pd.to_datetime([
            '2015-01-01', '2015-01-01', '2015-01-01', '2015-01-01'
        ])
    })

    # Initialize processor
    processor = ICD10OutcomeProcessor(min_incident_days=30)

    # Classify diagnoses
    df_classified = processor.classify_diagnoses(
        sample_diagnoses,
        sample_acc
    )

    print("\nClassified diagnoses:")
    print(df_classified[['eid', 'icd10_code', 'disease_group', 'temporal_class', 'days_to_diagnosis']])

    # Create outcome summary
    df_outcomes = processor.create_outcome_summary(df_classified)

    print("\nOutcome summary:")
    print(df_outcomes)

    # Validate
    validation = processor.validate_outcomes(df_outcomes)

    print("\nValidation:")
    for group, result in validation.items():
        print(f"  {group}: {result}")

    logger.info("✓ ICD-10 processor test passed!")


if __name__ == '__main__':
    test_icd10_processor()
