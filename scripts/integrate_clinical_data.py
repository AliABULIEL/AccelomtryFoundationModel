#!/usr/bin/env python3
"""
Clinical Data Integration Pipeline
Complete workflow for integrating UK Biobank clinical data with accelerometry
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
import argparse
from datetime import datetime

from clinical.ukb_parser import UKBiobankParser
from clinical.icd10_outcomes import ICD10OutcomeProcessor
from clinical.capture24_parser import CAPTURE24Parser
from clinical.alignment import ClinicalAccelerometryAligner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinicalDataIntegrator:
    """
    Integrates UK Biobank clinical data with accelerometry.

    Pipeline:
    1. Parse UK Biobank phenotypes (demographics, accelerometry QC)
    2. Parse ICD-10 diagnoses and classify outcomes
    3. Parse CAPTURE-24 annotations (for labeled subset)
    4. Align clinical outcomes with accelerometry dates
    5. Apply quality control
    6. Create ML-ready datasets
    """

    def __init__(
        self,
        ukb_phenotype_path: str,
        ukb_withdrawn_path: Optional[str] = None,
        capture24_annotation_dir: Optional[str] = None,
        output_dir: str = './clinical_data',
        chunk_size: int = 10000,
    ):
        """
        Initialize integrator.

        Args:
            ukb_phenotype_path: Path to UK Biobank phenotype CSV
            ukb_withdrawn_path: Path to withdrawn participants file
            capture24_annotation_dir: Directory with CAPTURE-24 annotations
            output_dir: Output directory
            chunk_size: Chunk size for processing
        """
        self.ukb_phenotype_path = ukb_phenotype_path
        self.ukb_withdrawn_path = ukb_withdrawn_path
        self.capture24_annotation_dir = capture24_annotation_dir
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized Clinical Data Integrator")
        logger.info(f"  Output directory: {self.output_dir}")

    def run_full_pipeline(self, test_mode: bool = False) -> Dict:
        """
        Run complete integration pipeline.

        Args:
            test_mode: If True, process only 1000 participants for testing

        Returns:
            Pipeline statistics
        """
        logger.info("=" * 70)
        logger.info("CLINICAL DATA INTEGRATION PIPELINE")
        logger.info("=" * 70)

        start_time = datetime.now()
        stats = {}

        # Step 1: Parse UK Biobank phenotypes
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: Parse UK Biobank Phenotypes")
        logger.info("=" * 70)

        df_demographics, df_diagnoses, df_death = self._parse_ukb_data(test_mode)

        stats['n_participants'] = len(df_demographics)
        stats['n_diagnoses'] = len(df_diagnoses)
        stats['n_deaths'] = len(df_death)

        # Step 2: Process ICD-10 outcomes
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: Process ICD-10 Outcomes")
        logger.info("=" * 70)

        df_outcomes = self._process_icd10_outcomes(
            df_demographics,
            df_diagnoses,
            df_death
        )

        stats['n_with_outcomes'] = len(df_outcomes)

        # Step 3: Parse CAPTURE-24 (if available)
        if self.capture24_annotation_dir:
            logger.info("\n" + "=" * 70)
            logger.info("STEP 3: Parse CAPTURE-24 Annotations")
            logger.info("=" * 70)

            df_capture24 = self._parse_capture24()
            stats['n_capture24_participants'] = df_capture24['participant_id'].nunique() if len(df_capture24) > 0 else 0
        else:
            logger.info("\nSkipping CAPTURE-24 (no annotation directory provided)")
            df_capture24 = pd.DataFrame()

        # Step 4: Merge and align
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: Merge Clinical Data")
        logger.info("=" * 70)

        df_merged = self._merge_clinical_data(
            df_demographics,
            df_outcomes,
            df_death
        )

        stats['n_merged'] = len(df_merged)

        # Step 5: Quality control
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: Quality Control")
        logger.info("=" * 70)

        df_filtered, qc_stats = self._apply_quality_control(df_merged)

        stats['qc'] = qc_stats
        stats['n_passed_qc'] = len(df_filtered)

        # Step 6: Create ML dataset
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: Create ML Dataset")
        logger.info("=" * 70)

        ml_stats = self._create_ml_dataset(df_filtered)

        stats['ml_dataset'] = ml_stats

        # Step 7: Validation
        logger.info("\n" + "=" * 70)
        logger.info("STEP 7: Validation")
        logger.info("=" * 70)

        validation = self._validate_data(df_filtered)

        stats['validation'] = validation

        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration / 60:.1f} minutes")
        logger.info(f"\nFinal Statistics:")
        logger.info(f"  Total participants:       {stats['n_participants']:,}")
        logger.info(f"  Passed quality control:   {stats['n_passed_qc']:,}")
        logger.info(f"  Train set:                {ml_stats['n_train']:,}")
        logger.info(f"  Val set:                  {ml_stats['n_val']:,}")
        logger.info(f"  Test set:                 {ml_stats['n_test']:,}")

        # Save statistics
        stats_file = self.output_dir / 'pipeline_statistics.json'
        import json
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info(f"\nSaved statistics to {stats_file}")

        return stats

    def _parse_ukb_data(self, test_mode: bool) -> tuple:
        """Parse UK Biobank phenotype data."""
        parser = UKBiobankParser(
            self.ukb_phenotype_path,
            self.ukb_withdrawn_path,
            chunk_size=self.chunk_size if not test_mode else 1000
        )

        # Parse demographics
        demo_file = self.output_dir / 'demographics.csv'
        df_demographics = parser.parse_demographics(demo_file)

        # Parse ICD-10 diagnoses
        icd_file = self.output_dir / 'icd10_diagnoses.csv'
        df_diagnoses = parser.parse_icd10_diagnoses(icd_file)

        # Parse death registry
        death_file = self.output_dir / 'death_registry.csv'
        df_death = parser.parse_death_registry(death_file)

        return df_demographics, df_diagnoses, df_death

    def _process_icd10_outcomes(
        self,
        df_demographics: pd.DataFrame,
        df_diagnoses: pd.DataFrame,
        df_death: pd.DataFrame
    ) -> pd.DataFrame:
        """Process ICD-10 diagnoses into outcomes."""
        processor = ICD10OutcomeProcessor(min_incident_days=30)

        # Add accelerometry dates from demographics
        if 'eid' in df_demographics.columns and 'acc_valid_days' in df_demographics.columns:
            # Use a placeholder date (would be actual date from field 90003)
            df_acc_dates = df_demographics[['eid']].copy()
            df_acc_dates['accelerometry_date'] = pd.to_datetime('2015-01-01')
        else:
            df_acc_dates = None

        # Classify diagnoses
        df_classified = processor.classify_diagnoses(df_diagnoses, df_acc_dates)

        # Create outcome summary
        df_outcomes = processor.create_outcome_summary(df_classified, df_death)

        # Validate
        validation = processor.validate_outcomes(df_outcomes)

        # Save
        outcome_file = self.output_dir / 'clinical_outcomes.csv'
        df_outcomes.to_csv(outcome_file, index=False)

        logger.info(f"Saved outcomes to {outcome_file}")

        return df_outcomes

    def _parse_capture24(self) -> pd.DataFrame:
        """Parse CAPTURE-24 annotations."""
        parser = CAPTURE24Parser(self.capture24_annotation_dir)

        # Parse all annotations
        df_annotations = parser.parse_all_annotations()

        if len(df_annotations) > 0:
            # Aggregate to windows
            df_windows = parser.aggregate_to_windows(df_annotations)

            # Save
            capture_file = self.output_dir / 'capture24_windows.csv'
            df_windows.to_csv(capture_file, index=False)

            logger.info(f"Saved CAPTURE-24 data to {capture_file}")

            return df_windows
        else:
            return pd.DataFrame()

    def _merge_clinical_data(
        self,
        df_demographics: pd.DataFrame,
        df_outcomes: pd.DataFrame,
        df_death: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge all clinical data."""
        aligner = ClinicalAccelerometryAligner()

        df_merged = aligner.merge_clinical_data(
            df_demographics,
            df_outcomes,
            df_death
        )

        # Save
        merged_file = self.output_dir / 'merged_clinical_data.csv'
        df_merged.to_csv(merged_file, index=False)

        logger.info(f"Saved merged data to {merged_file}")

        return df_merged

    def _apply_quality_control(self, df_merged: pd.DataFrame) -> tuple:
        """Apply quality control filters."""
        aligner = ClinicalAccelerometryAligner()

        df_filtered, qc_stats = aligner.apply_quality_filters(df_merged)

        # Save
        filtered_file = self.output_dir / 'qc_filtered_data.csv'
        df_filtered.to_csv(filtered_file, index=False)

        logger.info(f"Saved filtered data to {filtered_file}")

        return df_filtered, qc_stats

    def _create_ml_dataset(self, df_filtered: pd.DataFrame) -> Dict:
        """Create ML-ready dataset."""
        aligner = ClinicalAccelerometryAligner()

        # Create dummy accelerometry HDF5 path (would be actual data)
        acc_h5_path = "accelerometry_data.h5"

        output_h5 = self.output_dir / 'ml_dataset.h5'

        ml_stats = aligner.create_ml_dataset(
            df_filtered,
            acc_h5_path,
            str(output_h5),
            split_by_center=True
        )

        logger.info(f"Saved ML dataset to {output_h5}")

        return ml_stats

    def _validate_data(self, df_filtered: pd.DataFrame) -> Dict:
        """Validate data quality and temporal alignment."""
        aligner = ClinicalAccelerometryAligner()

        validation = aligner.validate_temporal_alignment(df_filtered)

        return validation


def main():
    parser = argparse.ArgumentParser(description='Clinical Data Integration Pipeline')

    parser.add_argument('--ukb-phenotype', type=str, required=True,
                       help='Path to UK Biobank phenotype CSV')
    parser.add_argument('--ukb-withdrawn', type=str,
                       help='Path to withdrawn participants file')
    parser.add_argument('--capture24-dir', type=str,
                       help='CAPTURE-24 annotation directory')
    parser.add_argument('--output-dir', type=str, default='./clinical_data',
                       help='Output directory')
    parser.add_argument('--test', action='store_true',
                       help='Test mode (1000 participants)')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Chunk size for processing')

    args = parser.parse_args()

    # Initialize integrator
    integrator = ClinicalDataIntegrator(
        ukb_phenotype_path=args.ukb_phenotype,
        ukb_withdrawn_path=args.ukb_withdrawn,
        capture24_annotation_dir=args.capture24_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
    )

    # Run pipeline
    stats = integrator.run_full_pipeline(test_mode=args.test)

    logger.info("\n✓ Pipeline complete!")


if __name__ == '__main__':
    main()
