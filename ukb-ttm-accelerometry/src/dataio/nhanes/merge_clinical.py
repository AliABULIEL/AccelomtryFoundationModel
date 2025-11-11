"""
Download and merge NHANES clinical/demographic data with accelerometry.

Handles:
- Demographics (DEMO)
- Body measures (BMX)
- Blood pressure (BPX)
- Lab results
- Variable name harmonization across cycles
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import pyreadstat


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_xpt_file(filepath: Path) -> pd.DataFrame:
    """
    Parse SAS XPT file using pyreadstat.

    Args:
        filepath: Path to XPT file

    Returns:
        DataFrame with parsed data
    """
    try:
        df, meta = pyreadstat.read_xport(str(filepath))
        return df
    except Exception as e:
        logging.error(f"Error reading {filepath}: {e}")
        return pd.DataFrame()


def download_clinical_data(
    cycles: List[str],
    download_dir: Path,
    tables: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load NHANES clinical data from XPT files (no R required!).

    Assumes XPT files have been downloaded by download_nhanes.py.

    Args:
        cycles: List of cycle strings like '2011-2012'
        download_dir: Directory containing downloaded XPT files
        tables: List of table prefixes to download (default: DEMO, BMX, BPX)

    Returns:
        Dictionary mapping table_cycle to DataFrame
    """
    if tables is None:
        tables = ['DEMO', 'BMX', 'BPX']

    logger = logging.getLogger(__name__)

    # Map cycles to letter codes
    cycle_map = {
        '2003-2004': 'C',
        '2005-2006': 'D',
        '2011-2012': 'G',
        '2013-2014': 'H'
    }

    downloaded_data = {}

    for cycle in cycles:
        if cycle not in cycle_map:
            logger.warning(f"Unknown cycle: {cycle}")
            continue

        letter = cycle_map[cycle]
        cycle_dir = download_dir / cycle.replace('-', '_')

        for table in tables:
            table_name = f"{table}_{letter}"
            logger.info(f"Loading {table_name}...")

            # Find XPT file
            xpt_file = cycle_dir / f"{table_name}.XPT"

            if not xpt_file.exists():
                logger.warning(f"  ✗ {table_name}: Not found at {xpt_file}")
                logger.warning(f"     Download it first with:")
                logger.warning(f"     python -m src.dataio.nhanes.download_nhanes --cycles {cycle} --data-types {table.lower()}")
                continue

            try:
                df = parse_xpt_file(xpt_file)
                if not df.empty:
                    downloaded_data[table_name] = df
                    logger.info(f"  ✓ {table_name}: {len(df)} rows")
                else:
                    logger.warning(f"  ✗ {table_name}: Empty dataframe")

            except Exception as e:
                logger.error(f"  ✗ {table_name}: {e}")

    return downloaded_data


def normalize_clinical_variables(
    dataframes: Dict[str, pd.DataFrame],
    participant_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalize clinical variables across cycles.

    Harmonizes variable names and merges data from multiple cycles.

    Args:
        dataframes: Dict mapping table_cycle to DataFrame
        participant_ids: Optional list to filter participants

    Returns:
        Normalized DataFrame with columns:
            participant_id, cycle, age, sex, race_eth, bmi, sbp, dbp, ...
    """
    logger = logging.getLogger(__name__)

    # Variable name mappings (cycle-specific variations)
    var_mappings = {
        'SEQN': 'participant_id',
        'RIDAGEYR': 'age',
        'RIAGENDR': 'sex',
        'RIDRETH1': 'race_eth',
        'RIDRETH3': 'race_eth',  # Different cycles use different codes
        'BMXBMI': 'bmi',
        'BMXWT': 'weight_kg',
        'BMXHT': 'height_cm',
        'BPXSY1': 'sbp',
        'BPXDI1': 'dbp'
    }

    normalized_list = []

    # Process DEMO tables first to get participant list
    demo_dfs = {k: v for k, v in dataframes.items() if k.startswith('DEMO')}

    for table_name, df in demo_dfs.items():
        # Extract cycle letter
        cycle_letter = table_name.split('_')[-1]
        cycle_map_rev = {
            'C': '2003-2004',
            'D': '2005-2006',
            'G': '2011-2012',
            'H': '2013-2014'
        }
        cycle = cycle_map_rev.get(cycle_letter, 'unknown')

        # Rename columns
        df_norm = df.copy()
        for old_name, new_name in var_mappings.items():
            if old_name in df_norm.columns:
                df_norm = df_norm.rename(columns={old_name: new_name})

        # Ensure participant_id exists
        if 'participant_id' not in df_norm.columns:
            logger.warning(f"No SEQN in {table_name}")
            continue

        df_norm['participant_id'] = df_norm['participant_id'].astype(str)
        df_norm['cycle'] = cycle

        # Filter to requested participants
        if participant_ids is not None:
            df_norm = df_norm[df_norm['participant_id'].isin(participant_ids)]

        # Select relevant columns
        cols_to_keep = ['participant_id', 'cycle']
        for col in ['age', 'sex', 'race_eth']:
            if col in df_norm.columns:
                cols_to_keep.append(col)

        df_norm = df_norm[cols_to_keep]
        normalized_list.append(df_norm)

    if not normalized_list:
        logger.warning("No DEMO data found")
        return pd.DataFrame()

    # Concatenate demographics
    clinical_df = pd.concat(normalized_list, ignore_index=True)

    # Merge body measures (BMX)
    bmx_dfs = {k: v for k, v in dataframes.items() if k.startswith('BMX')}
    for table_name, df in bmx_dfs.items():
        cycle_letter = table_name.split('_')[-1]
        cycle = cycle_map_rev.get(cycle_letter, 'unknown')

        df_norm = df.copy()
        for old_name, new_name in var_mappings.items():
            if old_name in df_norm.columns:
                df_norm = df_norm.rename(columns={old_name: new_name})

        if 'participant_id' not in df_norm.columns:
            continue

        df_norm['participant_id'] = df_norm['participant_id'].astype(str)

        # Select BMI, weight, height
        cols = ['participant_id']
        for col in ['bmi', 'weight_kg', 'height_cm']:
            if col in df_norm.columns:
                cols.append(col)

        if len(cols) > 1:
            df_norm = df_norm[cols]
            clinical_df = clinical_df.merge(
                df_norm,
                on='participant_id',
                how='left',
                suffixes=('', '_bmx')
            )

    # Merge blood pressure (BPX)
    bpx_dfs = {k: v for k, v in dataframes.items() if k.startswith('BPX')}
    for table_name, df in bpx_dfs.items():
        df_norm = df.copy()
        for old_name, new_name in var_mappings.items():
            if old_name in df_norm.columns:
                df_norm = df_norm.rename(columns={old_name: new_name})

        if 'participant_id' not in df_norm.columns:
            continue

        df_norm['participant_id'] = df_norm['participant_id'].astype(str)

        cols = ['participant_id']
        for col in ['sbp', 'dbp']:
            if col in df_norm.columns:
                cols.append(col)

        if len(cols) > 1:
            df_norm = df_norm[cols]
            clinical_df = clinical_df.merge(
                df_norm,
                on='participant_id',
                how='left',
                suffixes=('', '_bpx')
            )

    # Recode sex: 1=Male, 2=Female
    if 'sex' in clinical_df.columns:
        clinical_df['sex'] = clinical_df['sex'].map({1: 'Male', 2: 'Female'})

    # Create age bins for stratification
    if 'age' in clinical_df.columns:
        clinical_df['age_bin'] = pd.cut(
            clinical_df['age'],
            bins=[0, 18, 40, 60, 80, 120],
            labels=['0-17', '18-39', '40-59', '60-79', '80+']
        )

    logger.info(f"Normalized {len(clinical_df)} participants")

    return clinical_df


def merge_with_accelerometry(
    clinical_df: pd.DataFrame,
    accel_participants: List[str]
) -> pd.DataFrame:
    """
    Filter clinical data to only participants with accelerometry.

    Args:
        clinical_df: Clinical DataFrame with participant_id column
        accel_participants: List of participant IDs with accelerometry data

    Returns:
        Filtered clinical DataFrame
    """
    merged = clinical_df[clinical_df['participant_id'].isin(accel_participants)]
    logger = logging.getLogger(__name__)
    logger.info(f"Merged: {len(merged)} / {len(clinical_df)} have accelerometry")
    return merged


def create_stratification_groups(
    clinical_df: pd.DataFrame
) -> pd.Series:
    """
    Create stratification groups for train/val/test splitting.

    Args:
        clinical_df: Clinical DataFrame

    Returns:
        Series of stratification group labels
    """
    # Combine age_bin, sex, and race_eth if available
    strat_cols = []

    if 'age_bin' in clinical_df.columns:
        strat_cols.append(clinical_df['age_bin'].astype(str))
    elif 'age' in clinical_df.columns:
        age_bins = pd.cut(
            clinical_df['age'],
            bins=[0, 18, 40, 60, 80, 120],
            labels=['0-17', '18-39', '40-59', '60-79', '80+']
        )
        strat_cols.append(age_bins.astype(str))

    if 'sex' in clinical_df.columns:
        strat_cols.append(clinical_df['sex'].astype(str))

    if 'race_eth' in clinical_df.columns:
        strat_cols.append(clinical_df['race_eth'].astype(str))

    if not strat_cols:
        # No stratification variables, return dummy groups
        return pd.Series(['all'] * len(clinical_df), index=clinical_df.index)

    # Combine into single stratification label
    strat_labels = strat_cols[0]
    for col in strat_cols[1:]:
        strat_labels = strat_labels + '_' + col

    return strat_labels


def save_clinical_csv(
    clinical_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save clinical data to CSV.

    Args:
        clinical_df: Clinical DataFrame
        output_path: Output CSV path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clinical_df.to_csv(output_path, index=False)
    logger = logging.getLogger(__name__)
    logger.info(f"Saved clinical data to {output_path}")
