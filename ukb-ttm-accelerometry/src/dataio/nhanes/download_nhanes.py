#!/usr/bin/env python3
"""
Download NHANES accelerometry data directly using Python.

No R dependencies required! Downloads XPT files directly from CDC servers.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict
import logging
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
import time

import pandas as pd
import pyreadstat
from tqdm import tqdm


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


# NHANES accelerometry file mappings
NHANES_FILES = {
    # Raw 80 Hz data (2011-2014)
    '2011-2012': {
        'raw_80hz': 'PAXRAW_G',
        'minute': 'PAXMIN_G',
        'demo': 'DEMO_G',
        'bmx': 'BMX_G',
        'bpx': 'BPX_G'
    },
    '2013-2014': {
        'raw_80hz': 'PAXRAW_H',
        'minute': 'PAXMIN_H',
        'demo': 'DEMO_H',
        'bmx': 'BMX_H',
        'bpx': 'BPX_H'
    },
    # Minute-level only (2003-2006)
    '2003-2004': {
        'minute': 'PAXMIN_C',
        'demo': 'DEMO_C',
        'bmx': 'BMX_C',
        'bpx': 'BPX_C'
    },
    '2005-2006': {
        'minute': 'PAXMIN_D',
        'demo': 'DEMO_D',
        'bmx': 'BMX_D',
        'bpx': 'BPX_D'
    }
}


def construct_nhanes_url(file_name: str) -> str:
    """
    Construct CDC NHANES URL for a given file.

    Args:
        file_name: NHANES file name (e.g., 'PAXRAW_G', 'DEMO_H')

    Returns:
        Full URL to XPT file
    """
    base_url = "https://wwwn.cdc.gov/Nchs/Nhanes"

    # Determine cycle from suffix
    suffix = file_name.split('_')[-1]
    cycle_map = {
        'C': '2003-2004',
        'D': '2005-2006',
        'G': '2011-2012',
        'H': '2013-2014'
    }
    cycle = cycle_map.get(suffix, '2011-2012')

    # Construct URL
    url = f"{base_url}/{cycle}/{file_name}.XPT"

    return url


def download_file_with_progress(url: str, output_path: Path, desc: str = "Downloading") -> bool:
    """
    Download file with progress bar.

    Args:
        url: URL to download from
        output_path: Where to save the file
        desc: Description for progress bar

    Returns:
        True if successful
    """
    try:
        # Get file size
        import urllib.request
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req) as response:
            file_size = int(response.headers.get('Content-Length', 0))

        # Download with progress bar
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=desc) as pbar:
            def reporthook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                pbar.update(downloaded - pbar.n)

            urlretrieve(url, output_path, reporthook=reporthook)

        return True

    except (URLError, HTTPError) as e:
        logging.error(f"Failed to download {url}: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error downloading {url}: {e}")
        return False


def download_nhanes_file(
    file_name: str,
    output_dir: Path,
    force: bool = False
) -> Optional[Path]:
    """
    Download a single NHANES XPT file.

    Args:
        file_name: NHANES file name (e.g., 'PAXRAW_G')
        output_dir: Output directory
        force: Force re-download even if file exists

    Returns:
        Path to downloaded file, or None if failed
    """
    logger = logging.getLogger(__name__)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{file_name}.XPT"

    # Check if already downloaded
    if output_file.exists() and not force:
        logger.info(f"  ✓ {file_name}: Already downloaded")
        return output_file

    # Construct URL
    url = construct_nhanes_url(file_name)

    # Download
    logger.info(f"  Downloading {file_name}...")
    success = download_file_with_progress(url, output_file, desc=file_name)

    if success:
        logger.info(f"  ✓ {file_name}: Downloaded ({output_file.stat().st_size / 1e6:.1f} MB)")
        return output_file
    else:
        return None


def download_cycle_data(
    cycle: str,
    output_dir: Path,
    data_types: List[str] = None,
    force: bool = False
) -> Dict[str, Path]:
    """
    Download all files for a specific cycle.

    Args:
        cycle: NHANES cycle (e.g., '2011-2012')
        output_dir: Output directory
        data_types: List of data types to download (e.g., ['raw_80hz', 'demo'])
                   If None, downloads all available
        force: Force re-download

    Returns:
        Dict mapping data type to downloaded file path
    """
    logger = logging.getLogger(__name__)

    if cycle not in NHANES_FILES:
        logger.error(f"Unknown cycle: {cycle}")
        return {}

    cycle_files = NHANES_FILES[cycle]

    # Determine which files to download
    if data_types is None:
        data_types = list(cycle_files.keys())

    # Create cycle directory
    cycle_dir = output_dir / cycle.replace('-', '_')
    cycle_dir.mkdir(parents=True, exist_ok=True)

    # Download each file
    downloaded_files = {}

    for data_type in data_types:
        if data_type not in cycle_files:
            logger.warning(f"  ✗ {data_type}: Not available for {cycle}")
            continue

        file_name = cycle_files[data_type]
        file_path = download_nhanes_file(file_name, cycle_dir, force)

        if file_path:
            downloaded_files[data_type] = file_path

    return downloaded_files


def load_xpt_to_dataframe(xpt_path: Path) -> pd.DataFrame:
    """
    Load XPT file into pandas DataFrame.

    Args:
        xpt_path: Path to XPT file

    Returns:
        DataFrame with data
    """
    df, meta = pyreadstat.read_xport(str(xpt_path))
    return df


def extract_participant_ids(xpt_path: Path) -> List[str]:
    """
    Extract unique participant IDs (SEQN) from XPT file.

    Args:
        xpt_path: Path to XPT file

    Returns:
        List of participant IDs
    """
    df = load_xpt_to_dataframe(xpt_path)

    if 'SEQN' in df.columns:
        return df['SEQN'].unique().astype(str).tolist()
    else:
        return []


def create_manifest(
    downloaded_files: Dict[str, Dict[str, Path]],
    output_path: Path
) -> pd.DataFrame:
    """
    Create manifest CSV of downloaded files.

    Args:
        downloaded_files: Dict mapping cycle -> data_type -> file_path
        output_path: Where to save manifest

    Returns:
        Manifest DataFrame
    """
    manifest_data = []

    for cycle, files in downloaded_files.items():
        for data_type, file_path in files.items():
            manifest_data.append({
                'cycle': cycle,
                'data_type': data_type,
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_size_mb': file_path.stat().st_size / 1e6
            })

    manifest_df = pd.DataFrame(manifest_data)
    manifest_df.to_csv(output_path, index=False)

    return manifest_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download NHANES accelerometry data (pure Python, no R!)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/nhanes/downloads',
        help='Output directory for downloaded files'
    )

    parser.add_argument(
        '--cycles',
        type=str,
        default='2011-2012,2013-2014',
        help='Comma-separated list of cycles'
    )

    parser.add_argument(
        '--data-types',
        type=str,
        default='raw_80hz,minute,demo',
        help='Comma-separated data types: raw_80hz, minute, demo, bmx, bpx'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if files exist'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logging(args.log_level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cycles = [c.strip() for c in args.cycles.split(',')]
    data_types = [dt.strip() for dt in args.data_types.split(',')]

    logger.info("=" * 60)
    logger.info("NHANES Data Downloader (Pure Python)")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Cycles: {cycles}")
    logger.info(f"Data types: {data_types}")
    logger.info("")

    # Download data for each cycle
    all_downloaded = {}

    for cycle in cycles:
        logger.info(f"Downloading {cycle}...")

        downloaded_files = download_cycle_data(
            cycle,
            output_dir,
            data_types,
            args.force
        )

        if downloaded_files:
            all_downloaded[cycle] = downloaded_files
            logger.info(f"  ✓ Downloaded {len(downloaded_files)} files for {cycle}")
        else:
            logger.warning(f"  ✗ No files downloaded for {cycle}")

        logger.info("")

    # Create manifest
    if all_downloaded:
        manifest_path = output_dir / "manifest.csv"
        manifest_df = create_manifest(all_downloaded, manifest_path)

        logger.info("=" * 60)
        logger.info("Download Complete!")
        logger.info("=" * 60)
        logger.info(f"Total files: {len(manifest_df)}")
        logger.info(f"Total size: {manifest_df['file_size_mb'].sum():.1f} MB")
        logger.info(f"Manifest: {manifest_path}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. For 80 Hz: python -m src.dataio.nhanes.parse_80hz_xpt --input {output_dir}")
        logger.info("  2. For minute: python -m src.dataio.nhanes.parse_minute --input {output_dir}")
    else:
        logger.error("No files downloaded!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
