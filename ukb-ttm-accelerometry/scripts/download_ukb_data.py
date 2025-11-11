#!/usr/bin/env python3
"""
Download UK Biobank accelerometry data using official tools.

IMPORTANT: This script requires:
1. Approved UK Biobank access (Application: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access)
2. UK Biobank key file (.ukbkey)
3. ukbfetch and ukbunpack utilities installed

For testing without real data, use: scripts/generate_demo_data.py
"""
import argparse
import subprocess
import sys
from pathlib import Path
import logging
import json


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def check_ukb_tools():
    """Check if UK Biobank tools are installed."""
    tools = ['ukbfetch', 'ukbunpack']
    missing = []

    for tool in tools:
        try:
            subprocess.run([tool, '--help'], capture_output=True, check=False)
        except FileNotFoundError:
            missing.append(tool)

    return missing


def download_with_ukbfetch(
    key_file: Path,
    output_dir: Path,
    dataset_id: str,
    participant_ids: list = None,
    logger: logging.Logger = None
) -> bool:
    """
    Download data using ukbfetch tool.

    Args:
        key_file: Path to .ukbkey file
        output_dir: Output directory
        dataset_id: UK Biobank dataset ID
        participant_ids: Optional list of specific participant IDs
        logger: Logger instance

    Returns:
        True if successful
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build ukbfetch command
    cmd = [
        'ukbfetch',
        '-a', str(key_file),
        '-e', dataset_id,
        '-o', str(output_dir)
    ]

    if participant_ids:
        # Create temporary file with participant IDs
        id_file = output_dir / 'participant_ids.txt'
        with open(id_file, 'w') as f:
            f.write('\n'.join(map(str, participant_ids)))
        cmd.extend(['-i', str(id_file)])

    logger.info(f"Running: {' '.join(cmd)}")
    logger.info("This may take a while depending on dataset size...")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Download completed successfully")
        logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        logger.error(e.stderr)
        return False


def unpack_data(
    input_dir: Path,
    output_dir: Path,
    logger: logging.Logger = None
) -> bool:
    """
    Unpack downloaded UK Biobank data.

    Args:
        input_dir: Directory containing downloaded .enc files
        output_dir: Output directory for unpacked files
        logger: Logger instance

    Returns:
        True if successful
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find .enc files
    enc_files = list(input_dir.glob('*.enc'))

    if not enc_files:
        logger.warning(f"No .enc files found in {input_dir}")
        return False

    logger.info(f"Found {len(enc_files)} encrypted files to unpack")

    for enc_file in enc_files:
        logger.info(f"Unpacking {enc_file.name}...")

        cmd = ['ukbunpack', str(enc_file), str(output_dir)]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"  ✓ Unpacked {enc_file.name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"  ✗ Failed to unpack {enc_file.name}: {e}")
            return False

    logger.info("All files unpacked successfully")
    return True


def download_accelerometry_data(
    key_file: Path,
    output_dir: Path,
    dataset_id: str = None,
    participant_ids: list = None,
    auto_unpack: bool = True,
    logger: logging.Logger = None
) -> dict:
    """
    Complete workflow to download and unpack UK Biobank accelerometry data.

    Args:
        key_file: Path to .ukbkey file
        output_dir: Output directory
        dataset_id: Dataset ID (if None, will prompt)
        participant_ids: List of participant IDs to download
        auto_unpack: Automatically unpack after download
        logger: Logger instance

    Returns:
        Dictionary with download statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Create directories
    download_dir = output_dir / 'downloaded'
    unpacked_dir = output_dir / 'unpacked'
    raw_cwa_dir = output_dir / 'raw_cwa'

    stats = {
        'download_success': False,
        'unpack_success': False,
        'n_files_downloaded': 0,
        'n_files_unpacked': 0
    }

    # Download
    logger.info("="*60)
    logger.info("Step 1: Downloading data from UK Biobank")
    logger.info("="*60)

    success = download_with_ukbfetch(
        key_file,
        download_dir,
        dataset_id,
        participant_ids,
        logger
    )

    stats['download_success'] = success
    stats['n_files_downloaded'] = len(list(download_dir.glob('*.enc')))

    if not success:
        return stats

    # Unpack
    if auto_unpack:
        logger.info("\n" + "="*60)
        logger.info("Step 2: Unpacking encrypted files")
        logger.info("="*60)

        success = unpack_data(download_dir, unpacked_dir, logger)
        stats['unpack_success'] = success
        stats['n_files_unpacked'] = len(list(unpacked_dir.glob('*.cwa')))

        if success:
            # Move .cwa files to raw directory
            logger.info(f"\nOrganizing .cwa files to {raw_cwa_dir}")
            raw_cwa_dir.mkdir(parents=True, exist_ok=True)

            for cwa_file in unpacked_dir.glob('*.cwa'):
                dest = raw_cwa_dir / cwa_file.name
                cwa_file.rename(dest)
                logger.info(f"  Moved: {cwa_file.name}")

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download UK Biobank accelerometry data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
REQUIREMENTS:
  1. UK Biobank access approval
  2. .ukbkey file from UK Biobank
  3. ukbfetch and ukbunpack tools installed

INSTALLATION OF UK BIOBANK TOOLS:
  # Download from: https://biobank.ctsu.ox.ac.uk/crystal/download.cgi
  # Or use:
  wget -nd biobank.ctsu.ox.ac.uk/crystal/util/ukbfetch
  wget -nd biobank.ctsu.ox.ac.uk/crystal/util/ukbunpack
  chmod +x ukbfetch ukbunpack
  sudo mv ukbfetch ukbunpack /usr/local/bin/

GETTING ACCESS:
  1. Apply: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access
  2. Wait for approval (can take weeks/months)
  3. Download .ukbkey file from approved application
  4. Use this script to download data

FOR TESTING WITHOUT REAL DATA:
  Use: python scripts/generate_demo_data.py

EXAMPLES:
  # Download all accelerometry data for your application
  python scripts/download_ukb_data.py \\
      --key-file /path/to/your.ukbkey \\
      --dataset-id your_dataset_id \\
      --output-dir ./data/ukb

  # Download specific participants only
  python scripts/download_ukb_data.py \\
      --key-file your.ukbkey \\
      --dataset-id your_dataset_id \\
      --participant-ids 1000001 1000002 1000003 \\
      --output-dir ./data/ukb
        """
    )

    parser.add_argument(
        '--key-file',
        type=str,
        required=True,
        help='Path to UK Biobank .ukbkey file'
    )

    parser.add_argument(
        '--dataset-id',
        type=str,
        required=True,
        help='UK Biobank dataset/application ID'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/ukb_raw',
        help='Output directory for downloaded data'
    )

    parser.add_argument(
        '--participant-ids',
        type=str,
        nargs='+',
        help='Specific participant IDs to download (optional)'
    )

    parser.add_argument(
        '--no-unpack',
        action='store_true',
        help='Skip automatic unpacking of downloaded files'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("UK Biobank Accelerometry Data Download")
    logger.info("="*60)

    # Check if key file exists
    key_file = Path(args.key_file)
    if not key_file.exists():
        logger.error(f"Key file not found: {key_file}")
        logger.error("\nTo get a key file:")
        logger.error("  1. Log in to UK Biobank Access Management System")
        logger.error("  2. Navigate to your approved application")
        logger.error("  3. Download the .ukbkey file")
        return 1

    # Check if UK Biobank tools are installed
    logger.info("Checking for UK Biobank tools...")
    missing_tools = check_ukb_tools()

    if missing_tools:
        logger.error(f"Missing required tools: {', '.join(missing_tools)}")
        logger.error("\nInstallation instructions:")
        logger.error("  wget -nd biobank.ctsu.ox.ac.uk/crystal/util/ukbfetch")
        logger.error("  wget -nd biobank.ctsu.ox.ac.uk/crystal/util/ukbunpack")
        logger.error("  chmod +x ukbfetch ukbunpack")
        logger.error("  sudo mv ukbfetch ukbunpack /usr/local/bin/")
        return 1

    logger.info("✓ All required tools found")

    # Download data
    output_dir = Path(args.output_dir)

    try:
        stats = download_accelerometry_data(
            key_file,
            output_dir,
            args.dataset_id,
            args.participant_ids,
            auto_unpack=not args.no_unpack,
            logger=logger
        )

        # Save statistics
        stats_file = output_dir / 'download_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("="*60)
        logger.info(f"Files downloaded: {stats['n_files_downloaded']}")
        logger.info(f"Files unpacked: {stats['n_files_unpacked']}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Statistics saved to: {stats_file}")

        if stats['unpack_success']:
            logger.info("\n" + "="*60)
            logger.info("NEXT STEPS")
            logger.info("="*60)
            logger.info("1. Process the data:")
            logger.info(f"   python scripts/prepare_ukb.py \\")
            logger.info(f"       --input {output_dir}/raw_cwa \\")
            logger.info(f"       --outdir ./data/processed")
            logger.info("")
            logger.info("2. Create train/val/test splits:")
            logger.info("   python scripts/make_splits.py \\")
            logger.info("       --data-dir ./data/processed \\")
            logger.info("       --output-dir ./data/splits")

        return 0 if stats['download_success'] else 1

    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Download failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
