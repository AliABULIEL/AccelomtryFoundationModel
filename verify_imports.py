#!/usr/bin/env python3
# FILE: verify_imports.py

"""Quick script to verify all imports work correctly."""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    print()

    try:
        print("1. Testing config module...")
        from ukb_ttm_accel.config import TrainingConfig, load_config_from_yaml, save_config_to_yaml
        print("   ✓ config module imported successfully")

        print("\n2. Testing utils modules...")
        from ukb_ttm_accel.utils import setup_logger, set_global_seed, is_colab
        print("   ✓ utils modules imported successfully")

        print("\n3. Testing data modules...")
        from ukb_ttm_accel.data import (
            resample_and_calibrate,
            create_windows,
            zscore_normalize_windows,
            load_clinical_data,
            build_label_and_metadata,
            UKBAccelDataset,
            accel_collate_fn,
        )
        print("   ✓ data modules imported successfully")

        print("\n4. Testing configuration creation...")
        config = TrainingConfig()
        print(f"   ✓ Created config with batch_size={config.batch_size}")

        print("\n5. Testing logger...")
        logger = setup_logger("test")
        logger.info("Test log message")
        print("   ✓ Logger works correctly")

        print("\n" + "="*60)
        print("ALL IMPORTS SUCCESSFUL!")
        print("="*60)
        print("\nThe package is correctly installed and ready to use.")

        return True

    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
