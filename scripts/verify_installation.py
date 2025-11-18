#!/usr/bin/env python3
"""
Verify installation and basic functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("=" * 70)
print("TTM ACCELEROMETRY SYSTEM - INSTALLATION VERIFICATION")
print("=" * 70)

# Test 1: Import core modules
print("\n1. Testing module imports...")
try:
    from data.data_loader import AccelerometryDataLoader, StreamingDataset
    from data.augmentations import SSLAugmentations
    from models.ttm_classifier import TTMAccelerometryClassifier, FocalLoss
    from training.trainer import ThreeStageTrainer
    from evaluation.evaluator import AccelerometryEvaluator
    from utils.config import Config
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create configuration
print("\n2. Testing configuration...")
try:
    config = Config()
    print(f"   ✓ Configuration created")
    print(f"     - Window size: {config.data.window_size}")
    print(f"     - Batch size: {config.training.batch_size}")
except Exception as e:
    print(f"   ✗ Config failed: {e}")
    sys.exit(1)

# Test 3: Test model initialization
print("\n3. Testing model initialization...")
try:
    import torch
    model = TTMAccelerometryClassifier(
        n_classes=4,
        n_channels=3,
        context_length=512,
        hidden_dim=256,
        dropout=0.3,
    )
    stats = model.get_parameter_stats()
    print(f"   ✓ Model initialized")
    print(f"     - Total parameters: {stats['total']:,}")
    print(f"     - Trainable parameters: {stats['trainable']:,}")
except Exception as e:
    print(f"   ✗ Model initialization failed: {e}")
    sys.exit(1)

# Test 4: Test forward pass
print("\n4. Testing forward pass...")
try:
    x = torch.randn(2, 820, 3)
    output = model(x)
    print(f"   ✓ Forward pass successful")
    print(f"     - Input shape: {x.shape}")
    print(f"     - Output shape: {output['logits'].shape}")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 5: Test augmentations
print("\n5. Testing SSL augmentations...")
try:
    import numpy as np
    ssl_aug = SSLAugmentations()
    x = np.random.randn(820, 3).astype(np.float32)
    x_aug, labels = ssl_aug.apply_all(x, return_labels=True)
    print(f"   ✓ Augmentations working")
    print(f"     - Arrow of Time: {labels['arrow_of_time']}")
    print(f"     - Permutation: {labels['permutation']}")
except Exception as e:
    print(f"   ✗ Augmentations failed: {e}")
    sys.exit(1)

# Test 6: Test focal loss
print("\n6. Testing focal loss...")
try:
    logits = torch.randn(10, 4)
    targets = torch.randint(0, 4, (10,))
    criterion = FocalLoss(gamma=2.0)
    loss = criterion(logits, targets)
    print(f"   ✓ Focal loss working")
    print(f"     - Loss value: {loss.item():.4f}")
except Exception as e:
    print(f"   ✗ Focal loss failed: {e}")
    sys.exit(1)

# Test 7: Test data loader (without actual data)
print("\n7. Testing data loader initialization...")
try:
    loader = AccelerometryDataLoader(
        data_dir='./data',
        cache_dir='./cache',
        window_size=820,
        stride=410,
    )
    print(f"   ✓ Data loader initialized")
except Exception as e:
    print(f"   ✗ Data loader failed: {e}")
    sys.exit(1)

# Test 8: Check GPU availability
print("\n8. Checking GPU availability...")
try:
    if torch.cuda.is_available():
        print(f"   ✓ GPU available")
        print(f"     - Device: {torch.cuda.get_device_name(0)}")
        print(f"     - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"   ⚠ No GPU available (will use CPU)")
except Exception as e:
    print(f"   ⚠ GPU check failed: {e}")

# Test 9: Check dependencies
print("\n9. Checking key dependencies...")
dependencies = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'h5py': 'HDF5',
    'sklearn': 'scikit-learn',
    'scipy': 'SciPy',
}

all_deps_ok = True
for module_name, display_name in dependencies.items():
    try:
        __import__(module_name)
        print(f"   ✓ {display_name}")
    except ImportError:
        print(f"   ✗ {display_name} not found")
        all_deps_ok = False

# Summary
print("\n" + "=" * 70)
if all_deps_ok:
    print("✓ INSTALLATION VERIFIED - SYSTEM READY")
else:
    print("⚠ SOME DEPENDENCIES MISSING - Install with: pip install -r requirements.txt")
print("=" * 70)
print("\nNext steps:")
print("  1. Prepare your UK Biobank .cwa data")
print("  2. Run: python scripts/train.py --data <path_to_data.h5>")
print("  3. Or open the Colab notebook: notebooks/TTM_Accelerometry_Training.ipynb")
print("=" * 70 + "\n")
