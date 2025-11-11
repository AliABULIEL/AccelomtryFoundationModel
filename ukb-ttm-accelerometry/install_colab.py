"""
Installation helper for Google Colab.

Run this cell at the start of your Colab notebook:
    !wget -q https://raw.githubusercontent.com/yourusername/ukb-ttm-accelerometry/main/install_colab.py
    !python install_colab.py
"""
import subprocess
import sys

def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(e.stderr)
        return False


def main():
    """Install all dependencies for Colab."""

    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  UK Biobank TTM Accelerometry - Colab Installation        ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    # Check Python version
    print(f"Python version: {sys.version}")

    # Mount Google Drive
    print("\nMounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted")
    except Exception as e:
        print(f"Note: Could not mount Drive (may not be in Colab): {e}")

    # Install IBM tsfm
    if not run_command(
        "pip install -q git+https://github.com/IBM/tsfm.git",
        "Installing IBM tsfm from GitHub"
    ):
        print("\n⚠ WARNING: tsfm installation failed")
        print("TTM models will not be available")
        print("Data processing will still work")

    # Clone repository
    repo_exists = False
    try:
        subprocess.run("cd ukb-ttm-accelerometry", shell=True, check=True, capture_output=True)
        repo_exists = True
        print("\n✓ Repository already cloned")
    except:
        if not run_command(
            "git clone https://github.com/yourusername/ukb-ttm-accelerometry.git",
            "Cloning repository"
        ):
            print("\n✗ ERROR: Failed to clone repository")
            print("Please check the repository URL")
            return

    # Change to repository directory
    import os
    os.chdir('ukb-ttm-accelerometry')

    # Install dependencies
    if not run_command(
        "pip install -q -r requirements.txt",
        "Installing dependencies"
    ):
        print("\n✗ ERROR: Failed to install dependencies")
        return

    # Set PyTorch precision
    print("\n" + "="*60)
    print("Configuring PyTorch for Colab")
    print("="*60)
    try:
        import torch
        torch.set_float32_matmul_precision("high")
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"✗ Error configuring PyTorch: {e}")

    # Run basic tests
    print("\n" + "="*60)
    print("Running basic functionality tests")
    print("="*60)

    if run_command("python test_basic.py", "Basic tests"):
        print("""

    ╔════════════════════════════════════════════════════════════╗
    ║              Installation Successful! ✓                    ║
    ╚════════════════════════════════════════════════════════════╝

    Next steps:

    1. Upload your .cwa files to Google Drive

    2. Process data:
       !python scripts/prepare_ukb.py \\
           --input /content/drive/MyDrive/data/raw \\
           --outdir /content/drive/MyDrive/data/processed

    3. Create splits:
       !python scripts/make_splits.py \\
           --data-dir /content/drive/MyDrive/data/processed \\
           --output-dir /content/drive/MyDrive/data/splits

    4. See example_pipeline.py for usage examples

    Documentation: https://github.com/yourusername/ukb-ttm-accelerometry
        """)
    else:
        print("""

    ⚠ WARNING: Basic tests failed
    Installation may be incomplete

    Please check error messages above
        """)


if __name__ == "__main__":
    main()
