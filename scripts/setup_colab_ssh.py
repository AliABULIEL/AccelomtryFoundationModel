#!/usr/bin/env python3
"""
Setup SSH tunnel to Google Colab for VS Code remote development
Run this in a Colab notebook cell to enable VS Code connection
"""

import subprocess
import os
import getpass


def setup_colab_ssh():
    """
    Setup SSH access to Colab runtime.
    This allows VS Code to connect remotely.
    """
    print("=" * 70)
    print("VS Code + Colab GPU Setup")
    print("=" * 70)

    # Check if running in Colab
    try:
        import google.colab
        in_colab = True
        print("✓ Running in Google Colab")
    except ImportError:
        in_colab = False
        print("✗ Not running in Colab - use this script in a Colab notebook")
        return

    print("\nInstalling requirements...")

    # Install cloudflared for tunnel
    commands = [
        "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb",
        "dpkg -i cloudflared-linux-amd64.deb",
        "rm cloudflared-linux-amd64.deb"
    ]

    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)

    print("✓ Cloudflared installed")

    # Setup SSH
    print("\nSetting up SSH...")

    # Get password
    password = getpass.getpass("Enter a password for SSH (remember this!): ")

    # Set password
    subprocess.run(f"echo 'root:{password}' | chpasswd", shell=True, check=True)

    # Configure SSH
    os.makedirs("/var/run/sshd", exist_ok=True)

    # Start SSH service
    subprocess.run("service ssh start", shell=True, check=True)
    print("✓ SSH service started")

    # Start cloudflared tunnel
    print("\nStarting Cloudflared tunnel...")
    print("This will create a public URL for your Colab instance")
    print("\n" + "=" * 70)

    # Run cloudflared (this will print the URL)
    subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "ssh://localhost:22"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    import time
    time.sleep(5)

    print("\n" + "=" * 70)
    print("Setup complete!")
    print("\nTo connect from VS Code:")
    print("1. Copy the URL shown above (looks like: https://xxx.trycloudflare.com)")
    print("2. In VS Code, install 'Remote - SSH' extension")
    print("3. Press F1 and select 'Remote-SSH: Connect to Host'")
    print("4. Enter: ssh root@<your-cloudflare-url>")
    print("5. Enter the password you set above")
    print("6. Open folder: /content/AccelomtryFoundationModel")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    setup_colab_ssh()
