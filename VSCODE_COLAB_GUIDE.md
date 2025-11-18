# VS Code + Colab GPU Setup Guide

Complete guide to develop in VS Code while using Colab's free GPU.

---

## 🎯 What This Enables

✅ Code in VS Code (your favorite IDE)
✅ Execute on Colab GPU (free T4 GPU)
✅ Use VS Code debugger with breakpoints
✅ Access to 12GB RAM + 16GB VRAM
✅ Auto-complete and IntelliSense
✅ Git integration in VS Code

---

## 📋 Prerequisites

1. **VS Code** installed locally
2. **Google Account** (for Colab)
3. **VS Code Extensions**:
   - Remote - SSH (ms-vscode-remote.remote-ssh)
   - Python (ms-python.python)
   - Jupyter (ms-toolsai.jupyter)

Install extensions:
```bash
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
```

---

## 🚀 Method 1: SSH Tunnel via Cloudflared (Recommended)

### Step 1: Open Colab Notebook

1. Go to: https://colab.research.google.com/
2. Upload `notebooks/Colab_VSCode_Setup.ipynb`
3. Change runtime to GPU:
   - Runtime → Change runtime type → GPU → Save

### Step 2: Run Setup Cells

Execute each cell in order:

1. **Install Cloudflared** - Installs tunneling software
2. **Setup SSH** - Creates SSH server with password
3. **Clone Repository** - Downloads your code
4. **Mount Drive** - For saving checkpoints
5. **Start Tunnel** - Creates public URL

### Step 3: Copy Connection URL

After running Step 5, you'll see:
```
✓ TUNNEL READY!
Connection URL: https://abc123.trycloudflare.com
```

**Copy this URL!**

### Step 4: Connect from VS Code

1. Open VS Code
2. Press **F1** (or Cmd/Ctrl+Shift+P)
3. Type: `Remote-SSH: Connect to Host`
4. Enter: `root@abc123.trycloudflare.com` (use your URL)
5. Select: `Linux` as platform
6. Enter the password you set in Step 2
7. Wait for connection...
8. Open folder: `/content/AccelomtryFoundationModel`

### Step 5: You're Connected! 🎉

Now you can:
- Edit code in VS Code
- Run scripts on Colab GPU
- Use debugger with breakpoints
- Access terminal with GPU

---

## 🚀 Method 2: ngrok (Alternative)

If Cloudflared doesn't work:

### Step 1: Get ngrok Token

1. Go to: https://dashboard.ngrok.com/signup
2. Sign up (free)
3. Get your authtoken: https://dashboard.ngrok.com/get-started/your-authtoken

### Step 2: In Colab

```python
# Install
!pip install pyngrok

# Setup
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN_HERE")

# Start SSH
!sudo service ssh start

# Create tunnel
ssh_tunnel = ngrok.connect(22, "tcp")
print(ssh_tunnel.public_url)
```

### Step 3: Connect from VS Code

The URL will look like: `tcp://0.tcp.ngrok.io:12345`

In VS Code Remote-SSH:
```
root@0.tcp.ngrok.io:12345
```

---

## 💻 Using VS Code with Colab GPU

### Running Training Scripts

#### Option 1: VS Code Terminal
```bash
# In VS Code terminal (connected to Colab)
cd /content/AccelomtryFoundationModel
python scripts/train_vscode.py --data data/synthetic_data.h5 --quick
```

#### Option 2: VS Code Debugger
1. Open `scripts/train_vscode.py`
2. Set breakpoints (click left of line numbers)
3. Press **F5**
4. Select **"Train (Quick)"**
5. Debug interactively!

#### Option 3: Run Current File
1. Open any Python file
2. Right-click → "Run Python File in Terminal"
3. Or press **Ctrl+F5**

### Interactive Development

```python
# In VS Code terminal with IPython
ipython

# Now you have GPU!
import torch
print(torch.cuda.is_available())  # True!
print(torch.cuda.get_device_name(0))  # Tesla T4

# Load your model
from src.models.ttm_classifier import TTMAccelerometryClassifier
model = TTMAccelerometryClassifier().cuda()

# Test
x = torch.randn(8, 820, 3).cuda()
output = model(x)
print(output['logits'].shape)  # torch.Size([8, 4])
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test with debugger
pytest tests/test_model.py::test_forward_pass --pdb

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Using Jupyter Notebooks

1. Open `notebooks/TTM_Accelerometry_Training.ipynb` in VS Code
2. Select kernel: Python 3 (remote)
3. Run cells - they execute on Colab GPU!

---

## 🎨 VS Code Features on Colab

### 1. Debugging

Set breakpoints and inspect variables:
```python
# In train_vscode.py
def train_epoch():
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)  # ← Set breakpoint here
        output = model(x)
        # Inspect x, output in debugger!
```

Press F5 → execution stops at breakpoint

### 2. Git Integration

```bash
# In VS Code terminal
git status
git add src/models/ttm_classifier.py
git commit -m "Improve model architecture"
git push
```

### 3. IntelliSense & Auto-complete

Type `model.` and see all methods!
```python
model.  # Auto-complete shows: forward(), freeze_encoder(), etc.
```

### 4. Code Navigation

- **Go to Definition**: Ctrl+Click on function name
- **Find All References**: Right-click → Find All References
- **Rename Symbol**: F2 on variable name

### 5. Terminal Multi-plex

Open multiple terminals:
- Terminal 1: Training script
- Terminal 2: Monitor GPU (`watch -n 1 nvidia-smi`)
- Terminal 3: Interactive Python

---

## 📊 Monitoring GPU Usage

### In VS Code Terminal

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv -l 1
```

### In Python

```python
import torch

# Memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# During training
def train_step(batch):
    before = torch.cuda.memory_allocated()
    output = model(batch)
    after = torch.cuda.memory_allocated()
    print(f"Memory used: {(after - before) / 1e6:.2f} MB")
```

---

## 💾 Saving Work

### Checkpoints to Google Drive

Training automatically saves to Drive:
```python
# In train_vscode.py
# Checkpoints saved to: /content/drive/MyDrive/ttm_accelerometry/checkpoints/
```

### Download Results Locally

In VS Code Explorer:
1. Right-click on `checkpoints/ttm_final.pt`
2. Download...
3. Save to your local machine

Or via terminal:
```bash
# In VS Code terminal (connected to Colab)
cp /content/drive/MyDrive/ttm_accelerometry/checkpoints/ttm_final.pt \
   /content/AccelomtryFoundationModel/checkpoints/
```

Then download from VS Code file explorer.

---

## 🔧 Troubleshooting

### Connection Refused

**Problem:** Can't connect to Colab

**Solutions:**
1. Make sure cloudflared cell is still running
2. Check URL is correct (no typos)
3. Try Method 2 (ngrok) instead

### Permission Denied

**Problem:** Can't authenticate

**Solutions:**
1. Double-check password
2. Reset password in Colab:
   ```python
   import subprocess
   password = "your_new_password"
   subprocess.run(f"echo 'root:{password}' | sudo chpasswd", shell=True)
   ```

### Colab Disconnects

**Problem:** Colab session times out

**Solutions:**
1. Run keep-alive cell in Colab notebook
2. Get Colab Pro ($10/month) for longer sessions
3. Save checkpoints frequently to Drive

### Slow Connection

**Problem:** Editing is laggy

**Solutions:**
1. Use Cloudflared instead of ngrok (faster)
2. Close unnecessary tabs in Colab
3. Restart tunnel

### GPU Not Available

**Problem:** `torch.cuda.is_available()` returns False

**Solutions:**
1. Check Colab runtime type: Runtime → Change runtime type → GPU
2. Restart Colab runtime
3. Test in Colab notebook first before VS Code

---

## 📝 Best Practices

### 1. Keep Colab Session Alive

Run this in Colab:
```python
import time
from IPython.display import clear_output

while True:
    time.sleep(60)
    clear_output(wait=True)
    print(f"Session alive: {time.strftime('%H:%M:%S')}")
```

### 2. Save Frequently

```python
# In training script
if step % 1000 == 0:
    torch.save(model.state_dict(), f'checkpoint_{step}.pt')
```

### 3. Use Relative Paths

```python
# Good
data_path = './data/dataset.h5'

# Bad (breaks when switching between local/Colab)
data_path = '/content/AccelomtryFoundationModel/data/dataset.h5'
```

### 4. Monitor Memory

```python
# Add to training loop
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
```

### 5. Test Locally First

```bash
# Test on small dataset locally
python scripts/train_vscode.py --data data/small_test.h5 --quick

# Then run full training on Colab GPU
```

---

## 🎯 Example Workflow

### 1. Morning: Start Colab Session

```
1. Open Colab_VSCode_Setup.ipynb in Colab
2. Run all cells
3. Copy connection URL
```

### 2. Connect VS Code

```
1. F1 → Remote-SSH: Connect to Host
2. Enter URL
3. Open /content/AccelomtryFoundationModel
```

### 3. Develop Code

```python
# Edit in VS Code
# src/models/ttm_classifier.py

def forward(self, x):
    # Add new feature...
    embeddings = self.encoder(x)
    logits = self.classifier(embeddings)
    return {'logits': logits}
```

### 4. Test Changes

```bash
# In VS Code terminal
pytest tests/test_model.py -v
```

### 5. Train Model

```bash
# Quick test
python scripts/train_vscode.py --quick

# Full training
python scripts/train_vscode.py --data data/ukb_dataset.h5
```

### 6. Debug if Needed

```
1. Set breakpoint in train_vscode.py
2. F5 → Train (Quick)
3. Inspect variables when breakpoint hits
```

### 7. Save Results

```
1. Checkpoints auto-saved to Drive
2. Download from VS Code explorer
3. Commit code changes to git
```

---

## 🚀 Advanced: Custom Launch Configurations

Add to `.vscode/launch.json`:

```json
{
    "name": "Train on Colab GPU",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/scripts/train_vscode.py",
    "args": [
        "--data", "/content/drive/MyDrive/ttm_accelerometry/data/ukb_dataset.h5",
        "--checkpoint-dir", "/content/drive/MyDrive/ttm_accelerometry/checkpoints",
        "--mount-drive",
        "--batch-size", "64"
    ],
    "console": "integratedTerminal",
    "env": {
        "CUDA_VISIBLE_DEVICES": "0"
    }
}
```

Now press F5 and select "Train on Colab GPU"!

---

## 📚 Resources

- **VS Code Remote-SSH**: https://code.visualstudio.com/docs/remote/ssh
- **Cloudflared**: https://github.com/cloudflare/cloudflared
- **ngrok**: https://ngrok.com/docs
- **Colab**: https://colab.research.google.com/

---

## ✅ Quick Reference

| Task | Command |
|------|---------|
| Connect to Colab | F1 → Remote-SSH: Connect |
| Run script | `python scripts/train_vscode.py` |
| Debug | F5 |
| Run tests | `pytest tests/ -v` |
| Check GPU | `nvidia-smi` |
| IPython | `ipython` in terminal |
| Download file | Right-click → Download |
| Upload file | Drag & drop to explorer |

---

**You now have the power of VS Code + Colab GPU! Happy coding! 🚀**
