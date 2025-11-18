# VS Code + Colab GPU - Quick Start

**5 minutes to connect VS Code to Colab GPU**

---

## 🚀 Quick Setup

### 1. In Google Colab

```
1. Open: https://colab.research.google.com/
2. Upload: notebooks/Colab_VSCode_Setup.ipynb
3. Runtime → Change runtime type → GPU → Save
4. Run ALL cells (Shift+Enter on each)
5. Copy the URL from Step 5 (looks like: https://abc123.trycloudflare.com)
```

### 2. In VS Code (Local)

```
1. Install extension: "Remote - SSH"
2. Press F1
3. Type: "Remote-SSH: Connect to Host"
4. Enter: root@abc123.trycloudflare.com (your URL)
5. Enter password (from Colab Step 2)
6. Open folder: /content/AccelomtryFoundationModel
```

### 3. You're Connected! 🎉

Now in VS Code terminal (connected to Colab):

```bash
# Check GPU
nvidia-smi

# Quick training test
python scripts/train_vscode.py --quick

# Or press F5 and select "Train (Quick)"
```

---

## 💡 Common Commands

| What | Command |
|------|---------|
| **Train (quick test)** | `python scripts/train_vscode.py --quick` |
| **Train (full)** | `python scripts/train_vscode.py --data data/ukb_dataset.h5` |
| **Run tests** | `pytest tests/ -v` |
| **Check GPU** | `nvidia-smi` |
| **Debug** | Press **F5** in VS Code |
| **Interactive Python** | `ipython` |

---

## 🐛 Troubleshooting

**Can't connect?**
- Check cloudflared cell is still running in Colab
- Copy URL carefully (no extra spaces)
- Try closing and reconnecting

**Colab disconnects?**
- Run the "Keep-Alive" cell in Colab notebook
- Save checkpoints frequently

**No GPU in VS Code?**
- In Colab: Runtime → Change runtime type → GPU
- Restart Colab runtime

---

## 📁 File Locations

When connected to Colab:

```
/content/AccelomtryFoundationModel/     ← Your code
/content/drive/MyDrive/ttm_accelerometry/ ← Google Drive (checkpoints)
```

---

## 🎯 Typical Workflow

```bash
# 1. Edit code in VS Code
# (edit any file in src/)

# 2. Test changes
pytest tests/test_model.py -v

# 3. Train with changes
python scripts/train_vscode.py --quick

# 4. Debug if needed (F5)

# 5. Checkpoints auto-save to Drive
# 6. Download results from VS Code file explorer
```

---

## 📚 Full Guide

See **VSCODE_COLAB_GUIDE.md** for:
- Alternative connection methods (ngrok)
- Advanced debugging
- Custom launch configurations
- Best practices
- Troubleshooting

---

**That's it! You're coding in VS Code with Colab's GPU! 🚀**
