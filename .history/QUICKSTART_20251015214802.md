# Quick Start Guide - Drowsiness Detection System

## Choose Your Version

### 🖥️ Standalone Desktop Application (Simplest)
Best for: Quick testing, no browser needed

**Run this command:**
```bash
python drowsiness.py
```

**That's it!** A window will open with your camera feed.
- Press `q` or `t` to quit
- Close eyes for 5+ seconds to test alarm

---

### 🌐 Web Application (Better UI)
Best for: Better interface, adjustable settings

**Step 1 - Start Server:**
```bash
python app.py
```

**Step 2 - Open Browser:**
Go to: http://localhost:5000

**Step 3 - Click "Start Detection"**

**Step 4 - Close eyes for 5 seconds to test**

---

## First Time Setup

Only need to do this once:

```bash
# Install dependencies
pip install -r requirements.txt
```

---

## Common Issues

**❌ Camera not working?**
- Close other apps using camera (Zoom, Skype, etc.)
- Check camera permissions in Windows Settings

**❌ Module not found?**
- Run: `pip install -r requirements.txt`

**❌ No sound?**
- Check system volume is not muted
- Check Python is not muted in Volume Mixer

---

## Need Help?

Read the full README.md for detailed instructions and troubleshooting.
