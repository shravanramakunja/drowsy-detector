# Smart Driver Hub - Web Frontend

A modern, responsive web interface for the drowsiness detection system.

## Features

- 🎨 Modern, professional UI design
- 📱 Fully responsive (works on desktop, tablet, mobile)
- 🎥 Live camera feed display
- ⚙️ Adjustable settings (threshold, volume)
- 📊 Real-time status monitoring
- 🔔 Visual and audio alerts
- 📖 Comprehensive feature documentation

## Files

- `index.html` - Main HTML structure
- `styles.css` - All styling and responsive design
- `script.js` - Frontend JavaScript logic
- `README.md` - This file

## How to Use

### Option 1: Simple Local Usage

1. Open `index.html` directly in a web browser
2. Click "Start Detection" button
3. Allow camera access when prompted
4. The system will monitor your eyes

### Option 2: With Python Backend (Full Functionality)

To integrate with the actual Python drowsiness detection:

1. **Set up Flask Backend** (create `app.py`):

```python
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from drowsiness import DrowsinessDetector

app = Flask(__name__)
detector = DrowsinessDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detector.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return jsonify({
        'eye_state': detector.eye_state,
        'timer': detector.eyes_closed_time,
        'alert': detector.alarm_on
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

2. **Install Flask**:
```bash
pip install flask flask-cors
```

3. **Run the server**:
```bash
python app.py
```

4. **Access at**: http://localhost:5000

## Customization

### Change Alert Threshold

In `script.js`, modify:
```javascript
let alertThreshold = 5; // Change to desired seconds
```

### Change Colors

In `styles.css`, modify the CSS variables:
```css
:root {
    --primary-color: #2563eb;  /* Change primary color */
    --danger-color: #ef4444;   /* Change alert color */
    /* etc. */
}
```

### Add New Features

1. Add UI elements in `index.html`
2. Add styling in `styles.css`
3. Add logic in `script.js`

## Browser Compatibility

- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ⚠️ IE11 (limited support)

## Security Notes

- Camera access requires HTTPS in production
- For local testing, `http://localhost` is allowed
- Always request user permission before camera access

## Future Enhancements

- [ ] WebSocket integration for real-time Python backend communication
- [ ] Historical data charts
- [ ] Multiple user profiles
- [ ] Export detection logs
- [ ] Mobile app version
- [ ] Dark mode toggle

## Credits

Built for Smart Driver Hub Drowsiness Detection System
