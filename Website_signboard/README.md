# 🎯 Live Signboard Detection Website

Real-time signboard detection with OCR and voice reading powered by YOLOv8.

## Features

✨ **Live Video Stream** - Real-time camera feed with detection overlay
🚀 **YOLOv8 Detection** - Fast and accurate signboard detection
📖 **OCR Text Extraction** - Automatic text recognition from detected signboards
🔊 **Voice Reading** - Text-to-speech that reads detected signboards aloud
📊 **Live Statistics** - Real-time detection count and status
🎤 **Manual Text-to-Speech** - Manually enter and read any text

## Requirements

- Python 3.8+
- Webcam/Camera
- NVIDIA GPU (recommended for faster processing)

## Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Model is Present**
   Make sure your trained model is at: `../signboard_project/best.pt`

## Running the Application

1. **Start the Server**
   ```bash
   python app.py
   ```

2. **Open in Browser**
   Navigate to: `http://localhost:5000`

3. **Start Detection**
   - Click "Start Detection" button
   - The live feed will show detected signboards with boxes
   - Detected text will be read aloud automatically

## How It Works

### Detection Pipeline
1. **Capture Frame** - Get frame from webcam
2. **YOLOv8 Detection** - Detect signboards in the frame
3. **Crop Signboard** - Extract the detected signboard region
4. **OCR** - Extract text using EasyOCR
5. **Voice** - Read the detected text using pyttsx3
6. **Display** - Show detections on the web interface

### Architecture
- **Backend**: Flask web server
- **Detection**: YOLOv8 (ultralytics)
- **OCR**: EasyOCR
- **TTS**: pyttsx3
- **Frontend**: HTML5, CSS3, JavaScript with MJPEG streaming

## Configuration

### Adjust Detection Sensitivity
Edit `app.py`, in the `detector = SignboardDetector(...)` line:
```python
# Lower conf = more detections, more false positives
# Higher conf = fewer detections, fewer false positives
results = self.model(frame, conf=0.3, iou=0.45)
```

### Adjust Voice Speed
Edit `detector.py`:
```python
self.tts_engine.setProperty('rate', 150)  # Lower = slower, Higher = faster
```

### Adjust Voice Reading Frequency
Edit `detector.py`:
```python
self.read_threshold = 3  # Seconds between reading same text
```

## Performance Tips

🚀 **GPU Acceleration**
- If you have NVIDIA GPU, place the model on GPU for faster detection
- Edit `detector.py`: `self.model = YOLO(model_path).to('cuda')`

⚡ **Reduce Frame Resolution**
- Edit `app.py` to reduce camera resolution
- Smaller = faster, but less accurate

📦 **Use Smaller Model**
- Replace `best.pt` with a smaller model variant
- Available: `yolov8n.pt` (nano), `yolov8s.pt` (small)

## Troubleshooting

### Camera Not Found
- Check if camera is connected
- Try: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`
- May need to use different camera index (0, 1, 2...)

### Slow Performance
- Reduce detection frequency
- Use GPU if available
- Consider using a smaller YOLO model

### No Audio
- Check system volume
- Check if speakers are working
- Try manually clicking "Speak" button

### OCR Not Working
- Internet connection may be needed for EasyOCR on first run
- Check text visibility in detected regions

## API Endpoints

- `GET /` - Main web interface
- `GET /video_feed` - MJPEG video stream
- `GET /detections` - Current detections (JSON)
- `GET /camera_status` - Camera connection status
- `POST /start_detection` - Start detection
- `POST /stop_detection` - Stop detection
- `POST /speak` - Trigger text-to-speech

## Project Structure

```
Website_signboard/
├── app.py                 # Flask backend
├── detector.py           # Detection & voice module
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── templates/
│   └── index.html       # Web interface
└── static/
    ├── style.css        # Styling
    └── script.js        # Frontend logic
```

## Future Enhancements

- 🔴 Recording detected videos
- 🔴 Multiple language support
- 🔴 Custom YOLO model training UI
- 🔴 Statistics dashboard
- 🔴 Video file upload and detection

## License

MIT License - Feel free to use and modify!

## Support

For issues or suggestions, check the configuration or adjust detection parameters.

---

Made with ❤️ for real-time signboard detection
