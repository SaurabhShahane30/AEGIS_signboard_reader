"""
Signboard Detection Website
Live feed with real-time detection and voice
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import threading
from detector import SignboardDetector
import os
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global detector instance
detector = SignboardDetector(model_path="../signboard_project/best.pt")

# Video capture
camera = None
frame_lock = threading.Lock()
current_frame = None
detection_active = True
detected_texts = []

def init_camera():
    """Initialize camera"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
    return camera

def generate_frames():
    """Generate frames with detection"""
    global current_frame, detected_texts
    
    cap = init_camera()
    
    while detection_active:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Run detection
        detections = detector.detect_signboards(frame)
        
        # Update detected texts
        detected_texts.clear()
        for det in detections:
            if det['text']:
                detected_texts.append({
                    'text': det['text'],
                    'confidence': det['confidence']
                })
                # Read text out loud
                detector.read_text(det['text'])
        
        # Draw detections
        frame_with_boxes = detector.draw_detections(frame.copy(), detections)
        
        # Add stats
        cv2.putText(frame_with_boxes, f"Detections: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Store frame for local display
        with frame_lock:
            current_frame = frame_with_boxes.copy()
        
        # Encode for streaming
        ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    """Get current detections"""
    return jsonify({
        'detections': detected_texts,
        'count': len(detected_texts)
    })

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Start detection"""
    global detection_active
    detection_active = True
    return jsonify({'status': 'started'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop detection"""
    global detection_active
    detection_active = False
    return jsonify({'status': 'stopped'})

@app.route('/speak', methods=['POST'])
def speak_text():
    """Manually trigger voice reading"""
    data = request.json
    text = data.get('text', '')
    if text:
        detector.read_text(text)
        return jsonify({'status': 'speaking'})
    return jsonify({'status': 'error', 'message': 'No text provided'})

@app.route('/camera_status')
def camera_status():
    """Check camera status"""
    cap = init_camera()
    if cap.isOpened():
        return jsonify({'status': 'connected', 'message': 'Camera is ready'})
    else:
        return jsonify({'status': 'error', 'message': 'Camera not available'})

@app.errorhandler(Exception)
def handle_error(error):
    """Error handler"""
    return jsonify({'error': str(error)}), 500

if __name__ == '__main__':
    print("🎬 Starting Signboard Detection Server...")
    print("📍 Open http://localhost:5000 in your browser")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
