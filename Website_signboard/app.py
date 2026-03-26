"""
Signboard Detection Website
Live feed with real-time detection and voice
"""
import time
from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import threading
from detector import SignboardDetector
import os
from pathlib import Path
# import google.generativeai as genai
# import threading
# from dotenv import load_dotenv
# load_dotenv()

# LAST_GEMINI_CALL = {}
# GEMINI_COOLDOWN = 50  # seconds

LAST_SPOKEN = {}
SPEECH_COOLDOWN = 5  # seconds
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / 'signboard_project' / 'best.pt'

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# model = genai.GenerativeModel("gemini-1.5-flash")
# Create uploads folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global detector instance
detector = SignboardDetector(model_path=str(MODEL_PATH))

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

def is_potentially_important(text):
    keywords = [
    "exit", "emergency", "fire",
    "danger", "caution", "wet",
    "slippery", "construction",
    "road work", "stairs", "hazard"
]
    return any(word in text.lower() for word in keywords)

def generate_speech_from_text(text):
    text = text.lower()

    # Wet floor / slippery
    if any(word in text for word in ["wet", "slippery", "floor"]):
        return "Warning. The floor ahead appears to be wet. Please walk carefully."

    # Construction
    if any(word in text for word in ["construction", "road work", "work ahead"]):
        return "Caution. Construction work is ahead. Please proceed carefully."

    # Exit
    if "exit" in text:
        return "An emergency exit is nearby."

    # Stairs
    if "stairs" in text or "steps" in text:
        return "There are stairs ahead. Please be careful."

    # General caution
    if "danger" in text or "caution" in text:
        return "Caution. There may be a hazard ahead. Please stay alert."

    return None

# def gemini_refine(text):
#     print("🔥 Sending to Gemini:", text)
#     try:
#         prompt = f"""
# You are an assistive AI for blind users.

# A signboard says: "{text}"

# 1. Decide if it is important for safety.
# 2. If important → return a short warning (1 sentence)
# 3. If not important → return "IGNORE"

# Examples:
# "Caution Wet Floor" → Warning. The floor ahead may be slippery.
# "Exit" → Emergency exit ahead.
# "Pizza Hut" → IGNORE

# Answer:
# """

#         response = model.generate_content(prompt)

#         output = response.text.strip()

#         if "IGNORE" in output.upper():
#             return None

#         return output

#     except Exception as e:
#         print("Gemini error:", e)
#         return text  # fallback
    
# def handle_gemini_and_speak(text):
#     global PROCESSING

#     try:
#         alert = gemini_refine(text)

#         if alert:
#             detector.read_text(alert)

#     finally:
#         # Remove from processing safely
#         with LOCK:
#             PROCESSING.discard(text.lower())
            
def generate_frames():
    """Generate frames with detection"""
    global current_frame, detected_texts
    
    cap = init_camera()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            continue

        frame_with_boxes = frame.copy()

        if detection_active:
            # Run detection only when explicitly enabled.
            detections = detector.detect_signboards(frame)
            
            # Update detected texts
            detected_texts.clear()

            for det in detections:
                if not det['text']:
                    continue

                text = det['text']
                text_lower = text.lower()
                current_time = time.time()

                print("TEXT DETECTED:", text)

                detected_texts.append({
                    'text': text,
                    'confidence': det['confidence']
                })

                # Step 1 — filter important signs
                if not is_potentially_important(text):
                    continue

                # Step 2 — cooldown (prevent spam)
                if text_lower in LAST_SPOKEN:
                    if current_time - LAST_SPOKEN[text_lower] < SPEECH_COOLDOWN:
                        continue

                LAST_SPOKEN[text_lower] = current_time

                # Step 3 — generate speech
                alert = generate_speech_from_text(text)

                if alert:
                    detector.read_text(alert)
            # Draw detections
            frame_with_boxes = detector.draw_detections(frame_with_boxes, detections)

            # Add stats
            cv2.putText(frame_with_boxes, f"Detections: {len(detections)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Keep stream alive even while detection is stopped.
            detected_texts.clear()
            cv2.putText(frame_with_boxes, "Detection Paused", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Store frame for local display
        with frame_lock:
            current_frame = frame_with_boxes.copy()
        
        # Encode for streaming
        ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
        if not ret:
            continue
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
    port = int(os.getenv('PORT', '5001'))
    print("🎬 Starting Signboard Detection Server...")
    print(f"📍 Open http://localhost:{port} in your browser")
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
