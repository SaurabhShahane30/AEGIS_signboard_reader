"""
Signboard Detection & Voice Module
Real-time detection with OCR and text-to-speech
"""

import cv2
import numpy as np
from pathlib import Path
import easyocr
import pyttsx3
import threading
from collections import deque
import time
import torch
import pickle

# Monkey-patch torch.load to handle weights_only issue
_original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    """Patched torch.load that disables weights_only for old models"""
    try:
        # Try loading with weights_only=False to support old model formats
        kwargs['weights_only'] = False
        return _original_torch_load(f, *args, **kwargs)
    except Exception as e:
        print(f"Warning: torch.load failed with weights_only=False: {e}")
        raise

torch.load = patched_torch_load

# Now import YOLO after patching
from ultralytics import YOLO

class SignboardDetector:
    def __init__(self, model_path="signboard_project/best.pt"):
        """Initialize the detector with YOLO model and OCR engine"""
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        # Text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed
        self.tts_engine.setProperty('volume', 0.9)  # Volume
        
        # Queue to avoid duplicate voice reads
        self.read_texts = deque(maxlen=10)
        self.last_read_time = {}
        self.read_threshold = 3  # seconds between reading same text
        
        # Merge distance for grouping nearby text boxes into signboards
        self.merge_dist = 80  # pixels - boxes within this distance are same signboard
    
    def expand_box(self, box, dist):
        """Expand a box outward by `dist` pixels on all sides for overlap testing."""
        x1, y1, x2, y2 = box
        return (x1 - dist, y1 - dist, x2 + dist, y2 + dist)
    
    def boxes_overlap(self, a, b):
        """Return True if two boxes overlap (or touch)."""
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])
    
    def merge_boxes(self, boxes):
        """
        Merge boxes that are within merge_dist pixels of each other.
        This groups scattered text detections into single signboards.
        """
        if not boxes:
            return []
        
        # Work with list of lists for mutability
        groups = [[b] for b in boxes]
        merged = True
        
        while merged:
            merged = False
            new_groups = []
            used = [False] * len(groups)
            
            for i in range(len(groups)):
                if used[i]:
                    continue
                    
                current = groups[i]
                # Union box of current group
                ux1 = min(b[0] for b in current)
                uy1 = min(b[1] for b in current)
                ux2 = max(b[2] for b in current)
                uy2 = max(b[3] for b in current)
                expanded_i = self.expand_box((ux1, uy1, ux2, uy2), self.merge_dist)
                
                for j in range(i + 1, len(groups)):
                    if used[j]:
                        continue
                    jx1 = min(b[0] for b in groups[j])
                    jy1 = min(b[1] for b in groups[j])
                    jx2 = max(b[2] for b in groups[j])
                    jy2 = max(b[3] for b in groups[j])
                    expanded_j = self.expand_box((jx1, jy1, jx2, jy2), self.merge_dist)
                    
                    if self.boxes_overlap(expanded_i, expanded_j):
                        current = current + groups[j]
                        used[j] = True
                        merged = True
                        # Update union for continued merging
                        ux1 = min(b[0] for b in current)
                        uy1 = min(b[1] for b in current)
                        ux2 = max(b[2] for b in current)
                        uy2 = max(b[3] for b in current)
                        expanded_i = self.expand_box((ux1, uy1, ux2, uy2), self.merge_dist)
                
                new_groups.append(current)
                used[i] = True
            
            groups = new_groups
        
        # Convert each group to its union bounding rectangle
        merged_boxes = []
        for group in groups:
            x1 = min(b[0] for b in group)
            y1 = min(b[1] for b in group)
            x2 = max(b[2] for b in group)
            y2 = max(b[3] for b in group)
            merged_boxes.append((x1, y1, x2, y2))
        
        return merged_boxes
        
    def detect_signboards(self, frame):
        """
        Detect signboards in a frame by:
        1. Detecting all text boxes with YOLO
        2. Merging nearby boxes into single signboards
        3. Extracting text from each merged signboard
        
        Returns: list of merged detections
        """
        results = self.model(frame, conf=0.3, iou=0.45)
        
        # Collect all text box detections
        text_boxes = []
        box_confidences = {}
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                box_tuple = (x1, y1, x2, y2)
                text_boxes.append(box_tuple)
                box_confidences[box_tuple] = conf
        
        if not text_boxes:
            return []
        
        # Merge nearby boxes into signboards
        merged_boxes = self.merge_boxes(text_boxes)
        
        detections = []
        for merged_box in merged_boxes:
            x1, y1, x2, y2 = merged_box
            
            # Crop the merged signboard region
            crop = frame[y1:y2, x1:x2]
            
            # Extract text using OCR
            ocr_text = self.extract_text(crop)
            
            # Average confidence from boxes in this signboard
            avg_conf = np.mean([box_confidences.get(box, 0.5) for box in text_boxes 
                               if self.boxes_overlap(box, merged_box)])
            
            detections.append({
                'box': merged_box,
                'confidence': avg_conf,
                'text': ocr_text
            })
        
        return detections
    
    def extract_text(self, crop):
        """Extract text from signboard crop using EasyOCR"""
        try:
            if crop.size == 0:
                return ""
                
            results = self.reader.readtext(crop)
            text = " ".join([text[1] for text in results if text[2] > 0.3])
            return text.strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def read_text(self, text):
        """Speak out the detected text (non-blocking)"""
        if not text or len(text.strip()) == 0:
            return
            
        # Avoid reading the same text too frequently
        current_time = time.time()
        if text in self.last_read_time:
            if current_time - self.last_read_time[text] < self.read_threshold:
                return
        
        self.last_read_time[text] = current_time
        
        # Run in separate thread to not block video
        thread = threading.Thread(target=self._speak, args=(text,), daemon=True)
        thread.start()
    
    def _speak(self, text):
        """Internal method to speak text"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def draw_detections(self, frame, detections):
        """Draw detection boxes and text on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['box']
            conf = det['confidence']
            text = det['text']
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f"Signboard {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
            
            # Draw extracted text
            if text:
                # Wrap text if too long
                words = text.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) > 25:
                        lines.append(current_line)
                        current_line = word
                    else:
                        current_line += " " + word if current_line else word
                if current_line:
                    lines.append(current_line)
                
                y_offset = y2 + 25
                for line in lines[:3]:  # Show max 3 lines
                    cv2.putText(frame, line, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (255, 0, 0), 1)
                    y_offset += 20
        
        return frame
