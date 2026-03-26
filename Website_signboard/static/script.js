// Global state
let isDetecting = false;
let detectionInterval;
let videoRetryTimeout;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎯 Signboard Detection Website Loaded');
    initVideoFeed();
    checkCamera();
    startDetectionUpdates();
});

// Initialize/reconnect video feed stream
function initVideoFeed() {
    const videoFeed = document.getElementById('video-feed');
    if (!videoFeed) return;

    const streamUrl = `/video_feed?t=${Date.now()}`;
    videoFeed.src = streamUrl;

    videoFeed.onload = () => {
        if (videoRetryTimeout) {
            clearTimeout(videoRetryTimeout);
            videoRetryTimeout = null;
        }
    };

    videoFeed.onerror = () => {
        console.error('Video feed disconnected, retrying...');
        if (videoRetryTimeout) clearTimeout(videoRetryTimeout);
        videoRetryTimeout = setTimeout(initVideoFeed, 1000);
    };
}

// Start detection updates
function startDetectionUpdates() {
    detectionInterval = setInterval(updateDetections, 1000);
}

// Update detections from server
function updateDetections() {
    if (!isDetecting) return;
    
    fetch('/detections')
        .then(response => response.json())
        .then(data => {
            updateDetectionUI(data.detections, data.count);
        })
        .catch(error => {
            console.error('Error fetching detections:', error);
        });
}

// Update detection UI
function updateDetectionUI(detections, count) {
    // Update count
    document.getElementById('detection-count').textContent = count;
    
    // Update detections list
    const list = document.getElementById('detections-list');
    
    if (count === 0) {
        list.innerHTML = '<p class="empty-message">No detections yet...</p>';
        return;
    }
    
    let html = '';
    detections.forEach((det, index) => {
        html += `
            <div class="detection-item">
                <strong>${index + 1}. ${det.text || 'No text'}</strong>
                <div class="detection-confidence">
                    Confidence: ${(det.confidence * 100).toFixed(1)}%
                </div>
                <button onclick="speakText('${escapeHtml(det.text)}')" 
                        class="btn btn-info" style="margin-top: 5px; width: 100%; padding: 5px;">
                    🔊 Speak
                </button>
            </div>
        `;
    });
    
    list.innerHTML = html;
}

// Start detection
function startDetection() {
    fetch('/start_detection', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        isDetecting = true;
        document.getElementById('start-btn').disabled = true;
        document.getElementById('stop-btn').disabled = false;
        document.getElementById('detection-status').textContent = 'Running';
        document.getElementById('detection-status').style.color = '#10b981';
        console.log('✅ Detection started');
    })
    .catch(error => {
        console.error('Error starting detection:', error);
        alert('Failed to start detection');
    });
}

// Stop detection
function stopDetection() {
    fetch('/stop_detection', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        isDetecting = false;
        document.getElementById('start-btn').disabled = false;
        document.getElementById('stop-btn').disabled = true;
        document.getElementById('detection-status').textContent = 'Stopped';
        document.getElementById('detection-status').style.color = '#ef4444';
        console.log('⏹️ Detection stopped');
    })
    .catch(error => {
        console.error('Error stopping detection:', error);
        alert('Failed to stop detection');
    });
}

// Check camera status
function checkCamera() {
    fetch('/camera_status')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'connected') {
                console.log('✅ Camera connected');
                document.getElementById('status-text').textContent = 'Live';
                document.getElementById('status-text').style.color = '#10b981';
                initVideoFeed();
            } else {
                console.error('❌ Camera error:', data.message);
                alert('Camera Error: ' + data.message);
                document.getElementById('status-text').textContent = 'Error';
                document.getElementById('status-text').style.color = '#ef4444';
            }
        })
        .catch(error => {
            console.error('Error checking camera:', error);
            alert('Failed to check camera: ' + error);
        });
}

// Manually speak text
function speakManually() {
    const text = document.getElementById('manual-text').value.trim();
    
    if (!text) {
        alert('Please enter some text');
        return;
    }
    
    speakText(text);
    document.getElementById('manual-text').value = '';
}

// Send text to server to speak
function speakText(text) {
    if (!text || text.trim().length === 0) {
        alert('No text to speak');
        return;
    }
    
    fetch('/speak', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        console.log('🔊 Speaking:', text);
    })
    .catch(error => {
        console.error('Error speaking:', error);
    });
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add keyboard shortcut for speaking manually text
document.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        const manualTextInput = document.getElementById('manual-text');
        if (document.activeElement === manualTextInput) {
            speakManually();
        }
    }
});

// Auto-start detection on page load (optional)
window.addEventListener('load', function() {
    // Uncomment the next line to auto-start detection
    // setTimeout(startDetection, 1000);
});

console.log('✅ Script loaded successfully');
