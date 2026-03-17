import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# ================= CONFIG =================
MODEL_PATH = "gesture_model_final.h5"
IMG_SIZE = 128

# Adjusted thresholds for better detection
CONF_THRESHOLD = 0.45          # lowered for better sensitivity
CONF_GAP = 0.08                # reduced gap for more lenient detection
LOCK_FRAMES = 15               # increased for stability
SMOOTH_WINDOW = 7              # frames for temporal smoothing

CLASSES = ["none", "ok", "stop", "thumbs_up"]

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(MODEL_PATH)

# ================= STATE =================
gesture_lock = None
lock_timer = 0
locked_label = "NONE"
locked_conf = 0.0

# Add smoothing buffers
prediction_buffer = deque(maxlen=SMOOTH_WINDOW)
confidence_buffer = deque(maxlen=SMOOTH_WINDOW)
frame_count = 0

# ================= ENHANCED PREPROCESS =================
def preprocess_roi(roi, size=128):
    # Enhanced preprocessing for better gesture recognition
    
    # Convert to RGB for better color handling
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Apply slight Gaussian blur to reduce noise
    roi_rgb = cv2.GaussianBlur(roi_rgb, (3, 3), 0)
    
    # Make square crop from center
    h, w, _ = roi_rgb.shape
    if h > w:
        y = (h - w) // 2
        roi_rgb = roi_rgb[y:y+w, :]
    else:
        x = (w - h) // 2
        roi_rgb = roi_rgb[:, x:x+h]

    # Resize and normalize
    roi_rgb = cv2.resize(roi_rgb, (size, size))
    roi_rgb = roi_rgb.astype("float32") / 255.0
    
    # Add slight contrast enhancement
    roi_rgb = np.clip(roi_rgb * 1.1, 0, 1)
    
    return np.expand_dims(roi_rgb, axis=0)

# ================= SMOOTHING FUNCTIONS =================
def smooth_predictions(predictions, buffer):
    """Apply temporal smoothing to predictions"""
    buffer.append(predictions)
    if len(buffer) < 3:
        return predictions
    
    # Weighted average with more weight on recent predictions
    weights = np.exp(np.linspace(0, 1, len(buffer)))
    weights = weights / weights.sum()
    
    smoothed = np.zeros_like(predictions)
    for i, pred in enumerate(buffer):
        smoothed += pred * weights[i]
    
    return smoothed

def get_stable_gesture(smoothed_preds):
    """Get gesture with improved stability logic"""
    sorted_idx = np.argsort(smoothed_preds)[::-1]
    top1 = sorted_idx[0]
    top2 = sorted_idx[1]

    top1_conf = smoothed_preds[top1]
    top2_conf = smoothed_preds[top2]
    confidence_gap = top1_conf - top2_conf

    # Dynamic threshold based on gesture type
    dynamic_threshold = CONF_THRESHOLD
    if CLASSES[top1] in ["ok", "thumbs_up"]:
        dynamic_threshold *= 0.85  # More lenient for these gestures
    
    if top1_conf >= dynamic_threshold and confidence_gap >= CONF_GAP:
        return CLASSES[top1], top1_conf
    
    return "none", top1_conf

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
print("Continuous gesture recognition active | Press Q to quit")
print("Show gestures in the green box for detection")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_count += 1

    # ROI positioning
    box = 320  # Slightly larger box
    x1 = w//2 - box//2
    y1 = h//2 - box//2
    x2 = x1 + box
    y2 = y1 + box

    roi = frame[y1:y2, x1:x2]

    # ================= CONTINUOUS PREDICTION =================
    # Predict every 2 frames for better performance
    if frame_count % 2 == 0:
        img = preprocess_roi(roi, IMG_SIZE)
        preds = model.predict(img, verbose=0)[0]
        
        # Apply temporal smoothing
        smoothed_preds = smooth_predictions(preds, prediction_buffer)
        
        # Get stable gesture
        detected_gesture, confidence = get_stable_gesture(smoothed_preds)
        
        # ================= IMPROVED LOCKING LOGIC =================
        if detected_gesture != "none":
            if gesture_lock is None:
                # New gesture detected
                gesture_lock = detected_gesture
                locked_label = detected_gesture.upper().replace("_", " ")
                locked_conf = confidence
                lock_timer = LOCK_FRAMES
            elif gesture_lock == detected_gesture:
                # Same gesture, refresh timer and update confidence
                locked_conf = max(locked_conf * 0.7 + confidence * 0.3, confidence)
                lock_timer = LOCK_FRAMES
            else:
                # Different gesture detected, require higher confidence
                if confidence > locked_conf + 0.1:
                    gesture_lock = detected_gesture
                    locked_label = detected_gesture.upper().replace("_", " ")
                    locked_conf = confidence
                    lock_timer = LOCK_FRAMES

    # ================= LOCK TIMER =================
    if lock_timer > 0:
        lock_timer -= 1
        color = (0, 255, 0)  # Green when gesture is locked
    else:
        gesture_lock = None
        locked_label = "NONE"
        locked_conf = 0.0
        color = (0, 165, 255)  # Orange when ready for detection

    # ================= ENHANCED DISPLAY =================
    # Draw main ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Status text background
    text_bg_color = (0, 0, 0)
    cv2.rectangle(frame, (x1, y1 - 60), (x2, y1), text_bg_color, -1)
    
    # Main gesture label
    cv2.putText(
        frame,
        f"Gesture: {locked_label}",
        (x1 + 10, y1 - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )
    
    # Confidence
    cv2.putText(
        frame,
        f"Confidence: {locked_conf:.2f}",
        (x1 + 10, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    # Lock timer indicator
    if lock_timer > 0:
        lock_bar_width = int((lock_timer / LOCK_FRAMES) * (box - 20))
        cv2.rectangle(frame, (x1 + 10, y2 + 10), (x1 + 10 + lock_bar_width, y2 + 20), (0, 255, 0), -1)
    
    # Instructions
    cv2.putText(frame, "Show hand gestures in the box", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Enhanced Gesture Recognition", frame)

    # ================= CONTROLS =================
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
