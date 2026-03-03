import cv2
import numpy as np
import pygame
import mediapipe as mp
import os # Added for path handling
from tensorflow.keras.models import load_model
from vision_core import FocusVision

# --- DYNAMIC PATH RESOLUTION ---
# This automatically finds the folder where THIS script is sitting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. INITIALIZE AUDIO
pygame.mixer.init()
# We use os.path.join to ensure it works on Mac perfectly
ALERT_FILENAME = "kyu-nhi-ho-rhi-padhai-audio-meme-download.mp3" 
ALERT_PATH = os.path.join(BASE_DIR, ALERT_FILENAME)

try:
    alert_sound = pygame.mixer.Sound(ALERT_PATH)
    print(f"[SYS] AUDIO LOADED: {ALERT_PATH}")
except Exception as e:
    print(f"[ERROR] Could not load audio at {ALERT_PATH}: {e}")

# 2. LOAD AI MODEL & ENGINE
MODEL_PATH = os.path.join(BASE_DIR, 'focus_shield_v2.h5')
model = load_model(MODEL_PATH)
engine = FocusVision()
actions = ['FOCUSED','PAPER_FOCUS' ,'DISTRACTED']

# 3. TRACKING VARIABLES
sequence = []
distraction_timer = 0
is_playing = False
current_status = "CALIBRATING..."

cap = cv2.VideoCapture(0)

print("[SYS] FOCUS SHIELD ONLINE. SYSTEM IS WATCHING.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Convert frame for MediaPipe Tasks API
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # 4. VISION DETECTION
    results = engine.detector.detect(mp_image)
    keypoints = engine.extract_keypoints(results)
    
    # Maintain the 30-frame temporal window
    sequence.append(keypoints)
    sequence = sequence[-30:] 
    
    if len(sequence) == 30:
        # Predict state (0 = FOCUSED, 1 = DISTRACTED)
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        prediction_idx = np.argmax(res)
        confidence = res[prediction_idx]
        
        # High confidence threshold to avoid false alarms
        if confidence > 0.85:
            current_status = actions[prediction_idx]
            
            if current_status == 'DISTRACTED':
                distraction_timer += 1
            else:
                distraction_timer = 0 # Reset immediately if you look back
                is_playing = False # Reset audio lock
    
    # 5. TRIGGER LOGIC: 150 frames = ~5 seconds of distraction
    if distraction_timer > 150 and not is_playing:
        print("[ALERT] TRIGGERING CUSTOM MP3...")
        alert_sound.play()
        is_playing = True # Locks audio so it doesn't spam every frame

    # 6. RENDER HUD
    image = engine.render_HUD(frame, results)
    
    # UI Styling
    bg_color = (0, 0, 150) if current_status == 'DISTRACTED' else (0, 150, 0)
    cv2.rectangle(image, (0, 0), (640, 60), bg_color, -1)
    
    cv2.putText(image, f'STATE: {current_status}', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Progress bar for the "Yell" trigger
    if distraction_timer > 0:
        bar_end = int((distraction_timer / 150) * 200)
        cv2.rectangle(image, (400, 20), (400 + bar_end, 40), (255, 255, 255), -1)

    cv2.imshow('Focus Shield v1.0', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()