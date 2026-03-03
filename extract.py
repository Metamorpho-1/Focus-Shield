import os, cv2, numpy as np, mediapipe as mp
from vision_core import FocusVision
from pathlib import Path

engine = FocusVision()
cap = cv2.VideoCapture(0)

# UPDATED: 3 Classes for better precision
actions = ['SCREEN_FOCUS', 'PAPER_FOCUS', 'DISTRACTED']
no_sequences = 100 
sequence_length = 30 
DATA_PATH = Path(os.getcwd()) / "Shield_Data_V2"

for action in actions:
    for sequence in range(no_sequences):
        (DATA_PATH / action / str(sequence)).mkdir(parents=True, exist_ok=True)

for action in actions:
    print(f"--- PREPARING TO RECORD: {action} ---")
    cv2.waitKey(4000)
    
    for sequence in range(no_sequences):
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            results = engine.detector.detect(mp_image)
            keypoints = engine.extract_keypoints(results)
            
            npy_path = DATA_PATH / action / str(sequence) / f"{frame_num}.npy"
            np.save(str(npy_path), keypoints)

            image = engine.render_HUD(frame, results)
            cv2.putText(image, f'CLASS: {action} | SEQ: {sequence}', (15,35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Shield Data V2 - Collector', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()