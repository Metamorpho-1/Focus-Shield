import cv2
import numpy as np
import os
import mediapipe as mp
from vision_core import SignSynthVision

def compile_flat_dataset(raw_videos_path, output_numpy_path, sequence_length=30):
    """
    Converts a flat folder of .mp4 videos (e.g., bear.mp4) into LSTM-ready NumPy matrices.
    """
    engine = SignSynthVision()
    
    if not os.path.exists(raw_videos_path):
        print(f"[ERROR] Cannot find folder at: {raw_videos_path}")
        return

    # Find every .mp4 file in the giant folder
    video_files = [v for v in os.listdir(raw_videos_path) if v.endswith('.mp4')]
    
    if len(video_files) == 0:
        print("[ERROR] No .mp4 files found! Check your path.")
        return

    for video_file in video_files:
        # 1. Get the word from the filename (e.g., 'bear.mp4' -> 'BEAR')
        action_name = video_file.replace('.mp4', '').upper()
        video_path = os.path.join(raw_videos_path, video_file)
        
        cap = cv2.VideoCapture(video_path)
        frame_data = []
        
        # 2. Extract the math
        while cap.isOpened() and len(frame_data) < sequence_length:
            ret, frame = cap.read()
            if not ret: break 
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            detection_result = engine.detector.detect(mp_image)
            keypoints = engine.extract_keypoints(detection_result)
            frame_data.append(keypoints)
        
        cap.release()
        
        # 3. Pad short videos with zeros
        while len(frame_data) < sequence_length:
            frame_data.append(np.zeros(126))
            
        # 4. Save to MP_Data/BEAR/0/
        save_dir = os.path.join(output_numpy_path, action_name, "0")
        os.makedirs(save_dir, exist_ok=True)
        
        for frame_num, matrix in enumerate(frame_data):
            np.save(os.path.join(save_dir, f"{frame_num}.npy"), matrix)
        
        print(f"[COMPILED] GESTURE: {action_name} | File: {video_file}")

if __name__ == "__main__":
    # IMPORTANT: Put your absolute paths back in here!
    RAW_KAGGLE_DIR = "/Users/lakshyagupta/PycharmProjects/PythonProject/sign language translator/Sample Videos" 
    PROCESSED_NPY_DIR = "/Users/lakshyagupta/PycharmProjects/PythonProject/MP_Data"
    
    print("[SYS] INITIATING FLAT-FILE MASS COMPILATION...")
    compile_flat_dataset(RAW_KAGGLE_DIR, PROCESSED_NPY_DIR)
    print("[SYS] COMPILATION COMPLETE. DATA READY FOR TRAINING.")