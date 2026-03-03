import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, urllib.request, ssl

ssl._create_default_https_context = ssl._create_unverified_context

class FocusVision:
    def __init__(self):
        self.model_path = 'face_landmarker.task'
        if not os.path.exists(self.model_path):
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def extract_keypoints(self, detection_result):
        """
        REFINED: Isolates only high-value landmarks for focus detection.
        Interest Points: Iris, Eyelids, Nose Bridge, and Chin.
        """
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            
            # 1. Select specific landmarks (Iris: 468-477, Eyes: 33, 133, 362, 263, Pose: 1, 152)
            # Total points: 16 selected landmarks * 3 (x,y,z) = 48 points
            interest_indices = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 33, 133, 362, 263, 1, 152]
            
            extracted = []
            for idx in interest_indices:
                lm = landmarks[idx]
                extracted.extend([lm.x, lm.y, lm.z])
            
            return np.array(extracted)
        
        # Return 48 zeros if no face is found
        return np.zeros(16 * 3)

    def render_HUD(self, image, detection_result):
        if detection_result.face_landmarks:
            h, w, _ = image.shape
            # Draw only the points we are tracking
            interest_indices = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 33, 133, 362, 263, 1, 152]
            for idx in interest_indices:
                lm = detection_result.face_landmarks[0][idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        return image