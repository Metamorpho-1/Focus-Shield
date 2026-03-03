import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from neural_model import build_vanguard_model

# ==========================================
# 1. CONFIGURE PATHS & VARIABLES
# ==========================================
# REPLACE THIS WITH YOUR ABSOLUTE PATH TO MP_DATA!
DATA_PATH = "/Users/lakshyagupta/PycharmProjects/PythonProject/MP_Data"

# Automatically detect the actions based on the folder names in MP_Data
actions = np.array([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
sequence_length = 30

print(f"[SYS] DETECTED ACTIONS: {actions}")

# ==========================================
# 2. LOAD AND PREPROCESS DATA
# ==========================================
print("[SYS] LOADING MATRICES INTO RAM...")
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    sequence_folders = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
    
    for sequence_folder in sequence_folders:
        window = []
        for frame_num in range(sequence_length):
            # Load each of the 30 frames
            frame_path = os.path.join(action_path, sequence_folder, f"{frame_num}.npy")
            res = np.load(frame_path)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split the data: 85% for training, 15% for testing/validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
print(f"[SYS] DATA SPLIT: {X_train.shape[0]} Training Samples, {X_test.shape[0]} Validation Samples.")

# ==========================================
# 3. BOOT THE NEURAL NETWORK
# ==========================================
print("[SYS] INITIALIZING LSTM ARCHITECTURE...")
model = build_vanguard_model(num_actions=actions.shape[0])

# ==========================================
# 4. TRAINING LOOP
# ==========================================
print("[SYS] COMMENCING NEURAL TRAINING...")
# We use 'history' to capture the training data so we can graph it later
history = model.fit(X_train, y_train, epochs=75, validation_data=(X_test, y_test))

# ==========================================
# 5. EXPORT AND VISUALIZE
# ==========================================
model.save('vanguard_sign_synth.h5')
print("[SYS] TRAINING COMPLETE. MODEL SAVED AS 'vanguard_sign_synth.h5'")

# Generate the Vanguard Performance Graph
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy Plot
ax1.plot(history.history['categorical_accuracy'], label='Training Accuracy', color='#00f2ff')
ax1.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy', color='#facc15')
ax1.set_title('Neural Network Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Loss Plot
ax2.plot(history.history['loss'], label='Training Loss', color='#00f2ff')
ax2.plot(history.history['val_loss'], label='Validation Loss', color='#facc15')
ax2.set_title('Neural Network Loss (Error Rate)')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.show()