# 🛡️ Focus-Shield
**A custom computer vision AI that tracks my study habits and literally yells at me when I get distracted.**
Let’s be real: staying focused during a late-night coding or study session is hard. I built Focus-Shield because generic Pomodoro timers weren't cutting it. I needed an AI that actually *watches* my behavior, understands the difference between looking at my phone and writing in my notebook, and holds me accountable in real-time. 
## 🧠 How it works (The 3-Class System)
Most generic face-trackers are binary: if you look away from the screen, you are "distracted." But as a student, looking down at a desk to write is actually productive. 
I trained a custom neural network from scratch to recognize three specific cognitive states:
1. 💻 **`SCREEN_FOCUS`**: Looking directly at the monitor.
2. 📝 **`PAPER_FOCUS`**: Head tilted down at the desk (Writing/Solving math). *The AI recognizes this as a safe, productive state.*
3. 📱 **`DISTRACTED`**: Looking sideways, checking a phone, or leaving the frame.
**The Trigger:** If the system detects the `DISTRACTED` state for ~5 continuous seconds, it bypasses the system audio and blasts the legendary *"Padhai kyun nahi ho rahi hai?!"* meme audio until I look back at my work.
## ⚙️ The Tech Stack & Architecture
* **Vision Pipeline (`vision_core.py`):** Uses MediaPipe FaceLandmarker and OpenCV. To ensure high performance and avoid background noise, the engine isolates exactly **16 high-value landmarks** (irises, eyelids, nose bridge, and chin).
* **The Brain (`focus_shield_v2.h5`):** A custom Sequential LSTM model built with TensorFlow/Keras. Instead of analyzing static images, it processes a temporal window of **30 frames** at a time to understand movement and intention over time.
* **Audio Engine:** Pygame handles asynchronous audio triggering so the video feed doesn't drop frames when the alarm goes off.
## 📂 Repository Breakdown
* `extract.py`: The data collection script used to record my own facial landmarks for the three classes.
* `train.py`: The Kaggle script used to compile and train the LSTM model. 
* `vision_core.py`: The core object-oriented engine handling all MediaPipe landmark extraction and HUD rendering.
* `predictor.py`: The main execution script that runs the live webcam feed, runs the inference, and triggers the audio.
## 🚀 Run it Locally
Want to get yelled at by your own computer?
1. Clone this repository:
   ```bash
   git clone https://github.com/Metamorpho-1/Focus-Shield.git
   cd Focus-Shield
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the live tracking engine:
   ```bash
   python predictor.py
   ```
   *(Press 'q' to quit the camera window).*
## 🔮 What's Next?
* **Fatigue Tracking:** Calculating the Eye Aspect Ratio (EAR) to detect micro-sleeps and suggest a 5-minute break.
* **OS App Blocking:** Using Python sub-processes to automatically minimize Spotify/Discord when focus drops.
