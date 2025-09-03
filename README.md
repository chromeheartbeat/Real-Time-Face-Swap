# Real-Time Face Swap with Virtual Camera  

This project is a **real-time face swapper** that uses **OpenCV**, **MediaPipe**, and **PyVirtualCam**.  
It captures your webcam feed, detects your face, and replaces it with a target face image (`target.jpg`).  
The processed video is streamed into a **virtual camera**, making it usable in Zoom, OBS, Google Meet, Discord, and more.  

⚠️ **Disclaimer:** This project is for **educational and research purposes only**. Do not use it for impersonation, fraud, or any malicious activities.  

---

## ✨ Features  
- Real-time face landmark detection with **MediaPipe FaceMesh**  
- Face alignment and triangulated warping for realistic swaps  
- Optional **seamless cloning** (better quality, slower)  
- Fast **alpha blending** (faster, smoother FPS)  
- Streams into a **virtual webcam**  
- Adjustable resolution and FPS  

---

## 🖥️ Requirements  

### 1. Create Virtual Environment  
```bash
python -m venv .venv
```

Activate it:  
- **Windows (PowerShell)**  
  ```bash
  .venv\Scripts\Activate
  ```  
- **Linux / macOS**  
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

You’ll also need:  
- A working webcam  
- A target face image (`target.jpg`) in the project folder  

---

## ⚙️ Configuration  
Edit the top of the script (`faceswap_cam.py`) to tweak settings:  

```python
CAM_INDEX = 0            # Camera device index (0 = default webcam)
TARGET_IMG_PATH = "target.jpg"
FRAME_W, FRAME_H = 640, 480   # Output resolution
FPS = 30                 # Frames per second
USE_SEAMLESS_CLONE = False  # True = better quality, False = faster
```

---

## ▶️ Usage  
1. Place `target.jpg` in the project folder.  
2. Activate `.venv` and run the script:  
   ```bash
   python faceswap_cam.py
   ```  
3. A preview window will open. Press **`q`** to quit.  
4. In apps like **Zoom, OBS, Google Meet, or Discord**, select the **virtual camera** created by PyVirtualCam.  

---

## 📂 Project Structure  
```
├── faceswap_cam.py   # Main script
├── target.jpg        # Target face (replace with your own)
├── README.md         # Documentation
├── requirements.txt  # Dependencies
└── .venv/            # Virtual environment (not committed to GitHub)
```

---

## 🔧 How It Works  
1. **Preprocess Target** – Detect and align the target face (`target.jpg`).  
2. **Triangulation** – Split face into triangles for natural warping.  
3. **Live Capture** – Read webcam feed in real time.  
4. **Warping & Blending** – Warp target face to match live landmarks.  
5. **Virtual Camera Output** – Stream final video through PyVirtualCam.  

---

## 🎥 Screenshot / Demo GIF  
👉 *(Add your preview screenshot or GIF here so visitors can quickly see the result!)*  

Example placeholder:  

![Demo GIF Placeholder](docs/demo.gif)  

---

## ⚠️ Disclaimer  
This software is for **research, learning, and fun experiments** only.  
Do not misuse it for identity theft, deepfakes, or deceptive purposes.  
