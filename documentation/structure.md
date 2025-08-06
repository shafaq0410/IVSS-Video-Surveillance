### 🧩 Remaining Work / Freelance Task Scope

#### 🟡 1. Improve Detection Quality & Alert Precision

* Problem: False positives are common; false negatives aren't logged or measured.
* Action Needed:

  * Log precision, recall, and F1-score for motion, object, and face detection.
  * Build a basic review interface to mark alerts as valid/invalid (optional).
  * Optimize thresholds for each model (YOLO confidence, motion score, face match score).

---

#### 🟡 2. Evaluate or Replace YOLOv8

* Problem: YOLOv8 is used out-of-the-box with no training or modification.
* Action Needed:

  * Explore YOLO alternatives: YOLO-NAS, MobileNet SSD, or Faster R-CNN.
  * Justify YOLO or suggest a better model based on test results.

---

#### 🟡 3. Model Fine-Tuning & Hyperparameter Tuning

* Fine-tune YOLO on a small custom dataset (e.g., faces, persons, bags).
* Adjust YOLO hyperparameters:

  * confidence threshold
  * image size
  * epochs, batch size, learning rate
* Optional: Use wandb or Optuna for automated tuning.

---

#### 🟡 4. Edge Case Handling

* Simulate edge cases:

  * Low lighting
  * Camera shake
  * Small or far objects
  * Black frames from IP cameras
* Implement fallback logic: e.g., frame validity check, alert suppression.

---

#### 🟡 5. Solve Frame Corruption/Black Image Bug

* Issue: Black frames with white lines are saved occasionally during alerts.
* Required:

  * Debug shared memory read-write timing issues
  * Validate frames before saving (shape check, average pixel brightness, etc.)

---

#### 🟡 6. Improve Alert System

* Store and tag alerts with correct type (face/object/motion).
* Attach image only if it is valid (not black/corrupted).
* Include camera source IP/index and timestamp in alerts.

---

#### 🟡 7. Build Evaluation/Test Suite

* Write a script or Jupyter notebook that:

  * Runs detection on test video(s)
  * Logs true/false positives and negatives
  * Reports precision, recall, and F1 score

---

#### 🟡 8. Improve Face Recognition Accuracy

* Currently uses default dlib encodings.
* Optionally:

  * Integrate FaceNet or InsightFace for better results.
  * Add face similarity threshold control.

---

#### 🟡 9. Deployment

* Create:

  * requirements.txt
  * README.md
  * Optional: Dockerfile (for Windows & Ubuntu compatibility)

---

### 📂 Current Project Structure

* main.py: Starts multiprocessing, handles camera feeds
* alert_module.py: Generates alerts, sends notifications
* video_capture.py: Captures video using OpenCV
* object_detection.py: Uses YOLOv8
* motion_detection.py: Uses background subtraction
* face_recognition_module.py: Uses dlib-based recognition
* templates/: Flask HTML files (dashboard, settings, register face, alerts)
* static/: CSS/JS files
* camera_settings.json: User settings
* alerts.db: SQLite DB for alerts
* dataset/: Uploaded face images

---

### ⚙️ Technologies Used

* Python (OpenCV, face\_recognition, ultralytics/YOLOv8, Flask, SQLite)
* HTML/CSS (Jinja2 Templates)
* Multiprocessing + Shared Memory
* plyer (for desktop notifications)
* SMTP (for email alerts)

---

### 🔐 Credentials (to be configured)

* Gmail ID/password for sending alerts
* IP camera URLs (e.g., DroidCam: http://<ip>:4747/video)

---

### 📈 Goals for Freelancer

1. Improve model reliability (precision/recall)
2. Make detection more robust and modular
3. Handle edge cases + resolve frame bugs
4. Optional: improve face recognition or retrain models
5. Deliver a production-ready, testable IVSS app