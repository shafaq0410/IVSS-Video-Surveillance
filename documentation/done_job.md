Intelligent Video Surveillance System (IVSS) — Freelance Task Brief

### 📌 Project Context

We are building a real-time intelligent video surveillance system in Python. The system supports multi-camera feeds and detects motion, objects, and faces. Based on detection, it triggers alerts and logs them in a UI dashboard.

The core structure has already been developed:

* Multi-camera support via shared memory
* Modular detection for motion, object (YOLOv8), and face recognition
* Frontend in Flask (dashboard, settings, alerts, registration)
* Alert logging in SQLite and desktop notifications
* Live monitoring via video stream

However, the system still needs improvement, validation, and extension based on real-world issues and professor feedback.

---

### ✅ What’s Already Done

* Backend logic for capturing video using multiprocessing and shared memory
* YOLOv8-based object detection
* face_recognition-based face recognition
* Background subtraction-based motion detection
* Dynamic detection control per camera (configured in settings page)
* Web dashboard built with Flask and Jinja2
* Alerts stored in SQLite and optionally sent via desktop/email
* Face registration through web interface