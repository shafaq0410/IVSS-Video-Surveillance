# 👁️‍🗨️ IVSS - Intelligent Video Surveillance System

**IVSS** (Intelligent Video Surveillance System) is a full-stack, intelligent, and scalable surveillance solution developed in Python. It leverages cutting-edge technologies in computer vision to monitor multiple camera feeds in real-time, detect motion and objects, recognize faces, and generate alerts through an intuitive web interface.

---

## 🚀 Project Overview

> 📡 "Smarter security begins with smarter surveillance."

IVSS is designed to offer **modular, real-time monitoring** for home, office, or public spaces. From identifying unauthorized movements to recognizing familiar faces, IVSS automates the surveillance process and alerts users instantly through email and on-screen notifications.

---

## 🧠 Core Features

| Feature            | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| 🎥 Multi-Camera Support | Easily add and manage multiple video sources for wider coverage.         |
| 🎯 Motion Detection  | Detects movement using OpenCV background subtraction.                      |
| 🧍 Object Detection | Real-time object detection using YOLOv8 via Ultralytics.                    |
| 🙂 Face Recognition | Recognizes registered faces using `face_recognition` library.               |
| 🚨 Smart Alerts     | Instant email and browser alerts for motion, objects, or unknown faces.     |
| 🌐 Web Interface    | Flask-based dashboard to monitor feeds, register faces, and view analytics. |
| 📊 Analytics        | Track system usage and detection events visually.                           |
| 🗃️ SQLite Logging   | Detection events are saved to a lightweight database for future reference.  |

---

## 🖥️ Tech Stack

| Layer         | Tools Used                                                                 |
|---------------|------------------------------------------------------------------------------|
| 🔙 Backend     | Python, Flask, OpenCV, YOLOv8 (Ultralytics), face_recognition, SQLite       |
| 🖼️ Frontend     | HTML5, CSS3, Bootstrap 5, JavaScript                                        |
| 🧠 AI/ML       | YOLOv8 for object detection, HOG + CNN for face recognition                 |
| 💾 Database    | SQLite3 for lightweight event logging                                       |
| ✉️ Notifications | SMTP (email alerts), browser-based UI notifications                        |

---

## 🛠️ System Architecture

Camera Feeds --> OpenCV Processing --> [YOLOv8 | Motion | Face Detection]
↓
Flask Backend Server
↓
Web UI (Live Feed, Alerts, Analytics)
↓
SQLite (Alert History & Faces)
↓
Email Notifications


---

## 📸 Screenshots

<img width="1910" height="902" alt="Screenshot 2025-08-07 000850" src="https://github.com/user-attachments/assets/df7d4c97-b500-49db-a5b1-6d914818f29b" />
<img width="1891" height="833" alt="Screenshot 2025-08-07 000905" src="https://github.com/user-attachments/assets/9a3c7321-36ba-4f42-8602-30a2dd36434d" />
<img width="1914" height="812" alt="Screenshot 2025-08-07 000922" src="https://github.com/user-attachments/assets/a265b3e3-8905-47c0-b6fd-6b8562b6f612" />
<img width="1902" height="894" alt="Screenshot 2025-08-07 000936" src="https://github.com/user-attachments/assets/2d0aaf35-72f5-4540-a04a-ee02a975c7a4" />
<img width="1862" height="904" alt="Screenshot 2025-08-07 001000" src="https://github.com/user-attachments/assets/e4e36e7c-77ef-45ca-b1e8-3ff5daa84130" />
<img width="1887" height="899" alt="Screenshot 2025-08-07 001017" src="https://github.com/user-attachments/assets/3d1a69a4-d479-480f-9df0-ddaa3b54ab4b" />






- ✅ Live camera feed
- ✅ Motion trigger alert
- ✅ Face registration interface
- ✅ System settings and analytics

---

## 🧪 How to Run the Project Locally

1. **Clone the repository**:
  
   git clone https://github.com/shafaq0410/IVSS.git
   cd IVSS
   
Create and activate virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

Install dependencies:
pip install -r requirements.txt


Run the Flask server:
python main.py


Open your browser and go to:

http://localhost:5000

------
📦 Folder Structure
css

IVSS/
├── static/
│   └── css, js, images
├── templates/
│   └── HTML files (dashboard, face registration, etc.)
├── detection/
│   └── motion.py, yolo_detector.py, face_recognizer.py
├── database/
│   └── models.py, init_db.py
├── main.py
├── requirements.txt
└── README.md

-------
✨ Highlights
📷 Real-time, low-latency streaming across multiple cameras.

📡 Smart alerts help take quick action in case of intrusions.

👥 Face-based recognition reduces false alerts.

🌐 Easy-to-use, intuitive interface for non-technical users.

📬 Alerts & Notifications
Motion or object detection triggers alerts.

Alerts include:

Timestamp

Camera source

Detected entity (object/person)

Email alert sent using SMTP (pre-configured).

Alerts are saved in a local SQLite database.

------

🙋‍♀️ Use Cases
🏠 Home Security

🏢 Office Surveillance

🧪 Lab Equipment Monitoring

🧓 Elderly Monitoring

🏫 School/College Security

------

🧑‍💻 Contributors
Shafaq , Samyak , Saumitra , Zaki , Gowthami

-----

📘 License
This project is open-source and available under the MIT License.

----

💡 Future Enhancements
📱 Mobile notifications

☁️ Cloud-based video storage

🔐 Face mask and weapon detection

🧾 Admin panel with user roles

📦 Dockerization for deployment

------
⭐ Give a Star!
If you found this project helpful, feel free to ⭐ star it on GitHub!
-----
git add .
git commit -m "Initial commit with full surveillance system"
git push origin main

----
🤝 Let's Connect!
Connect with me on LinkedIn or GitHub to discuss security systems, Python, AI, or just to say hi!
-----
"Security is not a product, but a process." – Bruce Schneier
