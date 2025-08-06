import smtplib
import cv2
import numpy as np
import os
import time
import sqlite3
import json
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import plyer
from app import Alert, db  # Adjust the import if needed
from flask import current_app
from app import CameraSetting  # Replace 'your_app' with your actual app module name
# 🔹 Global settings
ALERT_INTERVAL = 60
last_alert_time = {
    "motion": {},
    "object": {},
    "face": {}
}


def load_camera_settings():
    """Loads camera settings from database."""
    try:
        with current_app.app_context():
            settings = CameraSetting.query.all()
            if settings:
                setts = [setting.to_dict() for setting in settings]
                print(setts)
                return setts
            else:
                # Return default settings if no settings exist in database
                return []
    except Exception as e:
        print(f"Error loading camera settings: {e}")
        # Return default settings on error
        return []

# 🔹 Capture Frame Function
def capture_frame(cam_id):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    time.sleep(2)

    if not cap.isOpened():
        print(f"[ERROR] Could not access camera {cam_id}")
        return None

    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        image_path = f"alert_frame_cam{cam_id}.jpg"
        cv2.imwrite(image_path, frame)
        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            print(f"[INFO] Frame captured successfully: {image_path}")
            return image_path
        else:
            print("[ERROR] Captured image is empty or corrupted.")
            return None
    else:
        print("[ERROR] Failed to capture frame.")
        return None

# 🔹 Send Email with Attachment
def send_email_notification(subject, message, attachment_path=None):
    sender_email = "@gmail.com"
    receiver_email = "@gmail.com"
    password = ""

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    if attachment_path and os.path.exists(attachment_path) and os.path.getsize(attachment_path) > 0:
        with open(attachment_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
            msg.attach(part)
        print(f"[INFO] Image attached: {attachment_path}")
    else:
        print("[WARNING] No valid image attached.")

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("[INFO] Email sent successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")

# 🔹 Send Local Notification
def send_local_notification(title, message):
    plyer.notification.notify(
        title=title,
        message=message,
        app_name='Alert System',
        timeout=10
    )


def store_alert(camera, location, message, severity):
    alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_alert = Alert(
        camera=camera,
        location=location,
        time=alert_time,
        message=message,
        severity=severity,
        status='New',  # Default status
        is_true_detection=None  # Will be reviewed later
    )
    db.session.add(new_alert)
    db.session.commit()
    print(f"[INFO] Alert stored: {camera}, {location}, {alert_time}, {message}, {severity}")

# 🔹 Check Alert Interval (per alert type)
def can_trigger_alert(alert_type, cam_id):
    global last_alert_time
    current_time = time.time()

    last_time = last_alert_time[alert_type].get(cam_id, 0)
    if current_time - last_time >= ALERT_INTERVAL:
        last_alert_time[alert_type][cam_id] = current_time
        return True
    else:
        print(f"[DEBUG] Skipping {alert_type} alert for Camera {cam_id} due to time restriction.")
        return False

# 🔹 Main Alert Processing Function
import time
from collections import defaultdict

import time
from collections import defaultdict
from queue import Empty
from app import app  # or whatever your Flask file is named

def alert_process(object_queue, face_queue, motion_queue):
    with app.app_context():
        camera_settings = load_camera_settings()
        print(camera_settings)
        log_file_path = "alerts_log.txt"

        last_alert_times = defaultdict(lambda: 0)
        alert_interval = 10  # seconds

        def log_to_file(alert_type, cam_id, message, severity, image_path):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"[{timestamp}] {alert_type.upper()} ALERT - Camera: {cam_id} | Severity: {severity} | Message: {message} | Image: {image_path}\n"
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(log_line)

        while True:
            now = time.time()
            # 🔥 Face Recognition Alerts

            try:
                alert = face_queue.get_nowait()
                print('Im here --------------------------')
                print(alert)
                alert = alert[0]  # Assuming alert is a list-like with one item
                cam_id = alert.get("cam_id")
                name = alert.get("label") or alert.get("name", "Unknown face")

                if isinstance(cam_id, int) and 0 <= cam_id < len(camera_settings):
                    if "face" in camera_settings[cam_id].get("detections", []):
                        key = ("face", cam_id)
                        if now - last_alert_times[key] >= alert_interval:
                            image_path = alert.get("image") or capture_frame(cam_id)
                            message = f"Face detected: {name}"
                            severity = alert.get("severity", "high")
                            log_to_file("face", cam_id, message, severity, image_path)
                            store_alert(f"Camera {cam_id}", "Face Recognition", message, severity)
                            send_email_notification("Face Detected", message, image_path)
                            send_local_notification("Face Detected", message)
                            last_alert_times[key] = now

                            # ✅ Flush the rest of the queue
                            while True:
                                try:
                                    face_queue.get_nowait()
                                except Empty:
                                    break
            except Empty:
                pass

            # # 🔥 Motion Detection Alerts
            try:

                alert = motion_queue.get_nowait()
                cam_id = alert.get("cam_id")
                if isinstance(cam_id, int) and 0 <= cam_id < len(camera_settings):
                    if "motion" in camera_settings[cam_id].get("detections", []):
                        key = ("motion", cam_id)
                        if now - last_alert_times[key] >= alert_interval:
                            image_path = alert.get("image") or capture_frame(cam_id)
                            message = alert.get("message", "Motion detected")
                            severity = alert.get("severity", "medium")
                            log_to_file("motion", cam_id, message, severity, image_path)
                            store_alert(f"Camera {cam_id}", "Motion Detection", message, severity)
                            send_local_notification("Motion Detected", message)
                            send_email_notification("Motion Detected", message, image_path)
                            last_alert_times[key] = now
            except Empty:
                pass

            # 🔥 Object Detection Alerts
            try:
                alert = object_queue.get_nowait()
                cam_id = alert.get("cam_id")
                if isinstance(cam_id, int) and 0 <= cam_id < len(camera_settings):
                    if "object" in camera_settings[cam_id].get("detections", []):
                        label = alert.get("detections")[0]["label"]
                        key = ("object", cam_id)
                        print(alert)
                        if now - last_alert_times[key] >= alert_interval:
                            # image_path = alert.get("image") or capture_frame(cam_id)
                            message = f"Object detected: {label}"
                            severity = alert.get("severity", "high")
                            log_to_file("object", cam_id, message, severity, "image_path")
                            store_alert(f"Camera {cam_id}", "Object Detection", message, severity)
                            send_email_notification("Object Detected", message, "image_path")
                            send_local_notification("Object Detected", message)
                            last_alert_times[key] = now
                                                    # ✅ Flush the rest of the queue
                            while True:
                                try:
                                    object_queue.get_nowait()
                                except Empty:
                                    break
            except Empty:
                pass



            time.sleep(0.05)


