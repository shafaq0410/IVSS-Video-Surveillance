import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import os
import json
import time
import cv2

from video_capture import video_capture_process
from motion_detection import motion_detection_process
from object_detection import object_detection_process
from face_recognition_module import face_recognition_process
from alert_module import alert_process
from flask import current_app
from app import db, CameraSetting  # Replace 'your_app' with your actual app module name
from app import app  # or whatever your Flask file is named

FRAME_SHAPE = (240, 320, 3)  # (height, width, channels)

def create_shared_memory(name, size):
    try:
        return shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        try:
            existing_shm = shared_memory.SharedMemory(name=name)
            existing_shm.unlink()
        except FileNotFoundError:
            pass
        return shared_memory.SharedMemory(name=name, create=True, size=size)



def load_camera_settings():
    """Loads camera settings from database."""
    try:
        with current_app.app_context():
            settings = CameraSetting.query.all()
            if settings:
                setts = [setting.to_dict() for setting in settings]
                return setts
            else:
                # Return default settings if no settings exist in database
                return []
    except Exception as e:
        print(f"Error loading camera settings: {e}")
        # Return default settings on error
        return []
    
def save_detection_image(frame, cam_id, detection_type, label=None):
    folder = "objects_detected" if detection_type == "object" else f"{detection_type}_alerts"
    os.makedirs(folder, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    label_part = f"_{label}" if label else ""
    filename = f"{detection_type}_cam{cam_id}{label_part}_{timestamp}.jpg"
    image_path = os.path.join(folder, filename)

    cv2.imwrite(image_path, frame)
    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        print(f"[INFO] {detection_type.capitalize()} detection frame saved: {image_path}")
        return image_path
    else:
        print(f"[ERROR] Failed to save {detection_type} detection image for Camera {cam_id}.")
        return None

if __name__ == "__main__":
    with app.app_context():
        camera_settings = load_camera_settings()
        shared_mem_list = []
        processes = []

        object_queue = mp.Queue()
        face_queue = mp.Queue()
        motion_queue = mp.Queue()

        for i, cam_config in enumerate(camera_settings):
            shm_name = f"video_frame_shm_{i}"
            detections = cam_config.get("detections", [])

            shm = create_shared_memory(shm_name, int(np.prod(FRAME_SHAPE)))
            shared_mem_list.append(shm)

            try:
                source = int(cam_config["source"])  # directly use index
            except (KeyError, ValueError):
                print(f"[ERROR] Invalid camera source in config: {cam_config.get('source')}")
                continue

            processes.append(mp.Process(target=video_capture_process, args=(shm_name, FRAME_SHAPE, source, i)))

            if "motion" in detections:
                processes.append(mp.Process(target=motion_detection_process, args=(shm_name, FRAME_SHAPE, motion_queue, i,cam_config.get('motionThreshold'))))
            if "object" in detections:
                processes.append(mp.Process(target=object_detection_process, args=(shm_name, FRAME_SHAPE, object_queue, i,cam_config.get('objectThreshold'))))
            if "face" in detections:
                processes.append(mp.Process(target=face_recognition_process, args=(shm_name, FRAME_SHAPE, face_queue, i)))

        # Add alert process once, not inside loop
        processes.append(mp.Process(target=alert_process, args=(object_queue, face_queue, motion_queue)))

        try:
            for p in processes:
                p.start()
            for p in processes:
                p.join()
        finally:
            print("\n[INFO] Shutting down all processes...")
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join()
            for shm in shared_mem_list:
                shm.close()
                shm.unlink()
            print("[INFO] Cleanup complete.")
