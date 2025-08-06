import cv2
import numpy as np
from multiprocessing import shared_memory
import os
import time

def motion_detection_process(shm_name, shape, motion_queue, cam_id,varThreshold):
    shared_mem = shared_memory.SharedMemory(name=shm_name)
    frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shared_mem.buf)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=varThreshold)
    
    while True:
        frame = frame_buffer.copy()

        # ✅ Ensure the frame is valid before processing
        if frame is None or frame.size == 0:
            print(f"[ERROR] Camera {cam_id}: Invalid frame received.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(gray)
        motion_score = cv2.countNonZero(fg_mask)

        if motion_score > 100:  # Adjust threshold if needed
            image_path = save_motion_frame(frame, cam_id)

            if image_path:
                alert_data = {
                    "cam_id": cam_id,
                    "message": f"Motion detected with score {motion_score}",
                    "severity": "medium",
                    "detection_type": "motion",
                    "image_path": image_path
                }
                motion_queue.put(alert_data)
            else:
                print(f"[ERROR] Camera {cam_id}: Failed to save motion frame.")

# 🔹 Save Motion Frame with Timestamp and Folder
def save_motion_frame(frame, cam_id):
    os.makedirs("motion_alerts", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join("motion_alerts", f"motion_cam{cam_id}_{timestamp}.jpg")
    
    cv2.imwrite(image_path, frame)
    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        print(f"[INFO] Motion frame saved: {image_path}")
        return image_path
    else:
        print(f"[ERROR] Failed to save motion image for Camera {cam_id}.")
        return None

if __name__ == "__main__":
    print("Run main.py to start the system.")
