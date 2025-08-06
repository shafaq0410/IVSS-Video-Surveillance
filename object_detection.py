import cv2
from ultralytics import YOLO
from multiprocessing import shared_memory
import numpy as np
import time
import os

def object_detection_process(shm_name, shape, output_queue, cam_id,objectThreshold):
    """
    Continuously reads frames from shared memory, runs YOLO object detection,
    and outputs detections via the output_queue. Also draws bounding boxes and
    saves the processed frame with object label in filename.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)

    # model = YOLO('best.pt')
    model = YOLO("yolo11m.pt")

    while True:
        time.sleep(0.01)

        frame = np.copy(frame_buffer)

        if frame is None or frame.shape != shape or np.all(frame == 0):
            print(f"[ERROR] Camera {cam_id}: Invalid or empty frame. Saving for inspection...")

            os.makedirs("invalid_frames", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            invalid_path = os.path.join("invalid_frames", f"invalid_cam{cam_id}_{timestamp}.jpg")
            try:
                cv2.imwrite(invalid_path, frame)
                print(f"[INFO] Invalid frame saved to: {invalid_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save invalid frame: {e}")
            continue

        print(f"[DEBUG] Camera {cam_id}: Frame mean pixel value: {frame.mean():.2f}")

        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[ERROR] Camera {cam_id}: Failed to convert color space: {e}")
            continue

        try:
            # print(frame)
            results = model.predict(frame, imgsz=320, verbose=False,conf=objectThreshold)
        except Exception as e:
            print(f"[ERROR] YOLO prediction failed for camera {cam_id}: {e}")
            continue

        detected_objects = []
        for result in results:
            try:
                boxes = result.boxes.cpu().numpy()
            except Exception as e:
                print(f"[ERROR] Camera {cam_id}: Failed to get bounding boxes: {e}")
                continue

            for box in boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    label = model.names[int(box.cls[0])]
                    confidence = float(box.conf[0])
                except Exception as e:
                    print(f"[ERROR] Camera {cam_id}: Error parsing bounding box: {e}")
                    continue

                detected_objects.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 🔻 Save frame with label in filename
                os.makedirs("objects_detected", exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"detected_cam{cam_id}_{label}_{timestamp}.jpg"
                filepath = os.path.join("objects_detected", filename)
                try:
                    cv2.imwrite(filepath, frame)
                    print(f"[INFO] Saved detected object: {filepath}")
                except Exception as e:
                    print(f"[ERROR] Failed to save detection image: {e}")

        if detected_objects:
            output_queue.put({"cam_id": cam_id, "detections": detected_objects})

if __name__ == "__main__":
    print("Run main.py to start the system.")
