# import os
# import cv2
# import face_recognition
# import pickle
# import numpy as np
# from multiprocessing import shared_memory

# encodings_file = "encodings.pickle"
# if os.path.exists(encodings_file):
#     with open(encodings_file, "rb") as f:
#         data = pickle.load(f)
#     known_encodings = data["encodings"]
#     known_names = data["names"]
# else:
#     known_encodings = []
#     known_names = []

# def face_recognition_process(shm_name, shape, output_queue, cam_id):
#     shared_mem = shared_memory.SharedMemory(name=shm_name)
#     frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shared_mem.buf)
    
#     while True:
#         frame = frame_buffer.copy()
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         boxes = face_recognition.face_locations(rgb_frame)
#         encodings = face_recognition.face_encodings(rgb_frame, boxes)
        
#         for encoding, (top, right, bottom, left) in zip(encodings, boxes):
#             matches = face_recognition.compare_faces(known_encodings, encoding,          tolerance=0.5)
#             name = "Unknown"
#             if True in matches:
#                 matched_idxs = [i for (i, b) in enumerate(matches) if b]
#                 counts = {}
#                 for i in matched_idxs:
#                     recognized_name = known_names[i]
#                     counts[recognized_name] = counts.get(recognized_name, 0) + 1
#                 name = max(counts, key=counts.get)
            
#             output_queue.put({
#                 "cam_id": cam_id,
#                 "name": name,
#                 "bbox": (left, top, right, bottom),
#                 "detection_type": "face"
#             })


# if __name__ == "__main__":
#     print("Run main.py to start the system.")


# import os
# import cv2
# import face_recognition
# import pickle
# import numpy as np
# from multiprocessing import shared_memory

# # Load known face encodings
# encodings_file = "encodings.pickle"
# if os.path.exists(encodings_file):
#     with open(encodings_file, "rb") as f:
#         data = pickle.load(f)
#     known_encodings = data["encodings"]
#     known_names = data["names"]
# else:
#     known_encodings = []
#     known_names = []

# def face_recognition_process(shm_name, shape, output_queue, cam_id):
#     shared_mem = shared_memory.SharedMemory(name=shm_name)
#     frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shared_mem.buf)

#     print(f"[INFO] Face recognition started for Camera {cam_id}...")

#     try:
#         while True:
#             if frame_buffer is None or frame_buffer.size == 0:
#                 continue  # Skip if frame is empty
            
#             # ✅ Resize image to speed up processing (Reduce resolution by half)
#             small_frame = cv2.resize(frame_buffer, (shape[1] // 2, shape[0] // 2))
            
#             # ✅ Ensure correct data type
#             if small_frame.dtype != np.uint8:
#                 small_frame = small_frame.astype(np.uint8)

#             # ✅ Convert frame to RGB
#             rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#             # Face detection
#             boxes = face_recognition.face_locations(rgb_frame)
#             encodings = face_recognition.face_encodings(rgb_frame, boxes)

#             detected_faces = []
#             for encoding, (top, right, bottom, left) in zip(encodings, boxes):
#                 matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
#                 name = "Unknown"

#                 if True in matches:
#                     matched_idxs = [i for (i, b) in enumerate(matches) if b]
#                     counts = {}
#                     for i in matched_idxs:
#                         recognized_name = known_names[i]
#                         counts[recognized_name] = counts.get(recognized_name, 0) + 1
#                     name = max(counts, key=counts.get)

#                 detected_faces.append({
#                     "cam_id": cam_id,
#                     "name": name,
#                     "bbox": (left, top, right, bottom),
#                     "detection_type": "face"
#                 })

#             if detected_faces:
#                 output_queue.put(detected_faces)

#     except Exception as e:
#         print(f"[ERROR] Face recognition process encountered an issue: {str(e)}")

#     finally:
#         print(f"[INFO] Face recognition shutting down for Camera {cam_id}...")
#         shared_mem.close()

import os
import cv2
import face_recognition
import pickle
import numpy as np
import time
from multiprocessing import shared_memory

# Load known face encodings
encodings_file = "encodings.pickle"
if os.path.exists(encodings_file):
    with open(encodings_file, "rb") as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]
else:
    known_encodings = []
    known_names = []

def face_recognition_process(shm_name, shape, output_queue, cam_id):
    shared_mem = shared_memory.SharedMemory(name=shm_name)
    frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shared_mem.buf)

    print(f"[INFO] Face recognition started for Camera {cam_id}...")

    try:
        while True:
            frame = frame_buffer.copy()
            if frame is None or frame.size == 0:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_frame = cv2.resize(rgb_frame, (shape[1] // 2, shape[0] // 2))

            boxes = face_recognition.face_locations(small_frame)
            encodings = face_recognition.face_encodings(small_frame, boxes)

            detected_faces = []
            for encoding, (top, right, bottom, left) in zip(encodings, boxes):
                matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
                name = "Unknown"

                if True in matches:
                    matched_idxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matched_idxs:
                        recognized_name = known_names[i]
                        counts[recognized_name] = counts.get(recognized_name, 0) + 1
                    name = max(counts, key=counts.get)

                # Scale box back to original size
                scale_x, scale_y = shape[1] / (shape[1] // 2), shape[0] / (shape[0] // 2)
                left = int(left * scale_x)
                top = int(top * scale_y)
                right = int(right * scale_x)
                bottom = int(bottom * scale_y)

                # Draw box and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detected_faces.append({
                    "cam_id": cam_id,
                    "name": name,
                    "bbox": (left, top, right, bottom),
                    "detection_type": "face"
                })

            if detected_faces:
                image_path = save_face_frame(frame, cam_id, name)
                for face in detected_faces:
                    face["image_path"] = image_path
                output_queue.put(detected_faces)

    except Exception as e:
        print(f"[ERROR] Face recognition process encountered an issue: {str(e)}")

    finally:
        print(f"[INFO] Face recognition shutting down for Camera {cam_id}...")
        shared_mem.close()

# 🔹 Save Face Detection Image
def save_face_frame(frame, cam_id, label):
    os.makedirs("face_alerts", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"face_cam{cam_id}_{label}_{timestamp}.jpg"
    image_path = os.path.join("face_alerts", filename)

    cv2.imwrite(image_path, frame)
    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        print(f"[INFO] Face detection frame saved: {image_path}")
        return image_path
    else:
        print(f"[ERROR] Failed to save face image for Camera {cam_id}.")
        return None
