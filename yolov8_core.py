import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import os
import uuid

reader = easyocr.Reader(['en'])
model = YOLO("yolov8n.pt")

def estimate_speed(distance_px, time_sec, ppm=10):
    distance_m = distance_px / ppm
    speed_mps = distance_m / time_sec
    speed_kmph = speed_mps * 3.6
    return round(speed_kmph, 2)

def process_video(video_path, save_path=None, roi=None):
    os.makedirs("outputs", exist_ok=True)
    if save_path is None:
        unique_id = uuid.uuid4().hex[:8]
        save_path = f"outputs/result_{unique_id}.mp4"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ppm = 10

    width = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    tracker = {}
    prev_time = {}
    detected_plates = set()

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls)
            if model.names[cls_id] != 'car':
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if roi and not (roi[0] <= center[0] <= roi[2] and roi[1] <= center[1] <= roi[3]):
                continue

            obj_id = f"car_{i}"

            if obj_id in tracker:
                px_distance = np.linalg.norm(np.array(center) - np.array(tracker[obj_id]))
                time_diff = (frame_id - prev_time[obj_id]) / fps
                speed = estimate_speed(px_distance, time_diff, ppm)
                cv2.putText(frame, f"Speed: {speed} km/h", (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            tracker[obj_id] = center
            prev_time[obj_id] = frame_id

            plate_text = ""
            plate_crop = frame[y1:y2, x1:x2]

            if plate_crop.size > 0:
                plate_result = reader.readtext(plate_crop)
                for (bbox, text, prob) in plate_result:
                    text = text.strip()
                    if 4 <= len(text) <= 15 and prob > 0.4:
                        plate_text = text
                        detected_plates.add(plate_text)
                        break

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, model.names[cls_id], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if plate_text:
                cv2.putText(frame, f"Plate: {plate_text}", (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    
    print("✅ Video processed.")
    print("📋 Detected Plates:", detected_plates)
    
    return save_path
