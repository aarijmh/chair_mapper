"""
chair_reid_3d_tracker.py

Detect and track unique chairs across a moving-camera video.
Keeps consistent IDs even if chairs disappear and reappear.

Dependencies:
    pip install ultralytics torch torchvision transformers opencv-python pillow tqdm pandas
"""

import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from collections import deque
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from scipy.spatial.distance import cosine

# ================= CONFIG =================
VIDEO_PATH = "output.mp4"
OUTPUT_PATH = "output_tracked_3d.mp4"

FRAME_SKIP = 2
SPATIAL_WEIGHT = 0.5
BASE_THRESHOLD = 0.75
INACTIVITY_LIMIT = 200
MAX_EMB_MEMORY = 5
# ==========================================

# Initialize YOLO (chair detection)
print("🧠 Loading YOLOv8 model...")
yolo_model = YOLO("yolov8n.pt")

# Initialize CLIP (appearance embeddings)
print("🎯 Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---------- Helper Functions ----------

def get_clip_embedding(image_pil):
    """Return normalized CLIP embedding for an image."""
    inputs = clip_processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()

def cosine_similarity(a, b):
    return 1 - cosine(a, b)

def get_average_embedding(embeddings):
    return np.mean(np.stack(embeddings), axis=0)

def combined_score(emb_new, emb_old, pos_new, pos_old):
    """Combine appearance and spatial similarity."""
    appearance_score = cosine_similarity(emb_new, emb_old)
    if pos_new is None or pos_old is None:
        return appearance_score
    spatial_dist = np.linalg.norm(pos_new - pos_old)
    spatial_score = max(0, 1 - spatial_dist / 3.0)
    return (1 - SPATIAL_WEIGHT) * appearance_score + SPATIAL_WEIGHT * spatial_score

def match_chair(emb, pos, known_chairs, frame_idx):
    """Match current detection to existing chair IDs."""
    best_id, best_score = None, 0
    for cid, data in known_chairs.items():
        # Skip long-inactive chairs
        if frame_idx - data["last_seen"] > INACTIVITY_LIMIT:
            continue

        avg_emb = get_average_embedding(data["embeddings"])
        score = combined_score(emb, avg_emb, pos, data["position"])
        if score > best_score:
            best_score, best_id = score, cid

    # Adaptive threshold based on known diversity
    dynamic_thresh = BASE_THRESHOLD - 0.05 * np.log1p(len(known_chairs))
    if best_score > dynamic_thresh:
        return best_id
    return None

def update_chair_record(cid, emb, pos, frame_idx, known_chairs):
    """Update embeddings, position, and timestamp for matched chair."""
    data = known_chairs[cid]
    data["embeddings"].append(emb)
    if len(data["embeddings"]) > MAX_EMB_MEMORY:
        data["embeddings"].popleft()
    data["position"] = 0.8 * data["position"] + 0.2 * pos
    data["last_seen"] = frame_idx

# ---------- ORB-based Camera Motion Tracking ----------

orb = cv2.ORB_create(1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
R_total = np.eye(3)
t_total = np.zeros((3, 1))
prev_frame, prev_kp, prev_des = None, None, None

# ---------- Video Setup ----------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

known_chairs = {}
chair_counter = 0
frame_index = 0
results_log = []

print("🚀 Starting 3D-aware chair tracking...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_index % FRAME_SKIP != 0:
        frame_index += 1
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    # Estimate camera motion
    if prev_frame is not None and des is not None and prev_des is not None and len(prev_kp) > 8:
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > 20:
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
            E, _ = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0,0), method=cv2.RANSAC, prob=0.999)
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, pts1, pts2)
                R_total = R @ R_total
                t_total += R_total @ t

    prev_frame, prev_kp, prev_des = frame, kp, des

    # Detect chairs
    detections = yolo_model(frame, verbose=False)[0]
    for box in detections.boxes:
        cls = int(box.cls[0])
        if yolo_model.names[cls].lower() != "chair":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        emb = get_clip_embedding(pil_img)

        # Approximate 3D position
        center = np.array([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, 1.0])
        pos_3d = (R_total @ center.reshape(3, 1) + t_total).flatten()

        # Match or create new chair ID
        match_id = match_chair(emb, pos_3d, known_chairs, frame_index)
        if match_id is None:
            chair_counter += 1
            match_id = chair_counter
            known_chairs[match_id] = {
                "embeddings": deque([emb], maxlen=MAX_EMB_MEMORY),
                "position": pos_3d,
                "last_seen": frame_index
            }
        else:
            update_chair_record(match_id, emb, pos_3d, frame_index, known_chairs)

        # Draw results
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 200), 2)
        cv2.putText(frame, f"Chair #{match_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

        results_log.append({
            "frame": frame_index,
            "chair_id": match_id,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "pos_x": pos_3d[0], "pos_y": pos_3d[1], "pos_z": pos_3d[2]
        })

    out.write(frame)
    frame_index += 1

cap.release()
out.release()

# Save log
pd.DataFrame(results_log).to_csv("chairs_tracked_3d.csv", index=False)
print(f"✅ Completed {frame_index} frames")
print(f"🎥 Output video saved: {OUTPUT_PATH}")
print(f"📄 Detection log saved: chairs_tracked_3d.csv")
