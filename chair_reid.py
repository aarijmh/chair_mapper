import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPATIAL_WEIGHT = 0.35
BASE_THRESHOLD = 0.78
MIN_SPATIAL_DIST = 2.0
INACTIVITY_LIMIT = 200
MAX_EMB_MEMORY = 5

# ============================================================
# Load Models
# ============================================================

print("[INFO] Loading models...")
yolo = YOLO("yolov8s.pt")  # detects chairs among many other objects
clip_model, preprocess = torch.hub.load("openai/CLIP", "clip", device=DEVICE)
clip_model.eval()
print("[INFO] Models loaded.")


# ============================================================
# Utility Functions
# ============================================================

def cosine_similarity(a, b):
    return 1 - cosine(a, b)

def get_average_embedding(embeddings):
    return np.mean(np.stack(embeddings), axis=0)

def combined_score(emb_new, emb_old, pos_new, pos_old):
    appearance_score = cosine_similarity(emb_new, emb_old)
    if pos_new is None or pos_old is None:
        return appearance_score
    spatial_dist = np.linalg.norm(pos_new - pos_old)
    if spatial_dist > MIN_SPATIAL_DIST:
        return 0
    spatial_score = max(0, 1 - spatial_dist / MIN_SPATIAL_DIST)
    return (1 - SPATIAL_WEIGHT) * appearance_score + SPATIAL_WEIGHT * spatial_score

def extract_clip_embedding(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        image_input = preprocess(img).unsqueeze(0).to(DEVICE)
        emb = clip_model.encode_image(image_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().cpu().numpy()

def estimate_camera_motion(prev_gray, gray):
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(gray, None)
    if des1 is None or des2 is None:
        return np.eye(3), np.zeros((3, 1))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return np.eye(3), np.zeros((3, 1))
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    E, _ = cv2.findEssentialMat(pts1, pts2, focal=800, pp=(320, 240), method=cv2.RANSAC)
    if E is None:
        return np.eye(3), np.zeros((3, 1))
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2)
    return R, t

def get_3d_position(bbox, t_total):
    x1, y1, x2, y2 = map(int, bbox)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    z = max(0.5, min(3.0, 2.5 - (y2 - y1) / 200.0))
    pos_cam = np.array([cx / 100.0, cy / 100.0, z])
    pos_world = pos_cam + t_total.flatten()
    return pos_world

# ============================================================
# Re-Identification Logic
# ============================================================

def match_chair(emb, pos, known_chairs, frame_index):
    best_id, best_score = None, 0
    for cid, data in known_chairs.items():
        if frame_index - data["last_seen"] > INACTIVITY_LIMIT:
            continue
        avg_emb = get_average_embedding(data["embeddings"])
        score = combined_score(emb, avg_emb, pos, data["position"])
        if score > best_score:
            best_score, best_id = score, cid
    dynamic_thresh = BASE_THRESHOLD - 0.02 * np.log1p(len(known_chairs))
    if best_score > dynamic_thresh:
        return best_id
    return None

def update_chair_record(cid, emb, pos, known_chairs, frame_index):
    data = known_chairs[cid]
    data["embeddings"].append(emb)
    if len(data["embeddings"]) > MAX_EMB_MEMORY:
        data["embeddings"].popleft()
    data["position"] = 0.8 * data["position"] + 0.2 * pos
    data["last_seen"] = frame_index

# ============================================================
# Main Video Loop
# ============================================================

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    known_chairs = {}
    frame_index = 0
    R_total, t_total = np.eye(3), np.zeros((3, 1))
    prev_gray = None
    next_id = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Update camera motion
        if prev_gray is not None:
            R, t = estimate_camera_motion(prev_gray, gray)
            t_total += R_total @ t
            R_total = R @ R_total
        prev_gray = gray

        # Chair detection
        results = yolo(frame)
        detections = [box for box in results[0].boxes.data.cpu().numpy()
                      if int(box[5]) == 56]  # class 56 = chair

        for box in detections:
            x1, y1, x2, y2, conf, cls = box
            emb = extract_clip_embedding(frame, (x1, y1, x2, y2))
            if emb is None:
                continue
            pos_3d = get_3d_position((x1, y1, x2, y2), t_total)

            match_id = match_chair(emb, pos_3d, known_chairs, frame_index)
            if match_id is None:
                cid = next_id
                next_id += 1
                known_chairs[cid] = {
                    "embeddings": deque([emb], maxlen=MAX_EMB_MEMORY),
                    "position": pos_3d,
                    "last_seen": frame_index
                }
            else:
                cid = match_id
                update_chair_record(cid, emb, pos_3d, known_chairs, frame_index)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Chair {cid}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Chair Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Total unique chairs: {len(known_chairs)}")

# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    video_path = "output.mp4"  # <-- replace with your video path
    process_video(video_path)
