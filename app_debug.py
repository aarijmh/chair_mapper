#!/usr/bin/env python3
"""
scene_reconstruct_v3/app_debug.py - DIAGNOSTIC VERSION

This version adds extensive debugging output to identify where reconstruction fails.
"""

import argparse, os, sys, csv, math
import cv2, numpy as np
import open3d as o3d

def extract_frames_and_colors(video_path, step=5, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[DEBUG] Video properties:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Will sample every {step} frames")
    
    frames = []
    colors = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            colors.append(frame.copy())
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    print(f"[INFO] Extracted {len(frames)} frames (step={step})")
    return frames, colors

def make_detector(method="sift"):
    if method == "sift":
        return cv2.SIFT_create()
    else:
        return cv2.ORB_create(8000)

def match_features(detector, img1, img2):
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    
    print(f"[DEBUG] Feature detection:")
    print(f"  - Frame 1: {len(kp1) if kp1 else 0} keypoints")
    print(f"  - Frame 2: {len(kp2) if kp2 else 0} keypoints")
    
    if des1 is None or des2 is None:
        print(f"[DEBUG] No descriptors computed!")
        return [], [], np.empty((0,2)), np.empty((0,2))
    
    if isinstance(detector, cv2.ORB):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    print(f"[DEBUG] Matching: {len(matches)} total, {len(good)} good matches (ratio test)")
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]) if len(good)>0 else np.empty((0,2))
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]) if len(good)>0 else np.empty((0,2))
    return kp1, kp2, pts1, pts2

def recover_pose_and_triangulate(pts1, pts2, K):
    print(f"[DEBUG] Pose recovery with {pts1.shape[0]} point correspondences")
    
    if pts1.shape[0] < 8:
        print(f"[DEBUG] FAILED: Need at least 8 points, got {pts1.shape[0]}")
        return None, None, None
    
    E, maskE = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        print(f"[DEBUG] FAILED: Could not compute essential matrix")
        return None, None, None
    
    inliers = np.sum(maskE)
    print(f"[DEBUG] Essential matrix: {inliers} inliers from {pts1.shape[0]} points")
    
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K)
    
    proj1 = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    proj2 = K.dot(np.hstack((R, t)))
    pts1_h = pts1[mask_pose.ravel()==1].T
    pts2_h = pts2[mask_pose.ravel()==1].T
    
    if pts1_h.shape[1] < 2:
        print(f"[DEBUG] FAILED: Only {pts1_h.shape[1]} points after pose recovery")
        return R, t, np.empty((0,3))
    
    pts4d = cv2.triangulatePoints(proj1, proj2, pts1_h, pts2_h)
    pts3d = (pts4d[:3] / pts4d[3]).T
    
    # Filter reasonable depths
    valid = pts3d[:,2] > 0.05
    pts3d_filtered = pts3d[valid]
    
    print(f"[DEBUG] Triangulation: {pts3d.shape[0]} points, {pts3d_filtered.shape[0]} with valid depth (Z>0.05)")
    
    return R, t, pts3d_filtered

def run_frame_keyframe_tri(frames, K, method="sift", keyframe_gap=5):
    detector = make_detector(method)
    keyframes_idx = [0]
    keyframes = [frames[0]]
    poses = [(np.eye(3), np.zeros((3,1)))]
    all_pts = []
    
    print(f"\n[INFO] Starting reconstruction with {len(frames)} frames")
    print(f"[INFO] Camera matrix K:\n{K}\n")
    
    successful_pairs = 0
    
    for i in range(1, len(frames)):
        print(f"\n--- Processing frame {i}/{len(frames)-1} ---")
        kp1, kp2, pts1, pts2 = match_features(detector, keyframes[-1], frames[i])
        
        if pts1.shape[0] < 8:
            print(f"[DEBUG] Skipping: insufficient matches ({pts1.shape[0]} < 8)")
            continue
        
        R, t, pts3d = recover_pose_and_triangulate(pts1, pts2, K)
        
        if pts3d is None or pts3d.shape[0] == 0:
            print(f"[DEBUG] Skipping: triangulation failed")
            continue
        
        successful_pairs += 1
        all_pts.append(pts3d)
        print(f"[DEBUG] ✓ SUCCESS: Added {pts3d.shape[0]} 3D points (total pairs: {successful_pairs})")
        
        if i - keyframes_idx[-1] >= keyframe_gap:
            keyframes_idx.append(i)
            keyframes.append(frames[i])
            poses.append((R, t))
            print(f"[DEBUG] Added keyframe {len(keyframes)-1}")
    
    print(f"\n[SUMMARY] Successfully processed {successful_pairs} frame pairs")
    
    if len(all_pts)==0:
        print("[ERROR] No 3D points reconstructed from any frame pair!")
        print("\nPossible issues:")
        print("  1. Video has insufficient texture/features")
        print("  2. Camera parameters (focal length, principal point) are incorrect")
        print("  3. Frames are too similar (no camera motion)")
        print("  4. Try: --step 1, different --focal values, or --method orb")
        raise RuntimeError("No 3D points reconstructed.")
    
    pts_all = np.vstack(all_pts)
    print(f"[INFO] Total reconstructed points: {pts_all.shape[0]}")
    return pts_all, poses

def run_midas_on_colors(color_frames, device="cpu"):
    try:
        import torch
    except Exception as e:
        raise RuntimeError("Torch not installed; MiDaS unavailable.") from e
    print("[INFO] Loading MiDaS model (weights may download).")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas.to(device).eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    depths = []
    with torch.no_grad():
        for img in color_frames:
            inp = transform(img).to(device)
            sample = torch.unsqueeze(inp, 0)
            prediction = midas(sample)
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False).squeeze()
            depths.append(prediction.cpu().numpy())
    print(f"[INFO] Computed MiDaS depths for {len(depths)} frames")
    return depths

def load_yolo(model_name="yolov8n.pt"):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Please install ultralytics (pip install ultralytics)") from e
    model = YOLO(model_name)
    return model

def detect_chairs_yolo(model, frame_bgr, conf_thresh=0.25):
    res = model(frame_bgr)
    boxes = []
    if len(res) == 0:
        return boxes
    r = res[0]
    if not hasattr(r, 'boxes'):
        return boxes
    for box in r.boxes:
        cls = int(box.cls[0].item()) if hasattr(box.cls, 'shape') or hasattr(box.cls, 'item') else int(box.cls)
        name = model.names[cls].lower() if hasattr(model, 'names') else str(cls)
        if name != "chair":
            continue
        conf = float(box.conf[0]) if hasattr(box.conf, 'shape') else float(box.conf)
        if conf < conf_thresh:
            continue
        xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else np.array(box.xyxy[0])
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        boxes.append(((x1,y1,x2,y2), conf))
    return boxes

def project_points_to_image(pts3d, K):
    pts_cam = pts3d.copy()
    proj = (K.dot(pts_cam.T)).T
    pts2d = proj[:, :2] / proj[:, 2:3]
    return pts2d

def map_bbox_center_to_depth_midas(center, depth_map):
    h, w = depth_map.shape
    cx, cy = int(center[0]), int(center[1])
    x1 = max(0, cx-5); x2 = min(w-1, cx+5)
    y1 = max(0, cy-5); y2 = min(h-1, cy+5)
    patch = depth_map[y1:y2+1, x1:x2+1]
    if patch.size == 0:
        return None
    z = float(np.median(patch))
    return z

def map_bbox_center_to_depth_from_sparse(center, pts3d, K, reproj_threshold=10.0):
    pts2d = project_points_to_image(pts3d, K)
    cx, cy = center
    dists = np.linalg.norm(pts2d - np.array([cx,cy])[None,:], axis=1)
    idx = np.where(dists < reproj_threshold)[0]
    if idx.size == 0:
        idx = np.argsort(dists)[:5]
    chosen = pts3d[idx]
    if chosen.size == 0:
        return None
    z = float(np.median(chosen[:,2]))
    return z

def run_detection_and_mapping(color_frames, pts3d, K, save_video=False, video_out_path="output_with_detections.mp4", use_midas=False):
    from copy import deepcopy
    model = load_yolo("yolov8n.pt")
    chairs_3d = []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    depths = None
    if use_midas:
        try:
            depths = run_midas_on_colors(color_frames)
        except Exception as e:
            print("[WARN] MiDaS failed, falling back to sparse mapping:", e)
            depths = None
    if save_video:
        H, W = color_frames[0].shape[:2]
        writer = cv2.VideoWriter(video_out_path, fourcc, 10.0, (W,H))
    
    for i, frame in enumerate(color_frames):
        display = frame.copy()
        boxes = detect_chairs_yolo(model, frame)
        for j, (bbox, conf) in enumerate(boxes):
            x1,y1,x2,y2 = bbox
            cx = int((x1+x2)/2); cy = int((y1+y2)/2)
            depth = None
            if depths is not None:
                depth = map_bbox_center_to_depth_midas((cx,cy), depths[i])
            if depth is None:
                depth = map_bbox_center_to_depth_from_sparse((cx,cy), pts3d, K)
            if depth is None or not np.isfinite(depth):
                continue
            X = (cx - K[0,2]) * depth / K[0,0]
            Y = (cy - K[1,2]) * depth / K[1,1]
            Z = depth
            chairs_3d.append({"frame_idx": i, "bbox": bbox, "conf": conf, "X": float(X), "Y": float(Y), "Z": float(Z)})
            cv2.rectangle(display, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(display, f"Chair {j+1} {conf:.2f} Z={Z:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if save_video and writer is not None:
            writer.write(display)
    if writer is not None:
        writer.release()
    return chairs_3d

def save_chairs_csv(chairs, out_path="chairs_3d.csv"):
    keys = ["frame_idx","bbox","conf","X","Y","Z"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for c in chairs:
            w.writerow([c.get(k) for k in keys])
    print(f"[INFO] Saved {len(chairs)} chair entries to {out_path}")

def visualize_with_chairs(pts3d, chairs):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    geom = [pcd]
    if len(chairs) > 0:
        positions = np.array([[c["X"], c["Y"], c["Z"]] for c in chairs])
        uniq = {}
        for pos in positions:
            key = tuple((pos/0.2).round().astype(int))
            uniq[key] = pos
        for k,pos in uniq.items():
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            mesh.translate(pos)
            mesh.paint_uniform_color([1.0,0.0,0.0])
            geom.append(mesh)
    o3d.visualization.draw_geometries(geom)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v", required=True)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--method", choices=["sift","orb"], default="sift")
    parser.add_argument("--focal", type=float, default=700.0)
    parser.add_argument("--pp", nargs=2, type=float, default=[640.0,360.0])
    parser.add_argument("--keyframe-gap", type=int, default=5)
    parser.add_argument("--midas", action="store_true")
    parser.add_argument("--detect_chairs", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--out", type=str, default="reconstruction_v3.ply")
    args = parser.parse_args()

    frames, colors = extract_frames_and_colors(args.video, step=args.step)
    K = np.array([[args.focal,0,args.pp[0]],[0,args.focal,args.pp[1]],[0,0,1.0]], dtype=float)

    pts3d, poses = run_frame_keyframe_tri(frames, K, method=args.method, keyframe_gap=args.keyframe_gap)
    print(f"[INFO] Reconstructed {pts3d.shape[0]} points")
    o3d.io.write_point_cloud(args.out, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts3d)), write_ascii=True)
    print(f"[INFO] Wrote point cloud to {args.out}")

    chairs = []
    if args.detect_chairs:
        chairs = run_detection_and_mapping(colors, pts3d, K, save_video=args.save_video, video_out_path="output_with_detections.mp4", use_midas=args.midas)
        save_chairs_csv(chairs, out_path="chairs_3d.csv")
    
    visualize_with_chairs(pts3d, chairs)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()