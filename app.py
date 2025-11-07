import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import torch
import torchvision
from torchvision import transforms as T
from ultralytics import YOLO

@dataclass
class TrackedObject:
    """Represents a tracked object in the video"""
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    centroid: Tuple[float, float]
    feature_descriptor: Optional[np.ndarray] = None
    frames_missing: int = 0
    trajectory: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = [self.centroid]

class PanTiltObjectTracker:
    """
    Object tracker that handles pan/tilt camera motion using homography-based stabilization
    """
    
    def __init__(self, 
                 max_missing_frames: int = 30,
                 iou_threshold: float = 0.3,
                 feature_match_threshold: float = 0.7):
        """
        Args:
            max_missing_frames: Max frames an object can be missing before removal
            iou_threshold: IoU threshold for matching detections to tracks
            feature_match_threshold: Threshold for feature-based re-identification
        """
        self.tracked_objects = {}
        self.next_object_id = 0
        self.max_missing_frames = max_missing_frames
        self.iou_threshold = iou_threshold
        self.feature_match_threshold = feature_match_threshold
        
        # For homography estimation
        self.prev_frame_gray = None
        self.feature_detector = cv2.ORB_create(nfeatures=500)
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
    def compute_homography(self, curr_frame_gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute homography matrix between previous and current frame
        """
        if self.prev_frame_gray is None:
            return None
        
        # Detect keypoints and descriptors
        kp1, des1 = self.feature_detector.detectAndCompute(self.prev_frame_gray, None)
        kp2, des2 = self.feature_detector.detectAndCompute(curr_frame_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None
        
        # Match features
        matches = self.feature_matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return None
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Compute homography with RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        return H
    
    def transform_bbox(self, bbox: Tuple[int, int, int, int], H: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Transform bounding box using homography matrix
        """
        x, y, w, h = bbox
        
        # Define bbox corners
        corners = np.float32([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ]).reshape(-1, 1, 2)
        
        # Transform corners
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        # Get new bounding box
        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]
        
        new_x = int(np.min(x_coords))
        new_y = int(np.min(y_coords))
        new_w = int(np.max(x_coords) - new_x)
        new_h = int(np.max(y_coords) - new_y)
        
        return (new_x, new_y, new_w, new_h)
    
    def compute_iou(self, bbox1: Tuple[int, int, int, int], 
                    bbox2: Tuple[int, int, int, int]) -> float:
        """
        Compute Intersection over Union between two bounding boxes
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Compute intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Compute union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def get_bbox_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Get centroid of bounding box
        """
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)
    
    def extract_roi_features(self, frame: np.ndarray, 
                            bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract feature descriptor from ROI for re-identification
        """
        x, y, w, h = bbox
        
        # Ensure bbox is within frame bounds
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        
        if w <= 0 or h <= 0:
            return None
        
        roi = frame[y:y+h, x:x+w]
        
        # Compute color histogram as simple feature
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def match_detections_to_tracks(self, detections: List[Tuple[int, int, int, int]], 
                                   frame: np.ndarray,
                                   H: Optional[np.ndarray]) -> dict:
        """
        Match new detections to existing tracks using IoU and appearance
        """
        if len(self.tracked_objects) == 0:
            return {}
        
        # Transform existing tracks using homography
        transformed_tracks = {}
        if H is not None:
            for obj_id, obj in self.tracked_objects.items():
                try:
                    transformed_bbox = self.transform_bbox(obj.bbox, H)
                    transformed_tracks[obj_id] = transformed_bbox
                except:
                    # If transformation fails, use original bbox
                    transformed_tracks[obj_id] = obj.bbox
        else:
            transformed_tracks = {obj_id: obj.bbox for obj_id, obj in self.tracked_objects.items()}
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(transformed_tracks), len(detections)))
        track_ids = list(transformed_tracks.keys())
        
        for i, track_id in enumerate(track_ids):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self.compute_iou(transformed_tracks[track_id], detection)
        
        # Match using greedy assignment (can be replaced with Hungarian algorithm)
        matches = {}
        matched_detections = set()
        matched_tracks = set()
        
        # Sort by IoU (highest first)
        while True:
            if len(matched_detections) == len(detections) or len(matched_tracks) == len(track_ids):
                break
            
            # Find best match
            max_iou = 0
            best_track_idx = -1
            best_det_idx = -1
            
            for i in range(len(track_ids)):
                if i in matched_tracks:
                    continue
                for j in range(len(detections)):
                    if j in matched_detections:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        best_track_idx = i
                        best_det_idx = j
            
            if max_iou < self.iou_threshold:
                break
            
            matches[track_ids[best_track_idx]] = best_det_idx
            matched_detections.add(best_det_idx)
            matched_tracks.add(best_track_idx)
        
        return matches
    
    def update(self, frame: np.ndarray, 
               detections: List[Tuple[int, int, int, int]]) -> List[TrackedObject]:
        """
        Update tracker with new frame and detections
        
        Args:
            frame: Current video frame (BGR)
            detections: List of detected bounding boxes [(x, y, w, h), ...]
        
        Returns:
            List of currently tracked objects
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute homography for camera motion compensation
        H = self.compute_homography(frame_gray)
        
        # Match detections to existing tracks
        matches = self.match_detections_to_tracks(detections, frame, H)
        
        # Update matched tracks
        matched_detection_indices = set(matches.values())
        for track_id, det_idx in matches.items():
            detection = detections[det_idx]
            obj = self.tracked_objects[track_id]
            
            # Update object
            obj.bbox = detection
            obj.centroid = self.get_bbox_centroid(detection)
            obj.trajectory.append(obj.centroid)
            obj.frames_missing = 0
            obj.feature_descriptor = self.extract_roi_features(frame, detection)
        
        # Increment missing frames for unmatched tracks
        for track_id in self.tracked_objects.keys():
            if track_id not in matches:
                self.tracked_objects[track_id].frames_missing += 1
        
        # Remove tracks that have been missing too long
        to_remove = [tid for tid, obj in self.tracked_objects.items() 
                    if obj.frames_missing > self.max_missing_frames]
        for tid in to_remove:
            del self.tracked_objects[tid]
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                centroid = self.get_bbox_centroid(detection)
                feature_desc = self.extract_roi_features(frame, detection)
                
                new_obj = TrackedObject(
                    id=self.next_object_id,
                    bbox=detection,
                    centroid=centroid,
                    feature_descriptor=feature_desc
                )
                self.tracked_objects[self.next_object_id] = new_obj
                self.next_object_id += 1
        
        # Store current frame for next iteration
        self.prev_frame_gray = frame_gray.copy()
        
        return list(self.tracked_objects.values())
    
    def draw_tracks(self, frame: np.ndarray, 
                   tracked_objects: List[TrackedObject],
                   show_trajectory: bool = True) -> np.ndarray:
        """
        Draw tracked objects on frame
        """
        output = frame.copy()
        
        for obj in tracked_objects:
            x, y, w, h = obj.bbox
            
            # Choose color based on object ID
            color = tuple(int(c) for c in np.random.RandomState(obj.id).randint(0, 255, 3))
            
            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Draw object ID
            label = f"ID: {obj.id}"
            cv2.putText(output, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trajectory
            if show_trajectory and len(obj.trajectory) > 1:
                points = np.array(obj.trajectory, dtype=np.int32)
                cv2.polylines(output, [points], False, color, 2)
        
        return output
    
    def create_trajectory_map(self, frame_shape: Tuple[int, int], 
                            include_inactive: bool = True) -> np.ndarray:
        """
        Create a visualization map showing all object trajectories
        
        Args:
            frame_shape: (height, width) of video frame
            include_inactive: Whether to include objects no longer being tracked
        
        Returns:
            Image showing trajectory map
        """
        h, w = frame_shape
        map_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Draw all trajectories
        for obj_id, obj in self.tracked_objects.items():
            if len(obj.trajectory) < 2:
                continue
            
            color = tuple(int(c) for c in np.random.RandomState(obj.id).randint(50, 255, 3))
            points = np.array(obj.trajectory, dtype=np.int32)
            
            # Draw trajectory line
            cv2.polylines(map_img, [points], False, color, 2)
            
            # Draw start point (green circle)
            cv2.circle(map_img, tuple(points[0]), 5, (0, 255, 0), -1)
            
            # Draw end point (red circle)
            cv2.circle(map_img, tuple(points[-1]), 5, (0, 0, 255), -1)
            
            # Label with object ID at end point
            cv2.putText(map_img, f"ID:{obj.id}", 
                       tuple(points[-1] + np.array([10, 0])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return map_img
    
    def create_heatmap(self, frame_shape: Tuple[int, int], 
                      grid_size: int = 20) -> np.ndarray:
        """
        Create a heatmap showing where objects appear most frequently
        
        Args:
            frame_shape: (height, width) of video frame
            grid_size: Size of grid cells for heatmap
        
        Returns:
            Heatmap image
        """
        h, w = frame_shape
        
        # Create grid
        grid_h = h // grid_size
        grid_w = w // grid_size
        heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
        
        # Count object appearances in each grid cell
        for obj_id, obj in self.tracked_objects.items():
            for cx, cy in obj.trajectory:
                grid_x = int(cx // grid_size)
                grid_y = int(cy // grid_size)
                
                if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
                    heatmap[grid_y, grid_x] += 1
        
        # Normalize and apply colormap
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)
        
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap_colored, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return heatmap_resized
    
    def get_tracking_statistics(self) -> dict:
        """
        Get statistics about tracked objects
        """
        stats = {
            'active_objects': len(self.tracked_objects),
            'total_objects_seen': self.next_object_id,
            'objects_info': []
        }
        
        for obj_id, obj in self.tracked_objects.items():
            obj_info = {
                'id': obj.id,
                'frames_tracked': len(obj.trajectory),
                'frames_missing': obj.frames_missing,
                'current_position': obj.centroid,
                'trajectory_length': len(obj.trajectory)
            }
            stats['objects_info'].append(obj_info)
        
        return stats
    
    def export_tracking_data(self, filename: str = "tracking_data.json"):
        """
        Export tracking data to JSON file
        """
        import json
        
        data = {
            'total_objects': self.next_object_id,
            'objects': {}
        }
        
        for obj_id, obj in self.tracked_objects.items():
            data['objects'][obj_id] = {
                'id': obj.id,
                'trajectory': [{'x': float(x), 'y': float(y)} for x, y in obj.trajectory],
                'frames_tracked': len(obj.trajectory),
                'frames_missing': obj.frames_missing,
                'current_bbox': obj.bbox
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Tracking data exported to {filename}")


class YOLODetector:
    def __init__(self, model_name: str = 'yolov8n.pt', conf_threshold: float = 0.5):
        """
        Initialize YOLO detector with specified model and confidence threshold
        
        Args:
            model_name: YOLO model name or path (e.g., 'yolov8n.pt', 'yolov8s.pt')
            conf_threshold: Minimum confidence threshold for detections
        """
        # Load YOLO model
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # COCO class names (for reference)
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect chairs in the input frame using YOLOv8
        
        Args:
            frame: Input BGR image (numpy array)
            
        Returns:
            List of detections, each as a dictionary with keys:
            - 'bbox': [x, y, w, h] - bounding box coordinates
            - 'confidence': detection confidence score
            - 'class_id': class ID (56 for chair)
            - 'class_name': class name ('chair')
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, classes=[56])  # 56 is the class ID for 'chair'
        
        # Process detections
        detections = []
        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w = x2 - x1
                h = y2 - y1
                
                # Get class and confidence
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                detections.append({
                    'bbox': [x1, y1, w, h],
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)
                })
        
        return detections


def detect_chairs(frame: np.ndarray, detector: YOLODetector) -> List[Tuple[int, int, int, int]]:
    """
    Detect chairs in the frame using YOLO detector
    
    Args:
        frame: Input BGR image
        detector: YOLODetector instance
        
    Returns:
        List of bounding boxes in (x, y, w, h) format
    """
    # Convert frame to RGB if needed
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    # Get detections
    detections = detector.detect(frame_rgb)
    
    # Filter only chairs (class_id 56) and return bboxes
    return [tuple(d['bbox']) for d in detections if d['class_id'] == 56]


# Example usage
if __name__ == "__main__":
    # Open video file or camera
    video_path = "output.mp4"  # Replace with your video path
    cap = cv2.VideoCapture(video_path)
    # For webcam use: cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video")
        exit()
    
    # Initialize YOLO detector for chairs
    print("Loading YOLO model...")
    yolo_detector = YOLODetector(conf_threshold=0.5)
    
    # Initialize tracker
    tracker = PanTiltObjectTracker(
        max_missing_frames=30,
        iou_threshold=0.3
    )
    
    # Initialize background subtractor for simple detection
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=True
    )
    
    frame_count = 0
    frame_shape = None
    
    # For storing map visualizations
    show_map = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_shape is None:
            frame_shape = (frame.shape[0], frame.shape[1])
        
        # Detect chairs using YOLO
        chair_boxes = detect_chairs(frame, yolo_detector)
        
        # Update tracker with chair detections
        tracked_objects = tracker.update(frame, chair_boxes)
        
        # Draw chair detections
        for x, y, w, h in chair_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Chair', (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw results
        output = tracker.draw_tracks(frame, tracked_objects, show_trajectory=True)
        
        # Display info
        cv2.putText(output, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output, f"Tracked Objects: {len(tracked_objects)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output, "Press 'm' for map view, 'h' for heatmap, 's' for stats, 'e' to export",
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        if show_map:
            # Create trajectory map
            trajectory_map = tracker.create_trajectory_map(frame_shape)
            
            # Create heatmap
            heatmap = tracker.create_heatmap(frame_shape)
            
            # Combine views
            combined = np.hstack([output, trajectory_map])
            cv2.imshow("Pan/Tilt Object Tracker", combined)
        else:
            cv2.imshow("Pan/Tilt Object Tracker", output)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('m'):
            # Toggle map view
            show_map = not show_map
            print(f"Map view: {'ON' if show_map else 'OFF'}")
        elif key == ord('h'):
            # Show heatmap
            heatmap = tracker.create_heatmap(frame_shape)
            cv2.imshow("Object Heatmap", heatmap)
            print("Heatmap displayed in separate window")
        elif key == ord('t'):
            # Show trajectory map
            trajectory_map = tracker.create_trajectory_map(frame_shape)
            cv2.imshow("Trajectory Map", trajectory_map)
            print("Trajectory map displayed in separate window")
        elif key == ord('s'):
            # Print statistics
            stats = tracker.get_tracking_statistics()
            print("\n=== Tracking Statistics ===")
            print(f"Active objects: {stats['active_objects']}")
            print(f"Total objects seen: {stats['total_objects_seen']}")
            print("\nPer-object info:")
            for obj_info in stats['objects_info']:
                print(f"  ID {obj_info['id']}: {obj_info['frames_tracked']} frames tracked, "
                      f"missing for {obj_info['frames_missing']} frames")
        elif key == ord('e'):
            # Export tracking data
            tracker.export_tracking_data(f"tracking_data_frame_{frame_count}.json")
        elif key == ord('p'):
            # Save current trajectory map
            trajectory_map = tracker.create_trajectory_map(frame_shape)
            filename = f"trajectory_map_frame_{frame_count}.png"
            cv2.imwrite(filename, trajectory_map)
            print(f"Trajectory map saved to {filename}")
        elif key == ord('a'):
            # Save heatmap
            heatmap = tracker.create_heatmap(frame_shape)
            filename = f"heatmap_frame_{frame_count}.png"
            cv2.imwrite(filename, heatmap)
            print(f"Heatmap saved to {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final tracking statistics
    print(f"\n=== Final Tracking Statistics ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Total unique objects tracked: {tracker.next_object_id}")
    
    # Create and save final trajectory map
    if frame_shape is not None:
        final_trajectory_map = tracker.create_trajectory_map(frame_shape)
        cv2.imwrite("final_trajectory_map.png", final_trajectory_map)
        print("Final trajectory map saved to 'final_trajectory_map.png'")
        
        # Create and save final heatmap
        final_heatmap = tracker.create_heatmap(frame_shape)
        cv2.imwrite("final_heatmap.png", final_heatmap)
        print("Final heatmap saved to 'final_heatmap.png'")
        
        # Export final tracking data
        tracker.export_tracking_data("final_tracking_data.json")
    
    # Show final statistics
    final_stats = tracker.get_tracking_statistics()
    print("\nFinal object tracking summary:")
    for obj_info in final_stats['objects_info']:
        print(f"  Object ID {obj_info['id']}: Tracked for {obj_info['frames_tracked']} frames")
