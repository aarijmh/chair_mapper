# Chair Re-Identification and 3D Tracking

A Python script for detecting, tracking, and re-identifying chairs in videos using YOLOv8 for detection, CLIP for appearance embeddings, and ORB feature matching for camera motion estimation.

## Features

- 🪑 Real-time chair detection using YOLOv8
- 🆔 Persistent chair identification across frames
- 📊 3D position tracking with camera motion compensation
- 🔄 Robust re-identification of chairs after temporary occlusions
- 📊 CSV export of tracked chair positions and timestamps
- 🎥 Video output with bounding boxes and chair IDs

## Dependencies

```bash
pip install ultralytics torch torchvision transformers opencv-python pillow tqdm pandas
```

## Usage

```bash
python chair_reid.py
```

### Configuration
Edit the following parameters in `chair_reid.py` as needed:
- `VIDEO_PATH`: Input video file path
- `OUTPUT_PATH`: Output video file path
- `FRAME_SKIP`: Process every N frames (increase for better performance)
- `SPATIAL_WEIGHT`: Weight for spatial vs appearance matching (0-1)
- `BASE_THRESHOLD`: Similarity threshold for re-identification
- `INACTIVITY_LIMIT`: Frames before considering a chair lost
- `MAX_EMB_MEMORY`: Number of embeddings to store per chair

## Outputs

- `output_tracked_3d.mp4`: Video with detected and tracked chairs
- `chairs_tracked_3d.csv`: CSV log of all detections with:
  - Frame number
  - Chair ID
  - Bounding box coordinates (x1, y1, x2, y2)
  - 3D position (x, y, z)

## How It Works

1. **Detection**: YOLOv8 detects chairs in each frame
2. **Feature Extraction**: CLIP generates appearance embeddings for each chair
3. **Camera Motion**: ORB features track camera movement between frames
4. **3D Positioning**: Chairs are positioned in 3D space relative to camera
5. **Matching**: New detections are matched to known chairs using appearance and spatial information
6. **Tracking**: Chair positions and appearances are updated over time

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended for better performance
- Internet connection required for first-time model downloads

## Notes

- First run will download YOLOv8 and CLIP models (saved locally for future use)
- For better performance on long videos, increase `FRAME_SKIP`
- Adjust `BASE_THRESHOLD` and `SPATIAL_WEIGHT` based on your tracking needs
