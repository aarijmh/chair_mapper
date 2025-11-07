
Scene Reconstructor v4 (full)

This package contains a robust reconstruction pipeline that:
- attempts SfM sparse reconstruction first
- automatically falls back to MiDaS depth-based reconstruction if SfM fails
- runs YOLOv8 chair detection (Ultralytics) and maps detections to 3D
- outputs point cloud (reconstruction_v4.ply), chairs_3d.csv, and output_with_detections.mp4 (if --save-video)

Usage:
1. Unzip and enter folder:
   unzip scene_reconstruct_v4_full.zip
   cd scene_reconstruct_v4_full

2. Create venv and install requirements:
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   pip install -r requirements.txt

3. Run:
   python app.py --video /path/to/video.mp4 --step 1 --method sift --detect_chairs --midas --save-video

Notes:
- ultralytics will download yolov8n.pt on first run (internet required).
- MiDaS (torch) will download model weights when first used (internet required).
- For large videos, increase --step to sample fewer frames, or provide --max-frames (not implemented here) and crop video as needed.
