# BaseLine AI - Computer Vision Tennis Analysis Tool

BaseLine AI is an advanced computer vision toolkit for automated tennis match analysis.

Watch a demo!

[![Watch the video](https://img.youtube.com/vi/yeL9oK3OBRk/default.jpg)](https://www.youtube.com/watch?v=yeL9oK3OBRk)

It leverages deep learning and geometric vision to provide detailed insights from tennis videos including:

- **Player and Ball Tracking:**

  - Uses YOLOv5/YOLOv8 models for robust detection of players and tennis balls in each frame.
  - Multi-object tracking assigns consistent IDs to players and balls throughout the match.
  - Supports cached detection stubs for rapid development and reproducibility.

- **Court Detection and Homography:**

  - Detects court lines and keypoints to compute a homography matrix.
  - Enables accurate mapping between video frames and real-world court coordinates.

- **Mini-Court Overlay Visualization:**

  - Renders a dynamic mini-court overlay showing player and ball positions in real time.
  - Visualizes shot placement, trajectories, and player movement heatmaps.

- **Shot and Speed Analysis:**

  - Computes ball and player speeds, shot impact frames, and shot type distributions.
  - Displays shot speed distributions and placement statistics with intuitive graphics.

- **Training and Customization:**

  - Includes Jupyter notebooks for training custom detection models using Roboflow datasets.
  - Modular design allows easy extension for new features or sports.

- **Output:**
  - Generates annotated output videos with overlays and analysis.
  - Saves results and statistics for further review.

**Project Status:**

- Actively developed, with new features and improvements released weekly.
- Designed for researchers, coaches, and enthusiasts seeking automated tennis analytics.

## Features

- YOLO-based player/ball detection (`yolov5/8`)
- Court line & keypoint detection → homography
- Player/ball trackers with cached stubs for rapid dev
- Mini-court overlay visualization
- Notebooks for training (Roboflow datasets)

## Acknowledgments

- Inspired by Code In a Jiffy's [www.youtube.com/@codeinajiffy] tutorial [https://youtu.be/L23oIHZE14w?si=rI1_3GC2W1LVLCWD]
- Learned training workflows from Code in a Jiffy then independently designed a full analysis pipeline.
- Engineered ball-tracking logic to calculate real-time gameplay stats and generate more detailed summaries.
- Implemented homography to correct pixel-to-meter distortion, significantly improving speed accuracy compared to the demo.
