# NBA System Analysis

A modular computer vision pipeline for NBA video analysis, featuring player detection, tracking, team assignment, and automated statistics annotation.

## 🚀 Project Overview

This project implements a sophisticated video analysis pipeline designed to process NBA game footage. It uses state-of-the-art deep learning models to identify players and referees, track their movement across frames, and classify them into respective teams based on jersey visual analysis.

## 🏗️ Folder Structure

```text
NBA System Analysis/
├── config.yaml             # Central configuration for paths and thresholds
├── main.py                # Main entry point to run the pipeline
├── constants/             # Centralized constant values for the project
│   ├── __init__.py        # Exports all constants for clean imports
│   ├── tracker_consts.py  # Configuration for ByteTrack and detection
│   └── team_assigner_consts.py # Team colors and model settings
├── core/                  # Core logic of the system
│   ├── pipeline.py        # Connects all modules (The "Brain")
│   ├── detection/         # YOLO inference logic
│   ├── track/             # ByteTrack object tracking implementation
│   ├── team_assignement/  # Visual jersey analysis using Fashion-CLIP
│   └── annotation/        # Modular drawing system (Entity & Stats)
├── models/                # Local storage for weights and config files
├── data/                  # Input and output video storage
│   ├── input/
│   └── output/
├── utils/                 # Shared helper functions (Video I/O, Bbox math)
└── stubs/                 # Cached tracking/team data for faster execution
```

## 🛠️ Step-by-Step Setup

1.  **Environment Setup**:
    Ensure you have Python 3.10+ installed. Create and activate a virtual environment:
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    Install the required packages. Note: NumPy must be `< 2.0.0` for compatibility with existing CV libraries.
    ```powershell
    pip install -r requirements.txt
    ```

3.  **Models Configuration**:
    -   Place your trained YOLOv8 model in `models/yolov8x_traind.pt`.
    -   The Fashion-CLIP model should be organized in `models/fashion-clip/` containing the `config.json` and `pytorch_model.bin` files.

4.  **Prepare Input**:
    Place your NBA video file in `data/input/video_1.mp4`.

## 🏃 How to Run

Simply execute the main script:
```powershell
python main.py
```

### What happens under the hood:
1.  **Reading Video**: Loads frames from the input path.
2.  **Detection & Tracking**: Runs YOLOv8 followed by ByteTrack. Results are cached in `stubs/track_stubs.pkl` so the heavy work only happens once.
3.  **Team Assignment**: Analyzes player jerseys using the CLIP model. Results are cached in `stubs/team_stubs.pkl`.
4.  **Annotation**: Draws bounding boxes, IDs, and possession triangles on the frames.
5.  **Saving Output**: Compiles the processed frames into `data/output/output.avi`.

## ⚙️ Configuration

You can tweak the system behavior by editing `config.yaml`:
-   `confidence_threshold`: Sensitivity of detections.
-   `team_model_path`: Path to your local CLIP model.
-   `input_video_path`: Change this to process different videos.
