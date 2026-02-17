import supervision as sv
import pickle
import os
import numpy as np
import cv2
import sys 
from utils.bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position
from ultralytics import YOLO
from constants import (
    CLASS_PLAYER, 
    CLASS_REFEREE, 
    CLASS_BALL, 
    CLASS_GOALKEEPER,
    TRACKER_ACTIVATION_THRESHOLD,
    TRACKER_LOST_BUFFER
)
import pandas as pd

class Tracker:
    def __init__(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=TRACKER_ACTIVATION_THRESHOLD, 
            lost_track_buffer=TRACKER_LOST_BUFFER
        )

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None, detections=None):

        """
        Get player tracking results for a sequence of frames with optional caching.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.
        """
        tracks = read_stub(read_from_stub,stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        tracks=[]

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks.append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['Player']:
                    tracks[frame_num][track_id] = {"bbox":bbox}
        
        save_stub(stub_path,tracks)
        return tracks    
    def interpolate_ball_positions(self, ball_positions):
        # 1. Convert to DataFrame with NaNs for missing frames
        processed_positions = []
        for x in ball_positions:
            bbox = x.get(1, {}).get('bbox', [])
            if not bbox:
                processed_positions.append([np.nan, np.nan, np.nan, np.nan])
            else:
                processed_positions.append(bbox)
        
        df_ball_positions = pd.DataFrame(processed_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # 2. Filter False Positives (Sudden "Teleportation")        # If the ball moves > 100 pixels in 1 frame, it's likely a false detection (e.g., a shoe)
        df_ball_positions['center_x'] = (df_ball_positions['x1'] + df_ball_positions['x2']) / 2
        df_ball_positions['center_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        
        # Calculate displacement from previous valid frame
        df_ball_positions['dist'] = np.sqrt(
            df_ball_positions['center_x'].diff()**2 + df_ball_positions['center_y'].diff()**2
        )
        
        # Mark detections with impossible speed as NaN (Threshold: 100px per frame)
        # You can adjust this threshold based on your video resolution
        MAX_PIXEL_MOVE_PER_FRAME = 100 
        outliers = df_ball_positions['dist'] > MAX_PIXEL_MOVE_PER_FRAME
        df_ball_positions.loc[outliers, ['x1', 'y1', 'x2', 'y2']] = np.nan

        # 3. Interpolate with a Limit
        # limit=20 means we only fill gaps of up to 20 frames (~0.6 seconds at 30fps).
        # Larger gaps are left empty because the ball is likely out of play.
        df_ball_positions = df_ball_positions.interpolate(method='linear', limit=20, limit_direction='both')

        # 4. Smooth the path (Rolling Average)
        # This removes the "shaking" effect of bounding boxes
        df_ball_positions[['x1', 'y1', 'x2', 'y2']] = (
            df_ball_positions[['x1', 'y1', 'x2', 'y2']]
            .rolling(window=5, min_periods=1, center=True)
            .mean()
        )

        # 5. Fill remaining NaNs (edges) cautiously
        # bfill only if we are confident, otherwise leave as is to avoid drawing ball at (0,0) or corners
        df_ball_positions = df_ball_positions.bfill(limit=5)

        # Convert back to original format
        # If a row is still NaN (because gap was too large), we return an empty dict {} 
        final_positions = []
        for row in df_ball_positions.to_numpy():
            if np.isnan(row[0]):
                final_positions.append({}) # No ball in this frame
            else:
                final_positions.append({1: {"bbox": row[:4].tolist()}})

        return final_positions 
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = (0, 0, 255) # Red for players
                x1, y1, x2, y2 = player["bbox"]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"Player {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw Referees
            for track_id, referee in referee_dict.items():
                color = (0, 255, 255) # Yellow for referees
                x1, y1, x2, y2 = referee["bbox"]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"Ref {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw Ball
            for track_id, ball in ball_dict.items():
                color = (0, 255, 0) # Green for ball
                x1, y1, x2, y2 = ball["bbox"]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
            output_video_frames.append(frame)

        return output_video_frames