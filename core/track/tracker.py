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

    def get_object_tracks(self, frames, detections):
        """
        Get player tracking results for a sequence of frames.

        Args:
            frames (list): List of video frames to process.
            detections (list): List of YOLO detections.

        Returns:
            list: List of dictionaries containing player tracking information for each frame.
        """
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv[CLASS_PLAYER]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                
                if cls_id == cls_names_inv[CLASS_REFEREE]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv[CLASS_BALL]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Interpolate Ball Positions
        tracks["ball"] = self.interpolate_ball_positions(tracks["ball"])

        return tracks    

    def interpolate_ball_positions(self,ball_positions):
        """
        Interpolate missing ball positions to create smooth tracking results.

        Args:
            ball_positions (list): List of ball positions with potential gaps.

        Returns:
            list: List of ball positions with interpolated values filling the gaps.
        """
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
        
    def draw_tracks(self, video_frames, tracks):
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