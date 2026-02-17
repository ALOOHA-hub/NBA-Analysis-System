from utils.video_utils import read_video, save_video
from core.track.tracker import Tracker
from core.detection.detector import Detector
from utils.config_loader import cfg

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.tracker = Tracker()
        self.detector = Detector(config['model_path'])

    
    def run(self):
        #1. Read video
        frames = read_video(self.config['input_video_path'])
        #2. Detection
        detections = self.detector.detect_frames(frames)
        #3. Track objects
        tracks = self.tracker.get_object_tracks(detections)

        #4. Draw annotation
        output_frames = self.tracker.draw_tracks(frames, tracks)

        #5. Save video
        save_video(output_frames, self.config['output_path'])



