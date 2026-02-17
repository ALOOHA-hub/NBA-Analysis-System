from utils.video_utils import read_video, save_video
from core.track.tracker import Tracker
from core.detection.detector import Detector
from utils.config_loader import cfg
from core.annotation.annotator import Annotator
from utils.stub_manager import StubManager

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.tracker = Tracker()
        self.detector = Detector(config['model_path'])
        self.stub_manager = StubManager()
        self.annotation_manager = Annotator()

    
    def run(self):
        #1. Read video
        frames = read_video(self.config['input_video_path'])
        #Step 2: Track objects
        stub_path = self.config['stub_path']
        tracks = self.stub_manager.load(stub_path)
        
        # If tracks were loaded, verify they match the number of frames
        if tracks is not None:
            # Check length of one of the track lists (e.g., players)
            if len(tracks.get("players", [])) != len(frames):
                tracks = None

        if tracks is None:
            # Step 1: Detect objects in frames
            detections = self.detector.detect_frames(frames)

            # Step 2: Track objects across frames
            tracks = self.tracker.get_object_tracks(frames, detections)

            # Cache the tracks for next time
            self.stub_manager.save(tracks, stub_path)

        #4. Draw annotation

        output_frames = self.annotation_manager.draw_annotations(frames, tracks)

        #5. Save video
        save_video(output_frames, self.config['output_path'])



