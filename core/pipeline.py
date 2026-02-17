from utils.video_utils import read_video, save_video
from core.track.tracker import Tracker
from core.detection.detector import Detector
from utils.config_loader import cfg
from core.annotation.annotator import Annotator
from utils.stub_manager import StubManager
from core.team_assignement.team_assigner import TeamAssigner

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.tracker = Tracker()
        self.detector = Detector(config['model_path'])
        self.stub_manager = StubManager()
        self.annotation_manager = Annotator()
        self.team_assigner = TeamAssigner(model_path=config.get('team_model_path'))

    
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

        # Step 3: Assign Teams
        team_stub_path = self.config.get('team_stub_path', 'stubs/team_stubs.pkl')
        team_assignments = self.team_assigner.get_player_teams_across_frames(frames, tracks['players'], stub_path=team_stub_path)

        # Merge team info into tracks
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assignments[frame_num].get(player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                # Default colors: Team 1 (Red-ish), Team 2 (Blue-ish)
                # You can customize these in constants/annotation_consts.py later
                tracks['players'][frame_num][player_id]['team_color'] = (0, 0, 255) if team == 1 else (255, 0, 0)

        #4. Draw annotation

        output_frames = self.annotation_manager.draw_annotations(frames, tracks)

        #5. Save video
        save_video(output_frames, self.config['output_path'])



