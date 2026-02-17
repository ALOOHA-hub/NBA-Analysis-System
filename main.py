from utils.config_loader import cfg
from core.detection.detector import Detector
from utils.video_utils import save_video, read_video
from ultralytics import YOLO
import os

def main():
    video_path = cfg['settings']['input_video_path']
    save_path = os.path.join(cfg['settings']['output_path'], 'output_video.mp4')
    model_path = cfg['settings']['model_path']
    conf_thres = cfg['settings']['confidence_threshold']
    iou_thres = cfg['settings']['iou_threshold']

    model = YOLO(model_path)
    results = model.track(video_path, save=True, project=save_path, name="inference", exist_ok=True, conf=conf_thres, iou=iou_thres,output=save_path)

if __name__ == "__main__":
    main()
