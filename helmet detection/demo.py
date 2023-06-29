from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
from PIL import Image

detector = Detector(classes = None) 
detector.load_model(r"weights\best.pt")

# Initialise  class that binds detector and tracker in one class
tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)

# output = None will not save the output video
tracker.track_video(r"4.mp4", output=r"video2.mp4",show_live = True, skip_frames = 0, count_objects = True, verbose=1)
