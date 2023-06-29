


from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
from PIL import Image



detector = Detector(classes = None) 
detector.load_model(r"last.pt")


# # RUN
tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)


tracker.track_video(r"11.mp4", output=r"video2.mp4",show_live = True, skip_frames = 0, count_objects = True, verbose=1)
