


from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
from PIL import Image



detector = Detector(classes = None) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
 # pass the path to the trained weight file
detector.load_model(r"last.pt")


# # RUN
tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)


tracker.track_video(r"D:\Freelancing\Vehicle YOLO\results\test videos 0\11.mp4", output=r"E:\downloads\video2.mp4",show_live = True, skip_frames = 0, count_objects = True, verbose=1)
