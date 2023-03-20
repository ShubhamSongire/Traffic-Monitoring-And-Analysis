#!/usr/bin/env python
# coding: utf-8

# https://github.com/deshwalmahesh/yolov7-deepsort-tracking

# ## Imports
# We have 3 important files for this purpose and each and every dependency, class, import, function, variable etc is being imported from these modules
# 
# 1. `detection_helpers` which I made to wrap the original `YOLOv-7` code along with helper functions
# 2. `tracking_helpers` has modular code which is used to wrap the `DeepSORT` repo and workings
# 3. `bridge_wrapper` acts as a bridge to bind **ANY** detection model with `DeepSORT`

# # RUN

# In[1]:


from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
from PIL import Image


# # Detection
# Detect objects using `Yolov-7`

# # Tracking
# 
# Works as follows:
# 1. Read each frame of video using `OpenCV`
# 2. Get Bounding Box or Detections from the model per frame
# 3. Crop those patches and pass on to `reID` model for re identification which is a part of `DeepSORT` method
# 4. Get the above embeddings and then use `Kalman Filtering` and `Hungerian assignment` to assign the correct BB to the respective object
# 5. Show, Save the frames

# In[11]:


detector = Detector(classes = None) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
# detector.load_model('./weights/yolov7x.pt',) # pass the path to the trained weight file
detector.load_model(r"D:\Freelancing\Vehicle YOLO\vehicle classification\yolov7\yolov7-main\yolov7-main\runs\train\yolov7-plate2\weights\last.pt",)


# # RUN

# In[6]:


# Initialise  class that binds detector and tracker in one class
tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)

# output = None will not save the output video
# tracker.track_video("0", output="E:\downloads\video.mp4",show_live = True, skip_frames = 0, count_objects = True, verbose=1)
# tracker.track_video(r"D:\Freelancing\Vehicle YOLO\results\test videos 2\0.mp4", output=r"E:\downloads\video2.mp4",show_live = True, skip_frames = 0, count_objects = True, verbose=1)
# Working
tracker.track_video(r"D:\Freelancing\Vehicle YOLO\results\test videos 0\11.mp4", output=r"E:\downloads\video2.mp4",show_live = True, skip_frames = 0, count_objects = True, verbose=1)

# tracker.track_video(r"rtsp://root:lediscet@123@182.71.126.154:4554/live1s3.sdp", output=r"E:\downloads\video2.mp4",show_live = True, skip_frames = 0, count_objects = True, verbose=1)
# tracker.track_video(r"rtsp://root:lediscet@123@182.71.126.154:4554/live1s3.sdp", output=r"E:\downloads\video2.mp4",show_live = True, skip_frames = 0, count_objects = True, verbose=1)
# tracker.track_video(r"rtsp://adminttpl:Tech@ttpl159@182.71.126.154:3554/LiveH264_1", output=r"E:\downloads\video2.mp4",show_live = True, skip_frames = 0, count_objects = True, verbose=1)