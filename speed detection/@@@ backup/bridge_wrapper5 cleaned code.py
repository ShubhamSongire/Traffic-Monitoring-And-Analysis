'''
A Moduele which binds Yolov7 repo with Deepsort with modifications
'''

import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # comment out below line to enable tensorflow logging outputs
import time, datetime
from collections import Counter
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto # DeepSORT official implementation uses tf1.x so we have to do some modifications to avoid errors

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# import from helpers
from tracking_helpers import read_class_names, create_box_encoder
from detection_helpers import *


 # load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True



class YOLOv7_DeepSORT:
    '''
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    '''
    def __init__(self, reID_model_path:str, detector, max_cosine_distance:float=0.4, nn_budget:float=None, nms_max_overlap:float=1.0,
    coco_names_path:str ="./io_data/input/classes/coco.names",  ):
        '''
        args: 
            reID_model_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
            coco_file_path: File wich contains the path to coco naames
        '''
        self.detector = detector
        self.coco_names_path = coco_names_path
        self.nms_max_overlap = nms_max_overlap
        self.class_names = read_class_names()

        # initialize deep sort
        self.encoder = create_box_encoder(reID_model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # calculate cosine distance metric
        self.tracker = Tracker(metric) # initialize tracker


    def track_video(self,video:str, output:str, skip_frames:int=0, show_live:bool=False, count_objects:bool=False, verbose:int = 0):
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            output: path to output video
            skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            count_objects: count objects being tracked on screen
            verbose: print details on the screen allowed values 0,1,2
        '''
        try: # begin video capture
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        out = None
        if output: # get video ready to save locally if flag is set
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        frame_num = 0
        # Tracking vehicle roundabout 
        vehicles_entering = {}    ##############################
        vehicles_elapsed_time = {}    ##############################
        while True: # while video is running
            return_value, frame = vid.read()
            if not return_value:
                print('Video has ended or failed!')
                break
            frame_num +=1

            if skip_frames and not frame_num % skip_frames: continue # skip every nth frame. When every frame is not important, you can use this to fasten the process
            if verbose >= 1:start_time = time.time()

            # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
            yolo_dets = self.detector.detect(frame.copy(), plot_bb = False)  # Get the detections
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if yolo_dets is None:
                bboxes = []
                scores = []
                classes = []
                num_objects = 0
            
            else:
                bboxes = yolo_dets[:,:4]
                bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
                bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

                scores = yolo_dets[:,4]
                classes = yolo_dets[:,-1]
                num_objects = bboxes.shape[0]
            # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
            
            names = []
            for i in range(num_objects): # loop through objects and use class index to get class name
                class_indx = int(classes[i])
                class_name = self.class_names[class_indx]
                names.append(class_name)

            names = np.array(names)
            
            count = len(names)

            if count_objects:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)

            # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
            features = self.encoder(frame, bboxes) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

            cmap = plt.get_cmap('tab20b') #initialize color map
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            self.tracker.predict()  # Call the tracker
            self.tracker.update(detections) #  updtate using Kalman Gain

            
            for track in self.tracker.tracks:  # update new findings AKA tracks
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()

                area_1 = [(180,520),(475,499),(530,548),(163,583)]    ############################## this are the two polygons drawn TL, TR,BR,BL
                area_2 = [(127,684),(670,625),(718,664),(109,736)]
                cx = int((bbox[0] + bbox[2]) / 2)  #this is the center x and y coordinate of each vehicle passing.
                cy = int((bbox[1] + bbox[3]) / 2)
                
                result = cv2.pointPolygonTest(np.array(area_1, np.int32), (int(cx), int(cy)), False) # this validates car center passed polygon or not 

                if result>=0:  ################ if the polygon is passed 
                    # vehicles_entering[track.track_id]=(cx,cy)
                    vehicles_entering[track.track_id]=time.time() # store starting time with its id in dict.
                # print(vehicles_entering)
                if track.track_id in vehicles_entering:    # if vehicle from vehicle entering dictonary is passed through another polygon
                    result = cv2.pointPolygonTest(np.array(area_2, np.int32), (int(cx), int(cy)), False) # vehicle passsed through another poly testing
                    if result>0:  # if passed
                        elapsed_time = time.time() - vehicles_entering[track.track_id] ## calculate total time taken

                        if track.track_id not in vehicles_elapsed_time: # if time taken of id not in list store it with id
                            vehicles_elapsed_time[track.track_id] = elapsed_time

                        if track.track_id in vehicles_elapsed_time:
                            elapsed_time = vehicles_elapsed_time[track.track_id] # assign value to each track id

                        # vehicles_elapsed_time[track.track_id] = elapsed_time

                        # Calculate average speed
                        distance=65 # meters
                        a_speed_ms = distance/elapsed_time
                        a_speed_kh = a_speed_ms*3.6

                        color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                        color = [i * 255 for i in color]
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                        # cv2.putText(frame, class_name + " : " + str(track.track_id) + " E.T. " + str(int(elapsed_time)),(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    
                        cv2.putText(frame, str(int(a_speed_kh))+' km/h',(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    
                        
                        cv2.circle(frame, (cx, cy), 5, color,-1) # marking center of vehicle
                        #cv2.line(frame, (317,490), (717,1071), [0,0,0], 2)   # this is lane divider line
                                
                if verbose == 2:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

                for i,area in enumerate([area_1, area_2]):
                        if i==1:
                            continue
                        # cv2.polylines(frame, [np.array(area, np.int32)],True, (15,220,10),6)    # this command prints polygon
                    # print(frame.shape)          ###############

            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
            if verbose >= 1:
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
               
                if not os.path.exists(r"file.txt"):                       
                    with open('myfile.txt', 'w') as fp:
                        pass
                
                with open(r"file.txt","a") as file:
                    if not count_objects: 
                        file.write(f"\n{datetime.datetime.now()} || Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                        print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                    else:
                        cc = Counter(classes)
                        data = "".join([f"{i}: {cc[i]} " for i in cc.keys()])
                        file.write(f"\n{datetime.datetime.now()} ||Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {count} || {data}") 
                        print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {count} || {data}") ###########

            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if output: out.write(result) # save output video

            if show_live:
                cv2.imshow("Output Video", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cv2.destroyAllWindows()
