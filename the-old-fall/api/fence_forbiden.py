# -*- coding: utf-8 -*-

'''
测试电子围栏
摄像头对准围墙那一侧

用法：
python testingfence.py
python testingfence.py --filename tests/yard_01.mp4
'''

# import the necessary packages
from oldcare.track import CentroidTracker
from oldcare.track import TrackableObject
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2
import argparse


class fenceForbiden():
    def __init__(self):

        # 传入参数
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-f", "--filename", required=False, default='',
                             help="")
        self.args = vars(self.ap.parse_args())

        # 全局变量
        self.prototxt_file_path = '../models/mobilenet_ssd/MobileNetSSD_deploy.prototxt'
        # # Contains the Caffe deep learning model files. We’ll be using a
        # MobileNet Single Shot Detector (SSD), “Single Shot Detectors for
        # object detection”.
        self.model_file_path = '../models/mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
        self.input_video = self.args['filename']
        self.skip_frames = 30  # of skip frames between detections

        # 超参数
        # minimum probability to filter weak detections
        self.minimum_confidence = 0.80

        # 物体识别模型能识别的物体（21种）
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair",
                        "cow", "diningtable", "dog", "horse", "motorbike",
                        "person", "pottedplant", "sheep", "sofa", "train",
                        "tvmonitor"]

        # 加载物体识别模型
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_file_path, self.model_file_path)

        # if a video path was not supplied, grab a reference to the webcam
        if not self.input_video:
            print("[INFO] starting video stream...")
            self.vs = cv2.VideoCapture(0)
            time.sleep(2)
        else:
            print("[INFO] opening video file...")
            self.vs = cv2.VideoCapture(self.input_video)

        # initialize the frame dimensions (we'll set them as soon as we read
        # the first frame from the video)
        self.W = None
        self.H = None

        # instantiate our centroid tracker, then initialize a list to store
        # each of our dlib correlation trackers, followed by a dictionary to
        # map each unique object ID to a TrackableObject
        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        self.trackers = []
        self.trackableObjects = {}

        # initialize the total number of frames processed thus far, along
        # with the total number of objects that have moved either up or down
        self.totalFrames = 0
        self.totalDown = 0
        self.totalUp = 0

        # start the frames per second throughput estimator
        self.fps = FPS().start()

    # loop over frames from the video stream
    def fenceForbidenAPI(self, ret, frame):

        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if self.input_video and not ret:
            return ret, frame, True

        if not self.input_video:
            frame = cv2.flip(frame, 1)

        # resize the frame to have a maximum width of 500 pixels (the
        # less data we have, the faster we can process it), then convert
        # the frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if self.totalFrames % self.skip_frames == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []

            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > self.minimum_confidence:
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a person, ignore it
                    if self.CLASSES[idx] != "person":
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in self.trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # draw a rectangle around the people
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(frame, (0, self.H // 2), (self.W, self.H // 2), (0, 255, 255), 2)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = self.ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = self.trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < self.H // 2:
                        self.totalUp += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > self.H // 2:
                        self.totalDown += 1
                        to.counted = True

            # store the trackable object in our dictionary
            self.trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4,
                       (0, 255, 0), -1)

        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            # ("Up", totalUp),
            ("Down", self.totalDown),
            ("Status", status),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, self.H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # show the output frame
        # cv2.imshow("Frame", frame)
        return ret, frame, False
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            return ret, frame, True

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        self.totalFrames += 1
        self.fps.update()



buf = fenceForbiden()
# 初始化摄像头
if not buf.input_video:
    vs = cv2.VideoCapture(0)
    time.sleep(2)
else:
    vs = cv2.VideoCapture(buf.input_video)
# 不断循环
counter = 0
flag = False
while True:
    counter += 1
    (grabbed, frame) = vs.read()
    (grabbed, frame, flag) = buf.fenceForbidenAPI(grabbed, frame)
    if flag:
        break
# stop the timer and display FPS information
buf.fps.stop()
print("[INFO] elapsed time: {:.2f}".format(buf.fps.elapsed()))  # 14.19
print("[INFO] approx. FPS: {:.2f}".format(buf.fps.fps()))  # 90.43

# close any open windows
vs.release()
cv2.destroyAllWindows()
