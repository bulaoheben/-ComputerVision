# -*- coding: utf-8 -*-
'''
摔倒检测模型主程序

用法：
python checkingfalldetection.py
python checkingfalldetection.py --filename tests/corridor_01.avi
'''

# import the necessary packages
from keras.utils import image_utils
from keras.models import load_model
import numpy as np
import cv2
import os
import time
import subprocess
import argparse


class fallDetection():

    def __init__(self):

        # 传入参数
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-f", "--filename", required=False, default='',
                             help="")
        self.args = vars(self.ap.parse_args())
        self.input_video = self.args['filename']

        # 控制陌生人检测
        self.fall_timing = 0  # 计时开始
        self.fall_start_time = 0  # 开始时间
        self.fall_limit_time = 1  # if >= 1 seconds, then he/she falls.

        # 全局变量
        self.model_path = '../models/fall_detection.hdf5'
        self.output_fall_path = '../supervision/fall'
        # your python path
        self.python_path = '/home/reed/anaconda3/envs/tensorflow/bin/python'

        # 全局常量
        self.TARGET_WIDTH = 64
        self.TARGET_HEIGHT = 64


        # 加载模型
        self.model = load_model(self.model_path)
        self.event = 4
        print('[INFO] 开始检测是否有人摔倒...')

    def fallDetectionAPI(self, grabbed, image, counter):

        # if we are viewing a video and we did not grab a frame, then we
        # have reached the end of the video
        if self.input_video and not grabbed:
            return grabbed, image, True

        if not self.input_video:
            image = cv2.flip(image, 1)

        roi = cv2.resize(image, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
        roi = roi.astype("float") / 255.0
        roi = image_utils.img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # determine facial expression
        (fall, normal) = self.model.predict(roi)[0]
        label = "Fall (%.2f)" % (fall) if fall > normal else "Normal (%.2f)" % (normal)

        # display the label and bounding box rectangle on the output frame
        cv2.putText(image, label, (image.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        if fall > normal:
            if self.fall_timing == 0:  # just start timing
                self.fall_timing = 1
                self.fall_start_time = time.time()
            else:  # alredy started timing
                self.fall_end_time = time.time()
                difference = self.fall_end_time - self.fall_start_time

                self.current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                             time.localtime(time.time()))

                if difference < self.fall_limit_time:
                    print('[INFO] %s, 走廊, 摔倒仅出现 %.1f 秒. 忽略.' % (self.current_time, difference))
                else:  # strangers appear
                    event_desc = '有人摔倒!!!'
                    event_location = '走廊'
                    print('[EVENT] %s, 走廊, 有人摔倒!!!' % (self.current_time))
                    cv2.imwrite(os.path.join(self.output_fall_path,
                                             'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))), image)  # snapshot
                    # insert into database
                    command = '%s inserting.py --event_desc %s --event_type 3 --event_location %s' % (
                        self.python_path, event_desc, event_location)
                    p = subprocess.Popen(command, shell=True)

        cv2.imshow('Fall detection', image)
        return grabbed, image, False
        # Press 'ESC' for exiting video
        # k = cv2.waitKey(1) & 0xff
        # if k == 27:
        #     return grabbed, image, True



buf = fallDetection()
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
    # (grabbed, frame, flag) = buf.fallDetectionAPI(grabbed, frame, counter)
    buf.fallDetectionAPI(grabbed, frame, counter)

    if flag:
        break
vs.release()
cv2.destroyAllWindows()
