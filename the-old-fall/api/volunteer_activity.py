# -*- coding: utf-8 -*-
'''
义工是否和老人有互动主程序

用法：
python checkingvolunteeractivity.py
python checkingvolunteeractivity.py --filename tests/desk_01.mp4
'''

from oldcare.facial import FaceUtil
from scipy.spatial import distance as dist
from oldcare.utils import fileassistant
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import os
import imutils
import numpy as np
import argparse
import subprocess


class volunteerActivity():
    def __init__(self):

        # 传入参数
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-f", "--filename", required=False, default='',
                             help="")
        self.args = vars(self.ap.parse_args())

        # 全局变量
        self.pixel_per_metric = None
        self.input_video = self.args['filename']
        self.output_activity_path = '../supervision/activity'
        self.model_path = '../models/face_recognition_hog.pickle'
        self.people_info_path = '../info/people_info.csv'
        self.camera_turned = 0
        self.python_path = '/home/reed/anaconda3/envs/tensorflow/bin/python'  # your python path

        # 全局常量
        self.FACE_ACTUAL_WIDTH = 20  # 单位厘米   姑且认为所有人的脸都是相同大小
        self.VIDEO_WIDTH = 640
        self.VIDEO_HEIGHT = 480
        self.ANGLE = 20
        self.ACTUAL_DISTANCE_LIMIT = 100  # cm

        # 得到 ID->姓名的map 、 ID->职位类型的map
        self.id_card_to_name, self.id_card_to_type = fileassistant.get_people_info(self.people_info_path)

        # 加载模型
        self.faceutil = FaceUtil(self.model_path)

        print('[INFO] 开始检测义工和老人是否有互动...')

    def volunteerActivityAPI(self, grabbed, frame, counter):
        self.camera_turned = 0

        # if we are viewing a video and we did not grab a frame, then we
        # have reached the end of the video
        if self.input_video and not grabbed:
            return grabbed, frame, True

        if not self.input_video:
            frame = cv2.flip(frame, 1)

        frame = imutils.resize(frame,
                               width=self.VIDEO_WIDTH,
                               height=self.VIDEO_HEIGHT)  # 压缩，为了加快识别速度

        face_location_list, names = self.faceutil.get_face_location_and_name(frame)

        # 得到画面的四分之一位置和四分之三位置，并垂直划线
        one_sixth_image_center = (int(self.VIDEO_WIDTH / 6), int(self.VIDEO_HEIGHT / 6))
        five_sixth_image_center = (int(self.VIDEO_WIDTH / 6 * 5),
                                   int(self.VIDEO_HEIGHT / 6 * 5))

        cv2.line(frame, (one_sixth_image_center[0], 0),
                 (one_sixth_image_center[0], self.VIDEO_HEIGHT),
                 (0, 255, 255), 1)
        cv2.line(frame, (five_sixth_image_center[0], 0),
                 (five_sixth_image_center[0], self.VIDEO_HEIGHT),
                 (0, 255, 255), 1)

        people_type_list = list(set([self.id_card_to_type[i] for i in names]))

        volunteer_name_direction_dict = {}
        volunteer_centroids = []
        old_people_centroids = []
        old_people_name = []

        # loop over the face bounding boxes
        for ((left, top, right, bottom), name) in zip(face_location_list, names):  # 处理单个人

            person_type = self.id_card_to_type[name]
            # 将人脸框出来
            rectangle_color = (0, 0, 255)
            if person_type == 'old_people':
                rectangle_color = (0, 0, 128)
            elif person_type == 'employee':
                rectangle_color = (255, 0, 0)
            elif person_type == 'volunteer':
                rectangle_color = (0, 255, 0)
            else:
                pass
            cv2.rectangle(frame, (left, top), (right, bottom),
                          rectangle_color, 2)

            if 'volunteer' not in people_type_list:  # 如果没有义工，直接跳出本次循环
                continue

            if person_type == 'volunteer':  # 如果检测到有义工存在
                # 获得义工位置
                volunteer_face_center = (int((right + left) / 2),
                                         int((top + bottom) / 2))
                volunteer_centroids.append(volunteer_face_center)

                cv2.circle(frame,
                           (volunteer_face_center[0], volunteer_face_center[1]),
                           8, (255, 0, 0), -1)

                adjust_direction = ''
                # face locates too left, servo need to turn right,
                # so that face turn right as well
                if volunteer_face_center[0] < one_sixth_image_center[0]:
                    adjust_direction = 'right'
                elif volunteer_face_center[0] > five_sixth_image_center[0]:
                    adjust_direction = 'left'

                volunteer_name_direction_dict[name] = adjust_direction

            elif person_type == 'old_people':  # 如果没有发现义工
                old_people_face_center = (int((right + left) / 2),
                                          int((top + bottom) / 2))
                old_people_centroids.append(old_people_face_center)
                old_people_name.append(name)

                cv2.circle(frame,
                           (old_people_face_center[0], old_people_face_center[1]),
                           4, (0, 255, 0), -1)
            else:
                pass

            # 人脸识别和表情识别都结束后，把表情和人名写上 (同时处理中文显示问题)
            img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_PIL)
            final_label = self.id_card_to_name[name]
            draw.text((left, top - 30), final_label,
                      font=ImageFont.truetype('NotoSansCJK-Black.ttc', 40),
                      fill=(255, 0, 0))  # linux
            # 转换回OpenCV格式
            frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

        # 义工追踪逻辑
        if 'volunteer' in people_type_list:
            volunteer_adjust_direction_list = list(volunteer_name_direction_dict.values())
            if '' in volunteer_adjust_direction_list:  # 有的义工恰好在范围内，所以不需要调整舵机
                print('%d-有义工恰好在可见范围内，摄像头不需要转动' % (counter))
            else:
                adjust_direction = volunteer_adjust_direction_list[0]
                self.camera_turned = 1
                print('%d-摄像头需要 turn %s %d 度' % (counter, adjust_direction, self.ANGLE))

        # 在义工和老人之间划线
        if self.camera_turned == 0:
            for i in volunteer_centroids:
                for j_index, j in enumerate(old_people_centroids):
                    pixel_distance = dist.euclidean(i, j)
                    face_pixel_width = sum([i[2] - i[0] for i in face_location_list]) / len(face_location_list)
                    pixel_per_metric = face_pixel_width / self.FACE_ACTUAL_WIDTH
                    actual_distance = pixel_distance / pixel_per_metric

                    if actual_distance < self.ACTUAL_DISTANCE_LIMIT:
                        cv2.line(frame, (int(i[0]), int(i[1])),
                                 (int(j[0]), int(j[1])), (255, 0, 255), 2)
                        label = 'distance: %dcm' % (actual_distance)
                        cv2.putText(frame, label, (frame.shape[1] - 150, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 2)

                        current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(time.time()))
                        event_desc = '%s正在与义工交互' % (self.id_card_to_name[old_people_name[j_index]])
                        event_location = '房间桌子'
                        print(
                            '[EVENT] %s, 房间桌子, %s 正在与义工交互.' % (current_time, self.id_card_to_name[old_people_name[j_index]]))
                        cv2.imwrite(
                            os.path.join(self.output_activity_path, 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                            frame)  # snapshot

                        # insert into database
                        command = '%s inserting.py --event_desc %s --event_type 1 --event_location %s --old_people_id %d' % (
                            self.python_path, event_desc, event_location, int(name))
                        p = subprocess.Popen(command, shell=True)

        # show our detected faces along with smiling/not smiling labels
        # cv2.imshow("Checking Volunteer's Activities", frame)
        return grabbed, frame, False
        # Press 'ESC' for exiting video
        # k = cv2.waitKey(1) & 0xff
        # if k == 27:
        #     return grabbed, frame, True


buf = volunteerActivity()
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
    (grabbed, frame, flag) = buf.volunteerActivityAPI(grabbed, frame, counter)
    cv2.imshow("Checking Volunteer's Activities", frame)
    if flag:
        break
# cleanup the camera and close any open windows
vs.release()
cv2.destroyAllWindows()
