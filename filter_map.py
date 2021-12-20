from glob import glob
import csv
import numpy as np

import mediapipe as mp
from utils import to_float

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

csv_files = glob('out/pose-preds/*.csv')

# class PoseLandmark(enum.IntEnum):
#   """The 33 pose landmarks."""
#   NOSE = 0
#   LEFT_EYE_INNER = 1
#   LEFT_EYE = 2
#   LEFT_EYE_OUTER = 3
#   RIGHT_EYE_INNER = 4
#   RIGHT_EYE = 5
#   RIGHT_EYE_OUTER = 6
#   LEFT_EAR = 7
#   RIGHT_EAR = 8
#   MOUTH_LEFT = 9
#   MOUTH_RIGHT = 10
#   LEFT_SHOULDER = 11
#   RIGHT_SHOULDER = 12
#   LEFT_ELBOW = 13
#   RIGHT_ELBOW = 14
#   LEFT_WRIST = 15
#   RIGHT_WRIST = 16
#   LEFT_PINKY = 17
#   RIGHT_PINKY = 18
#   LEFT_INDEX = 19
#   RIGHT_INDEX = 20
#   LEFT_THUMB = 21
#   RIGHT_THUMB = 22
#   LEFT_HIP = 23
#   RIGHT_HIP = 24
#   LEFT_KNEE = 25
#   RIGHT_KNEE = 26
#   LEFT_ANKLE = 27
#   RIGHT_ANKLE = 28
#   LEFT_HEEL = 29
#   RIGHT_HEEL = 30
#   LEFT_FOOT_INDEX = 31
#   RIGHT_FOOT_INDEX = 32

for f in csv_files:
    with open(f, newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(csv_reader):
            frame_no, falling = row[:2]
            frame_no = int(frame_no)
            falling = 1 if falling == 'True' else 0

            landmarks = np.array(to_float(row[2:]))
            la2 = landmarks.reshape((33, 4))
            _, _, _, v_left_foot = la2[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            _, _, _, v_right_foot = la2[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

            if v_left_foot < 0.1 and v_right_foot < 0.1:
                print(v_left_foot)
                print(v_right_foot)
