from pathlib import Path
from glob import glob
import csv
import numpy as np

import mediapipe as mp
from utils import to_float, center_vector

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

n_total_frames = 0
n_falling_frames = 0
n_errors = 0
n_falling_errors = 0

for f in csv_files:
    file_path = Path(f)
    new_file = str(file_path.parent.parent.joinpath(
        './filtered-pose/{0}'.format(file_path.name)))

    with open(f, newline='\n') as csvfile:
        with open(new_file, 'w', newline='\n') as new_csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            csv_writer = csv.writer(new_csvfile, delimiter=',')
            for idx, row in enumerate(csv_reader):
                n_total_frames += 1

                frame_no, falling = row[:2]
                frame_no = int(frame_no)
                falling = 1 if falling == 'True' else 0

                if falling == 1:
                    n_falling_frames += 1

                landmarks = np.array(to_float(row[2:]))
                la2 = landmarks.reshape((33, 4))
                _, _, _, v_left_foot = la2[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                _, _, _, v_right_foot = la2[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                _, _, _, v_left_hip = la2[mp_pose.PoseLandmark.LEFT_HIP]
                _, _, _, v_right_hip = la2[mp_pose.PoseLandmark.RIGHT_HIP]
                _, _, _, v_left_shoulder = la2[mp_pose.PoseLandmark.LEFT_SHOULDER]
                _, _, _, v_right_shoulder = la2[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                if (v_left_foot < 0.1 and v_right_foot < 0.1) or \
                        (v_left_hip < 0.1 and v_right_hip < 0.1) or \
                    (v_left_shoulder < 0.1 or v_right_shoulder < 0.1):
                    # print(falling)
                    # print(v_left_foot)
                    # print(v_right_foot)
                    n_errors += 1
                    if falling == 1:
                        n_falling_errors += 1
                    continue

                # x_left_hip, y_left_hip, z_left_hip, _ = la2[mp_pose.PoseLandmark.LEFT_HIP]
                # x_right_hip, y_right_hip, z_right_hip, _ = la2[mp_pose.PoseLandmark.RIGHT_HIP]

                # print(x_left_hip)

                # center_hip = center_vector(
                #     la2[mp_pose.PoseLandmark.LEFT_HIP][:3], la2[mp_pose.PoseLandmark.RIGHT_HIP][:3])
                # print(center_hip)
                center_shoulders = center_vector(la2[mp_pose.PoseLandmark.LEFT_SHOULDER]
                                                 [:3].tolist(), la2[mp_pose.PoseLandmark.RIGHT_SHOULDER][:3].tolist())
                csv_writer.writerow(
                    [frame_no, falling, *center_shoulders])

print('n_errors: {0}'.format(n_errors))
print('n_total_frames: {0}'.format(n_total_frames))
print('n_falling_frames: {0}'.format(n_falling_frames))
print('n_falling_errors: {0}'.format(n_falling_errors))
