from pathlib import Path
from glob import glob
import csv
import cv2
import mediapipe as mp

from ant import annotations

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def flatten(t):
    return [item for sublist in t for item in sublist]


video_files = glob('data/*/Videos/*.avi')

landmark_keys = [
    # mp_pose.PoseLandmark.LEFT_SHOULDER,
    # mp_pose.PoseLandmark.RIGHT_SHOULDER,
    # mp_pose.PoseLandmark.LEFT_ELBOW,
    # mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
]

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    for i, f in enumerate(video_files):
        fb, fe = annotations[i]
        cap = cv2.VideoCapture(f)
        count = 0
        count_failed = 0
        video_data = []
        while cap.isOpened():
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                break
            ret, frame = cap.read()
            count = count + 1

            falling = False
            if count >= fb and count <= fe:
                falling = True

            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            if not results.pose_world_landmarks:
                count_failed = count_failed + 1
                continue
            # print(results._fields)
            # print(results.pose_landmarks)
            # print(results.pose_world_landmarks)
            # print(results.pose_world_landmarks._fields)
            # print(dir(results.pose_world_landmarks))
            landmark_data = []
            # print(len(results.pose_world_landmarks.landmark))
            # print(mp_pose.PoseLandmark.NOSE)
            # print(
            #     results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
            # print(
            #     results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x)
            for k in landmark_keys:
                landmark = results.pose_world_landmarks.landmark[k]
                landmark_data.append([landmark.x, landmark.y, landmark.z])
            video_data.append([count, falling, *flatten(landmark_data)])

            # print(video_data)
            # exit()
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # mp_drawing.draw_landmarks(
            #     frame,
            #     results.pose_landmarks,
            #     mp_pose.POSE_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # cv2.imwrite("out/{0}-{1}.jpg".format(Path(f).name, count), frame)
        cap.release()
        print("{0} frames processed".format(count))
        print("{0} frames failed to process".format(count_failed))
        with open("out/{0}-{1}.csv".format(Path(f).name, count), 'w', newline='\n') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            for data in video_data:
                csv_writer.writerow(data)
