from pathlib import Path
from glob import glob
import csv
import cv2
import json
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def flatten(t):
    return [item for sublist in t for item in sublist]


def get_annotation(path):
    try:
        return list(map(lambda n: int(n), open(path, 'r').readlines()[:2]))
    except Exception as e:
        print(e)
        return [0, 0]


video_files = glob('data/*/Videos/*.avi')

with mp_pose.Pose(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    for i, f in enumerate(video_files):
        file_path = Path(f)
        annotation = get_annotation(str(file_path.parent.parent.joinpath(
            './Annotation_files/{0}.txt'.format(file_path.name.split('.')[0]))))
        fb, fe = annotation
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
            if fb != fe and count >= fb and count <= fe:
                falling = True

            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            if not results.pose_world_landmarks:
                count_failed = count_failed + 1
                continue

            landmark_data = []
            for i in range(len(results.pose_world_landmarks.landmark)):
                landmark = results.pose_world_landmarks.landmark[i]
                landmark_data.append(
                    [landmark.x, landmark.y, landmark.z, landmark.visibility])
            video_data.append([count, falling, *flatten(landmark_data)])

        cap.release()
        print("{0} frames processed".format(count))
        print("{0} frames failed to process".format(count_failed))
        with open("out/pose-preds/{0}-{1}.csv".format(Path(f).name, count), 'w', newline='\n') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            for data in video_data:
                csv_writer.writerow(data)
