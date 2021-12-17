from glob import glob
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

csv_files = glob('out/*.csv')
files_data = []
window = 10


def to_float(l):
    return list(map(lambda n: float(n), l))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


for f in csv_files:
    with open(f, newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        d_nose_x = 0
        d_nose_y = 0
        d_nose_z = 0
        d_sh_x = 0
        d_sh_y = 0
        d_sh_z = 0
        d_nose_center_sh_angle = 0
        d_sh_angle = 0
        n_falling = 0
        for idx, row in enumerate(csv_reader):
            frame_no = int(row[0])
            falling = 1 if row[1] == 'True' else 0
            nose_x, nose_y, nose_z = to_float(row[2:5])
            l_sh_x, l_sh_y, l_sh_z = to_float(row[5:8])
            r_sh_x, r_sh_y, r_sh_z = to_float(row[8:11])
            center_sh_x = (r_sh_x + l_sh_x) / 2
            center_sh_y = (r_sh_y + l_sh_y) / 2
            center_sh_z = (r_sh_z + l_sh_z) / 2
            sh_angle = angle_between(
                (l_sh_x, l_sh_y, l_sh_z), (r_sh_x, r_sh_y, r_sh_z))
            nose_center_sh_angle = angle_between(
                (center_sh_x, center_sh_y, center_sh_z), (nose_x, nose_y, nose_z))

            d_nose_y = nose_y - d_nose_y
            d_nose_center_sh_angle = nose_center_sh_angle - d_nose_center_sh_angle
            d_sh_angle = sh_angle - d_sh_angle

            if falling == 1:
                n_falling += 1

            if idx % window == 0:
                files_data.append(
                    [d_nose_y/window, d_nose_center_sh_angle/window, d_sh_angle/window, 1 if n_falling > window / 2 else 0])
                d_nose_x = 0
                d_nose_y = 0
                d_nose_z = 0
                d_sh_x = 0
                d_sh_y = 0
                d_sh_z = 0
                n_falling = 0
                d_nose_center_sh_angle = 0
                d_sh_angle = 0

df = pd.DataFrame(np.array(files_data),
                  columns=['1', '2', '3', 'label'])

label_column = ['label']
non_label_col = list(set(list(df.columns))-set(label_column))
X = df[non_label_col].values
y = df[label_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

# clf = DecisionTreeClassifier(
#     max_depth=15, criterion='entropy')
clf = RandomForestClassifier(
    max_depth=15, criterion='entropy')
# clf = svm.SVC(C=0.2)
# clf = SGDClassifier()
clf.fit(X_train, y_train.ravel())

pred_train_tree = clf.predict(X_train)
print("train MSE: {0}".format(
    np.sqrt(mean_squared_error(y_train, pred_train_tree))))
print("train accuracy: {0}".format(r2_score(y_train, pred_train_tree)))

pred_test_tree = clf.predict(X_test)
print("test MSE: {0}".format(
    np.sqrt(mean_squared_error(y_test, pred_test_tree))))
print("test accuracy: {0}".format(r2_score(y_test, pred_test_tree)))

# plot_tree(clf)
# plt.show()
