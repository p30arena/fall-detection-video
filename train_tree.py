from glob import glob
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

csv_files = glob('out/*.csv')
files_data = []
df = None
window = 10

for f in csv_files:
    with open(f, newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        d_lw_x = 0
        d_lw_y = 0
        d_lw_z = 0
        n_falling = 0
        for idx, row in enumerate(csv_reader):
            frame_no = int(row[0])
            falling = 1 if row[1] == 'True' else 0
            lw_x = float(row[2])
            lw_y = float(row[3])
            lw_z = float(row[4])
            d_lw_x = lw_x - d_lw_x
            d_lw_y = lw_y - d_lw_y
            d_lw_z = lw_z - d_lw_z
            if falling == 1:
                n_falling += 1
            if idx % window == 0:
                files_data.append(
                    [d_lw_x/window, d_lw_y/window, d_lw_z/window, 1 if n_falling > window / 2 else 0])
                d_lw_x = 0
                d_lw_y = 0
                d_lw_z = 0
                n_falling = 0

df = pd.DataFrame(np.array(files_data),
                  columns=['delta_x', 'delta_y', 'delta_z', 'label'])

_8p = round(len(df) * 0.8)
train = df[:_8p]
test = df[_8p:]

target_column_train = ['label']
predictors_train = list(set(list(train.columns))-set(target_column_train))

X_train = train[predictors_train].values
y_train = train[target_column_train].values

print(X_train.shape)
print(y_train.shape)

target_column_test = ['label']
predictors_test = list(set(list(test.columns))-set(target_column_test))

X_test = test[predictors_test].values
y_test = test[target_column_test].values

print(X_test.shape)
print(y_test.shape)

dtree = DecisionTreeClassifier(max_depth=10)
dtree.fit(X_train, y_train)

pred_train_tree = dtree.predict(X_train)
print("train MSE: {0}".format(
    np.sqrt(mean_squared_error(y_train, pred_train_tree))))
print("train accuracy: {0}".format(r2_score(y_train, pred_train_tree)))

pred_test_tree = dtree.predict(X_test)
print("test MSE: {0}".format(
    np.sqrt(mean_squared_error(y_test, pred_test_tree))))
print("test accuracy: {0}".format(r2_score(y_test, pred_test_tree)))

# plot_tree(dtree)
# plt.show()
