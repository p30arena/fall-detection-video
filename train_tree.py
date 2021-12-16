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

for f in csv_files:
    with open(f, newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            files_data.append([int(row[0]), float(row[2]), float(row[3]), float(row[4]),
                               1 if row[1] == 'True' else 0])

df = pd.DataFrame(np.array(files_data),
                  columns=['frame_no', 'x', 'y', 'z', 'label'])

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

dtree = DecisionTreeClassifier(max_depth=20)
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
