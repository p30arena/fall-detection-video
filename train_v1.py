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

pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
             columns=['a', 'b', 'c'])
for f in csv_files:
    with open(f, newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            # x**2 + y**2
            files_data.append([int(row[0]), float(row[2])**2 + float(row[3])**2,
                               1 if row[1] == 'True' else 0])

df = pd.DataFrame(np.array(files_data),
                  columns=['frame_no', 'distance', 'label'])

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

dtree = DecisionTreeClassifier(max_depth=8)
dtree.fit(X_train, y_train)

# Code lines 1 to 3
pred_train_tree = dtree.predict(X_train)
print("train MSE: {0}".format(
    np.sqrt(mean_squared_error(y_train, pred_train_tree))))
print("train accuracy: {0}".format(r2_score(y_train, pred_train_tree)))

# Code lines 4 to 6
pred_test_tree = dtree.predict(X_test)
print("test MSE: {0}".format(
    np.sqrt(mean_squared_error(y_test, pred_test_tree))))
print("test accuracy: {0}".format(r2_score(y_test, pred_test_tree)))

plot_tree(dtree, feature_names=train[predictors_train].columns,
          class_names=train[target_column_train].columns, filled=True, fontsize=6, rounded=True)
plt.show()
