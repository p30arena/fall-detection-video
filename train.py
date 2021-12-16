from glob import glob
import csv

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

csv_files = glob('out/*.csv')

clf = SGDClassifier()

for f in csv_files:
    raw_data = []
    df = None
    with open(f, newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            # x**2 + y**2
            raw_data.append([int(row[0]), float(row[2])**2 + float(row[3])**2,
                             1 if row[1] == 'True' else 0])

        df = pd.DataFrame(np.array(raw_data),
                          columns=['frame_no', 'distance', 'label'])

        train = df

        target_column_train = ['label']
        predictors_train = list(
            set(list(train.columns))-set(target_column_train))

        X_train = train[predictors_train].values
        y_train = train[target_column_train].values

        clf.fit(X_train, y_train)

        pred_train_tree = clf.predict(X_train)
        print("train MSE: {0}".format(
            np.sqrt(mean_squared_error(y_train, pred_train_tree))))
        print("train accuracy: {0}".format(r2_score(y_train, pred_train_tree)))
