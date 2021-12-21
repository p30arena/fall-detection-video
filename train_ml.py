from utils import get_df, get_df_keras, get_df_ex

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

df = get_df_ex()
# df = get_df_keras()
# df = shuffle(df)

label_column = ['label']
non_label_col = list(set(list(df.columns))-set(label_column))
X = df[non_label_col].values
y = df[label_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = RandomForestClassifier(
    n_estimators=300, max_depth=30, criterion='entropy')

clf.fit(X_train, y_train.ravel())

pred_train_tree = clf.predict(X_train)
print("train MSE: {0}".format(
    np.sqrt(mean_squared_error(y_train, pred_train_tree))))
print("train accuracy: {0}".format(r2_score(y_train, pred_train_tree)))

pred_test_tree = clf.predict(X_test)
print("test MSE: {0}".format(
    np.sqrt(mean_squared_error(y_test, pred_test_tree))))
print("test accuracy: {0}".format(r2_score(y_test, pred_test_tree)))

# plot_tree(clf.estimators_[0])
# plt.show()
