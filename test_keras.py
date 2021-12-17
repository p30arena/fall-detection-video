import tensorflow as tf
from tensorflow import keras

from keras_ds import df

model = keras.models.load_model('out/model/model_checkpoint.h5')

target_column_train = ['label']
predictors_train = list(set(list(df.columns))-set(target_column_train))

X = df[predictors_train].values
y = df[target_column_train].values

print(X.shape)
print(y.shape)


dataset = keras.preprocessing.timeseries_dataset_from_array(
    X,
    y,
    sequence_length=20,
    sampling_rate=1,
    batch_size=64,
)

# print(len(dataset))
# exit()
n_truth_fall = 0
n_truth_non_fall = 0
n_pred_fall = 0
n_pred_non_fall = 0

for _, (x, y) in enumerate(dataset):
    # print(x.shape)
    # print(y.shape)
    print(_)
    for idx, input in enumerate(x):
        truth = y[idx].numpy()[0]
        pred = 0 if tf.sigmoid(
            model.predict(input[None, ...])[0]) < 0.5 else 1

        if truth == 0:
            n_truth_non_fall += 1
            if truth == pred:
                n_pred_non_fall += 1
        else:
            n_truth_fall += 1
            if truth == pred:
                n_pred_fall += 1

print("fall accuracy: {0}".format(n_pred_fall / n_truth_fall))
print("non fall accuracy: {0}".format(n_pred_non_fall / n_truth_non_fall))
