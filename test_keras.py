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
    sequence_length=5,
    sampling_rate=1,
    batch_size=64,
)

# print(len(dataset))
# exit()
for x, y in dataset:
    print(x.shape)
    print(y.shape)
    for idx, input in enumerate(x):
        print("truth: {0} prediction: {1}".format(y[idx].numpy()[0], 0 if tf.sigmoid(
            model.predict(input[None, ...])[0]) < 0.5 else 1))
