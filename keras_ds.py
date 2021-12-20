from utils import get_df, get_df_keras
from tensorflow import keras

batch_size = 256

sequence_length = 10
df = get_df_keras()
# sequence_length = 1
# df = get_df()

num_samples = len(df)
_80p = round(0.8 * num_samples)
_10p = round(0.1 * num_samples)

train = df[:_80p]
val = df[_80p: _80p + _10p]
test = df[-_10p:]

target_column_train = ['label']
predictors_train = list(set(list(train.columns))-set(target_column_train))

X_train = train[predictors_train].values
y_train = train[target_column_train].values

print(X_train.shape)
print(y_train.shape)

target_column_val = ['label']
predictors_val = list(set(list(val.columns))-set(target_column_val))

X_val = val[predictors_val].values
y_val = val[target_column_val].values

print(X_val.shape)
print(y_val.shape)

target_column_test = ['label']
predictors_test = list(set(list(test.columns))-set(target_column_test))

X_test = test[predictors_test].values
y_test = test[target_column_test].values

print(X_test.shape)
print(y_test.shape)

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    X_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=1,
    batch_size=batch_size,
)

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    X_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=1,
    batch_size=batch_size,
)

for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)
