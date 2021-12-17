import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from keras_ds import inputs, dataset_train, dataset_val

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

# inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))

# conv1 = keras.layers.Conv1D(
#     filters=64, kernel_size=3, padding="same")(inputs)
# conv1 = keras.layers.BatchNormalization()(conv1)
# conv1 = keras.layers.ReLU()(conv1)

# conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
# conv2 = keras.layers.BatchNormalization()(conv2)
# conv2 = keras.layers.ReLU()(conv2)

# conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
# conv3 = keras.layers.BatchNormalization()(conv3)
# conv3 = keras.layers.ReLU()(conv3)

# gap = keras.layers.GlobalAveragePooling1D()(conv3)

# outputs = keras.layers.Dense(1)(gap)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(
    learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
model.summary()

path_checkpoint = "out/model/model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    # save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=100,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)


def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


visualize_loss(history, "Training and Validation Loss")
