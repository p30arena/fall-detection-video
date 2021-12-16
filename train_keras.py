import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from keras_ds import inputs, dataset_train, dataset_val

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

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
    epochs=50,
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


for x, y in dataset_val.take(5):
    print("truth: {0} prediction: {1}".format(y[0].numpy()[0], 0 if tf.sigmoid(
        model.predict(x)[0]) < 0.5 else 1))
