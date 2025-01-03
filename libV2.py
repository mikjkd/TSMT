from typing import List

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import Callback
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def plot_set(images, labels=None, rows=5, cols=5, mul_size=1.5):
    fig, axs = plt.subplots(rows, cols, figsize=(
        mul_size * cols, mul_size * rows))
    if rows == 1:
        axs = np.expand_dims(axs, 0)
    if cols == 1:
        axs = np.expand_dims(axs, 1)
    for i in range(rows):
        for j in range(cols):
            k = cols * i + j
            if k < images.shape[0]:
                axs[i][j].grid(False)
                axs[i][j].set_yticks([])
                axs[i][j].set_xticks([])
                axs[i][j].imshow(images[k], cmap=plt.cm.binary)
                if labels is not None:
                    axs[i][j].set_xlabel(labels[k], color='blue')
            else:
                axs[i][j].set_axis_off()
    fig.tight_layout()


def plot_trend(values, epochs, title):
    plt.plot(values, 'b-')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.xticks(np.arange(epochs))
    plt.show()


def plot_predictions(predictions, class_names, images, labels, print_perc=True, rows=5, cols=3):
    fig, axs = plt.subplots(rows, 2 * cols, figsize=(2 * 2 * cols, 2 * rows))
    if rows == 1:
        axs = np.expand_dims(axs, 0)
    for i in range(rows):
        for j in range(cols):
            axs[i][2 * j].grid(False)
            axs[i][2 * j].set_yticks([])
            axs[i][2 * j].set_xticks([])
            axs[i][2 * j].imshow(images[cols * i + j], cmap=plt.cm.binary)
            predicted_label = np.argmax(predictions[cols * i + j])
            l = "{}".format(class_names[predicted_label])
            if print_perc:
                l += "\n({:2.2f}%)".format(100 *
                                           np.max(predictions[cols * i + j]))
            if predicted_label == labels[cols * i + j]:
                axs[i][2 * j].set_xlabel(l, color='blue')
            else:
                axs[i][2 * j].set_xlabel(l + "\n[{}]".format(
                    class_names[labels[cols * i + j]]), color='red')
            axs[i][2 * j + 1].grid(False)
            axs[i][2 * j + 1].set_yticks([])
            axs[i][2 * j + 1].set_xticks(range(len(class_names)))
            axs[i][2 * j + 1].set_xticklabels(class_names, rotation=90)
            barplot = axs[i][2 * j + 1].bar(range(len(class_names)),
                                            predictions[cols * i + j], color="#777777")
            barplot[predicted_label].set_color('red')
            barplot[labels[cols * i + j]].set_color('blue')
    fig.tight_layout()


def print_accuracy(test_acc, test_loss):
    print('Accuracy on the Test set: {:.2f}% (loss = {:.3f})'.format(
        test_acc * 100, test_loss))


def plot_history(name, history):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
    ax1.plot(history.history['accuracy'], label='accuracy', color='orange')
    ax1.plot(history.history['val_accuracy'],
             label='val_accuracy', color='red')
    ax1.grid()
    ax2.plot(history.history['loss'], label='loss', color='blue')
    ax2.plot(history.history['val_loss'], label='val_loss', color='green')
    ax2.grid()
    ax1.set_title('Accuracy')
    ax2.set_title('Loss')
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    fig.tight_layout(pad=3.0)
    fig.savefig(name, dpi=fig.dpi)


class WeightCapture(Callback):
    "Capture the weights of each layer of the model"

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.weights = []
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)  # remember the epoch axis
        weight = {}
        for layer in self.model.layers:
            if not layer.weights:
                continue
            name = layer.weights[0].name.split("/")[0]
            weight[name] = layer.weights[0].numpy()
        self.weights.append(weight)


def plotweight(capture_cb):
    "Plot the weights' mean and s.d. across epochs"
    fig, ax = plt.subplots(
        2, 1, sharex=True, constrained_layout=True, figsize=(8, 10))
    ax[0].set_title("Mean weight")
    for key in capture_cb.weights[0]:
        ax[0].plot(capture_cb.epochs, [w[key].mean()
                                       for w in capture_cb.weights], label=key)
    ax[0].legend()
    ax[1].set_title("S.D.")
    for key in capture_cb.weights[0]:
        ax[1].plot(capture_cb.epochs, [w[key].std()
                                       for w in capture_cb.weights], label=key)
    ax[1].legend()
    plt.show()


epochs = 30
batch_size = 32


def train_model(X, y, model, n_epochs, batch_size, loss_fn, optimizer):
    "Run training loop manually"
    train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    gradhistory = []
    losshistory = []

    def recordweight():
        data = {}
        for g, w in zip(grads, model.trainable_weights):
            if '/kernel:' not in w.name:
                continue  # skip bias
            name = w.name.split("/")[0]
            data[name] = g.numpy()
        gradhistory.append(data)
        losshistory.append(loss_value.numpy())

    for epoch in range(n_epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y_pred = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, y_pred)

            grads = tape.gradient(loss_value, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step == 0:
                recordweight()
    # After all epochs, record again
    recordweight()
    return gradhistory, losshistory


def plot_gradient(gradhistory, losshistory):
    "Plot gradient mean and sd across epochs"
    fig, ax = plt.subplots(
        3, 1, sharex=True, constrained_layout=True, figsize=(8, 12))
    ax[0].set_title("Mean gradient")
    for key in gradhistory[0]:
        ax[0].plot(range(len(gradhistory)), [w[key].mean()
                                             for w in gradhistory], label=key)
    ax[0].legend()
    ax[1].set_title("S.D.")
    for key in gradhistory[0]:
        ax[1].semilogy(range(len(gradhistory)), [w[key].std()
                                                 for w in gradhistory], label=key)
    ax[1].legend()
    ax[2].set_title("Loss")
    ax[2].plot(range(len(losshistory)), losshistory)
    plt.show()


def save_model_to_file(model, filename):
    json_fn = '{}.json'.format(filename)
    h5_fn = '{}.h5'.format(filename)
    model_json = model.to_json()
    with open(json_fn, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5_fn)
    print("Saved model to disk")


def load_model_from_file(filename):
    json_fn = '{}.json'.format(filename)
    h5_fn = '{}.h5'.format(filename)
    json_file = open(json_fn, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(h5_fn)
    print("Loaded model from disk")
    return model


def accuracy(pred, labels):
    if (pred.shape[0] != labels.shape[0]):
        print('dimensioni non combaciano')
        return (pred.shape[0])
    total = pred.shape[0]
    count = 0
    for j in range(total):
        if np.argmax(pred[j]) == labels[j][0]:
            count += 1
    return count / total


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])










