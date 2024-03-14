import pickle as pkl
from datetime import datetime
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
# importing the requests library
import requests
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import keras
from keras.callbacks import Callback


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

# split a univariate sequence into samples
def split_sequence(sequence, n_steps, n_steps_y=1):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix + n_steps_y > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix + n_steps_y]
        # print(seq_x,seq_y)
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def generate_dataset(seq, scaler):
    """mean = seq.mean(axis=0)
    seq -= mean
    std = seq.std(axis = 0)
    seq /= std"""
    scaler = scaler.fit(seq.reshape(-1, 1))
    standardized = scaler.transform(seq.reshape(-1, 1))
    standardized = standardized.reshape(standardized.shape[0])
    X, y = split_sequence(standardized, 28)
    X_train, y_train = X[:int(len(X) * 0.95)], y[:int(len(y) * 0.95)]
    X_test, y_test = X[-int(len(X) * 0.05):], y[-int(len(y) * 0.05):]
    return (X_train, y_train), (X_test, y_test)


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


def is_weekend(date):
    format = '%Y-%m-%d'
    d = datetime.strptime(date, format)
    # print(d.weekday())
    if d.weekday() > 4:
        # print('Given date is weekend.')
        return 1
    else:
        # print('Given dataset is weekday.')
        return 0


def get_day(date):
    format = '%Y-%m-%d'
    d = datetime.strptime(date, format)
    return d.day


def get_month(date):
    format = '%Y-%m-%d'
    d = datetime.strptime(date, format)
    return d.month


def get_weekday(date):
    format = '%Y-%m-%d'
    d = datetime.strptime(date, format)
    return d.weekday()


def get_year(date):
    format = '%Y-%m-%d'
    d = datetime.strptime(date, format)
    return d.year


def oheData(seq, filename, map=None):
    if map is None:
        ohe = OneHotEncoder(sparse=False)
        transformed = ohe.fit_transform(seq.reshape(-1, 1))
        month_map = {}
        for idx, a in enumerate(seq):
            # print(a[0])
            month_map[a] = transformed[idx]
        with open(filename, 'wb') as output:
            pkl.dump(month_map, output)
    else:
        transformed = []
        for idx, a in enumerate(seq):
            # print(a[0])
            transformed.append(map[a])
    return np.array(transformed)


def preprocess_ds(frame, refill_zero=False, drop_zero=False, isMultipleTS=True, setIsCap=True, setIsWeekend=True,
                  withCoupons=True, oheCity=False, oheTs=True, oheMonth=True, oheYear=True, oheMicro=True,
                  dropColumns=True, withMicro=False, oheFiles=None,
                  oheMap=None):
    if oheFiles is None:
        oheFiles = {'weather': '../encoders/weather_ohe_map',
                    'month': '../encoders/oheMonth',
                    'year': '../encoders/oheYears',
                    'ts': '../encoders/oheTs',
                    'cities': '../encoders/oheCities',
                    'micro': '../encoders/oheMicro'}
    frame = frame.copy()  # creo una copia del frame in modo da non modificare quello originale
    if drop_zero:
        frame = frame.drop(
            np.where(frame['orders_completed'].values.astype('int32') == 0)[0])
        frame.reset_index(inplace=True, drop=True)
    if refill_zero:
        c = 'orders_completed'
        zero_pos = frame[frame[c].values.astype('float64') == 0].index
        prec_zero_pos = []
        dist = 4
        for zp in zero_pos:
            # pp = zp-1
            # while int(frame['orders_completed'].values[pp]) == 0:
            #	pp = pp-1
            # prec_zero_pos.append(pp)
            # prendo settiana precedente
            cnt = 1
            nzp = zp
            while float(frame[c].values[nzp]) == 0:
                if zp - (7 * cnt * dist) > 0:
                    nzp = zp - (7 * cnt * dist)
                else:
                    nzp = zp + (7 * cnt * dist)
                cnt += 1
            prec_zero_pos.append(nzp)
        prec_zero_pos = np.array(prec_zero_pos)
        columns = ['orders_completed', 'coupons_count',
                   'bookings_taken', 'bookings_auth']
        for c in columns:
            frame[c].values[zero_pos] = frame[c].values[prec_zero_pos]

        # aggiungo una colonna che indica se l'ordine è stato cappato o meno
        # ciò significa che ho saturato gli ordini possibili
    if oheCity:
        transformed = oheData(
            frame['city'].values.astype(int), oheFiles['cities'])
        for n in range(len(transformed[0])):
            s = f'city_ohe{n + 1}'
            frame[s] = transformed[:, n]
    if setIsCap:
        frame['is_cap'] = [0 for n in range(len(frame))]
        # cerco le righe dove orders_completed >= bookings_taken*5
        # in questo modo capisco se ho avuto un cap degli ordini dovuti ai pochi drivers.
        if not withMicro:
            loc = np.where(
                frame['orders_completed'].values.astype('int32') >= frame['bookings_taken'].values.astype('int32') * 5)
        else:  # se prendo in considerazione solo un'ora, un driver fa massimo 2 ordini
            loc = np.where(
                frame['orders_completed'].values.astype('int32') >= frame['bookings_taken'].values.astype('int32') * 2)
        # imposto che gli ordini sono stati cappati dal numero dei drivers
        frame.loc[loc[0], ('is_cap')] = 1
    if setIsWeekend:
        frame['is_weekend'] = [is_weekend(
            frame['date'].loc[n]) for n in range(len(frame))]
    if oheMonth:
        frame['month'] = [get_month(frame['date'].loc[n])
                          for n in range(len(frame))]

        transformed = oheData(frame['month'].values, oheFiles['month'])

        for n in range(len(transformed[0])):
            s = f'month_ohe{n + 1}'
            frame[s] = transformed[:, n]
    if oheYear:
        years = frame['date']
        # frame['year'] = [get_year(frame['date'].loc[n]) for n in range(len(frame))]
        num_y = []
        for y in years:
            num_y.append(get_year(y))
        num_y = np.array(num_y)
        # ohe = OneHotEncoder()
        # transformed = ohe.fit_transform(num_y.reshape(-1,1))
        transformed = oheData(num_y, oheFiles['year'])
        for n in range(len(transformed[0])):
            s = f'year_ohe{n + 1}'
            frame[s] = transformed[:, n]
    if oheTs:
        if isMultipleTS:
            # ohe = OneHotEncoder()
            # transformed = ohe.fit_transform(frame['timeshift'].values.reshape(-1,1))
            transformed = oheData(frame['timeshift'].values, oheFiles['ts'])
            for n in range(len(transformed[0])):
                s = f'ts_ohe{n + 1}'
                frame[s] = transformed[:, n]
    if withMicro:
        if oheMicro:
            # ohe = OneHotEncoder()
            # transformed = ohe.fit_transform(frame['micro'].values.reshape(-1,1))
            transformed = oheData(frame['micro'].values, oheFiles['micro'])
            for n in range(len(transformed[0])):
                s = f'micro_ohe{n + 1}'
                frame[s] = transformed[:, n]
    if dropColumns:
        frame = frame.drop('month', axis=1)
        frame = frame.drop('timeshift', axis=1)
        # One-hot encoding a single column
        frame = frame.drop(
            ['date', 'city', 'bookings_taken', 'bookings_auth'], axis=1)
        if withMicro:
            frame = frame.drop(['micro'], axis=1)
    if withCoupons:
        is_coupon = [0 for n in range(len(frame))]
        for idx, c in enumerate(frame['coupons_count'].values.astype(int)):
            if c > 0:
                is_coupon[idx] = 1
        frame['is_coupon'] = is_coupon

    return frame


def get_XYS(frame, seq_len, train_perc=0.95, isShuffled=True):
    seq = frame['orders_completed'].values.astype('int32')
    scaler = StandardScaler()
    scaler = scaler.fit(seq.reshape(-1, 1))
    standardized = scaler.transform(seq.reshape(-1, 1))
    frame['orders_completed'] = standardized
    seq = frame.values.astype('float64')
    X, y = split_sequence(seq, seq_len)
    y = y[:, 0]
    if isShuffled:
        ind_list = [i for i in range(X.shape[0])]
        shuffle(ind_list)
        X = X[ind_list]
        y = y[ind_list]

    X_train, y_train = X[:int(len(X) * train_perc)
                       ], y[:int(len(y) * train_perc)]
    X_test, y_test = X[-int(len(X) * (1 - train_perc)):], y[-int(len(y) * (1 - train_perc)):]
    return scaler, (X_train, y_train), (X_test, y_test)


def minMaxScale(frame, pos):
    seq = frame[pos].values.astype('float64')
    scaler = MinMaxScaler()
    scaler = scaler.fit(seq.reshape(-1, 1))
    minmax = scaler.transform(seq.reshape(-1, 1))
    frame[pos] = minmax
    return scaler


def standardScale(frame, pos):
    seq = frame[pos].values.astype('float64')
    scaler = StandardScaler()
    scaler = scaler.fit(seq.reshape(-1, 1))
    std = scaler.transform(seq.reshape(-1, 1))
    frame[pos] = std
    return scaler


def get_XY_fromseq(seq, seq_len, train_perc=0.95, isShuffled=True):
    X, y = split_sequence(seq, seq_len)
    y = y[:, 0]
    if isShuffled:
        ind_list = [i for i in range(X.shape[0])]
        shuffle(ind_list)
        X = X[ind_list]
        y = y[ind_list]

    X_train, y_train = X[:int(len(X) * train_perc)
                       ], y[:int(len(y) * train_perc)]
    X_test, y_test = X[-int(len(X) * (1 - train_perc)):], y[-int(len(y) * (1 - train_perc)):]
    return (X_train, y_train), (X_test, y_test)


def test_performance(X, y, model, n_steps, n_features, scaler):
    max = 0
    diffs = []
    correct = 0
    pos = 0
    for n in range(X.shape[0]):
        x_input = X[n]
        x_input = x_input.reshape((1, n_steps, n_features))
        # print(x_input.shape)
        yhat = model.predict(x_input)
        # print(yhat.shape)
        inversedyhat = np.rint(scaler.inverse_transform(yhat))
        # print(np.rint(inversed))
        inversed = np.rint(scaler.inverse_transform(y[n].reshape(-1, 1)))
        diff = np.rint(inversedyhat[0][0]) - inversed[0][0]
        print('predicted ', inversedyhat[0][0],
              ' real ', inversed[0][0], ' error ', diff)
        pos = n
        diffs.append(abs(diff))
        if abs(diff) > max:
            max = abs(diff)
        if abs(diff) <= 10:
            correct += 1

    accs = (correct) / (len(X)) * 100

    print('errore picco: ', max, ' media errori: ', np.mean(diffs), ' deviazione standard: ', np.std(diffs),
          'accuracy (10) ', accs)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def test_performance2(X, y, model, scaler):
    y_pred = model.predict(X)
    # accs = (correct)/(len(X))*100
    diff = scaler.inverse_transform(y_pred) - scaler.inverse_transform(y)
    abs_diff = np.abs(scaler.inverse_transform(
        y_pred) - scaler.inverse_transform(y))
    max = np.max(abs_diff)
    mean = np.mean(abs_diff)
    std = np.std(abs_diff)

    print('errore picco: ', max, ' media errori: ', mean,
          ' deviazione standard: ', std)  # ,'accuracy (10) ',accs)


#	url = f'http://history.openweathermap.org/data/2.5/history/city?lat=41.07493523509603&lon=14.341091479432484&type=hour&start=1648915200&end=1648936740&appid=62380bd96d3163426501ecf894f53917'
def get_history_weather(lat, lon, date, timeshift):
    t_start, t_end = date_to_ts(date, timeshift)
    URL = f'http://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={t_start}&end={t_end}&appid=62380bd96d3163426501ecf894f53917&units=metric'
    r = requests.get(url=URL)
    # print(r.json())
    return r



