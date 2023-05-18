from src.Data import data
from src.Modelling import models, losses
import numpy as np
import argparse
import matplotlib.pyplot as plt
from src.Modelling.utils import diff_log
import tensorflow as tf


def main(seq_len=110, len_train=110, len_test=500, epochs=200):
    x = np.load("../../../data/Processed/time_series_matrix.npy")
    N = x.shape[1]
    x_diff = diff_log(x)
    x_diff = np.maximum(np.minimum(x_diff, 1), -1)
    x_train = x_diff[:len_train]
    a_train = np.ones(shape=(len_train, N, N))
    x_test = x_diff[len_train+1:len_train+len_test+1]
    a_test = np.ones(shape=(len_test, N, N))
    y_train = x_diff[1:len_train+1, :, -2]
    y_test = x_diff[len_train+1:len_train+len_test+1, :, -2]
    dg_train = data.TimeSeriesBatchGenerator((x_train, a_train), y_train, sequence_len=seq_len)
    dg_test = data.TimeSeriesBatchGenerator((x_test, a_test), y_test, sequence_len=seq_len)
    model = models.RNNGAT(x_diff.shape[1], 0.1, 0.1, 4, 15, 1, stateful=True)
    o, p = model((x_test[None, :], a_test[None, :]))
    model.compile(loss=losses.custom_mse)
    history = model.fit(dg_train, validation_data=dg_test, epochs=epochs)
    _, p_train = model.predict(dg_train)
    _, p_test = model.predict(dg_test)

    for i in range(1, 10 * 10):
        plt.subplot(10, 10, i)
        plt.plot(p_train[0, :100,  i, 0], color="red")
        plt.plot(y_train[:100, i], color="blue")

    plt.figure()
    for i in range(1, 10 * 10):
        plt.subplot(10, 10, i)
        plt.plot(p_test[0, :100,  i, 0], color="red")
        plt.plot(y_test[:100, i], color="blue")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seq_len", type=int, default=110)
    parser.add_argument("--len_train", type=int, default=300)
    parser.add_argument("--len_test", type=int, default=300)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    epochs = args.epochs
    seq_len = args.seq_len
    len_train = args.len_train
    len_test = args.len_test
    gpu = args.gpu

    if gpu:
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            print(physical_devices[0])
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except Exception as e:
            raise e
    else:
        physical_devices = tf.config.list_physical_devices('CPU')
        tf.config.set_visible_devices(physical_devices)

    main(seq_len, len_train, len_test, epochs)
