import os
from src.Data.data import TimeSeriesBatchGenerator
from src.Modelling import models, losses
import numpy as np
import argparse
import matplotlib.pyplot as plt
from src.Modelling.utils import diff_log
import tensorflow as tf
import io
from datetime import datetime


def plot_preds(preds, true, name: str) -> plt.figure:
    fig, ax = plt.subplots(10, 10, figsize=(35, 20))
    fig.suptitle(name)
    counter = 1
    for ir in range(10):
        for jc in range(10):
            ax[ir, jc].plot(true[:, counter].flatten(), color="red", label="True")
            ax[ir, jc].plot(preds[0, :, counter].flatten(), color="blue", label="Predicted")
            ax[ir, jc].legend(loc="best")
            counter += 1
    return fig


def plot_train_test_callback(model: tf.keras.Model,
                             train_series: TimeSeriesBatchGenerator,
                             test_series: TimeSeriesBatchGenerator,
                             log_dir: os.path):
    file_writer = tf.summary.create_file_writer(os.path.join(log_dir, "Train Evolution Plot"))
    buffer_test = io.BytesIO()
    buffer_train = io.BytesIO()

    def on_epoch_end(epoch, log):
        _, p_train = model.predict(train_series)
        _, p_test = model.predict(test_series)

        true_train = train_series.y
        true_test = test_series.y
        fig_train = plot_preds(p_train, true_train, "Training")
        fig_test = plot_preds(p_test, true_test, "Test")
        fig_train.savefig(buffer_train, format="png")
        fig_test.savefig(buffer_test, format="png")
        # plt.show()
        plt.close(fig_test)
        plt.close(fig_train)

        buffer_train.seek(0)
        buffer_test.seek(0)
        train_image = tf.image.decode_png(buffer_train.getvalue(), channels=4)
        train_image = tf.expand_dims(train_image, 0)
        test_image = tf.image.decode_png(buffer_test.getvalue(), channels=4)
        test_image = tf.expand_dims(test_image, 0)

        with file_writer.as_default():
            tf.summary.image(f"Training Predictions", train_image, step=epoch)
            tf.summary.image(f"Test Predictions", test_image, step=epoch)
        buffer_train.flush()
        buffer_test.flush()

    return on_epoch_end


def main(seq_len=60, len_train=60, len_test=60, epochs=200):
    x = np.load("../../../data/Processed/time_series_matrix.npy")
    N = x.shape[1]
    x_diff = diff_log(x)
    x_diff = np.maximum(np.minimum(x_diff, 1), -1)
    x_train = x_diff[:len_train]
    a_train = np.ones(shape=(len_train, N, N))
    x_test = x_diff[len_train:len_train + len_test]
    a_test = np.ones(shape=(len_test, N, N))
    y_train = x_diff[1:len_train + 1, :, -2]
    y_test = x_diff[len_train + 1:len_train + len_test + 1, :, -2]
    dg_train = TimeSeriesBatchGenerator((x_train, a_train), y_train, sequence_len=seq_len)
    dg_test = TimeSeriesBatchGenerator((x_test, a_test), y_test, sequence_len=seq_len)
    model = models.RNNGAT(x_diff.shape[1], 0.1, 0.1, 4, 15, 1, stateful=True)
    o, p = model((x_test[None, :], a_test[None, :]))
    model.compile(loss=losses.custom_mse)
    log_dir = os.path.join(os.getcwd(), "Analysis", "Subclass", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir, update_freq=1, profile_batch='10, 15')
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=plot_train_test_callback(model,
                                                                                          dg_train,
                                                                                          dg_test,
                                                                                          log_dir))
    history = model.fit(dg_train, validation_data=dg_test, epochs=epochs, callbacks=[tb_callback, cm_callback])

    plt.figure()
    plt.plot(history.history["Training Loss"], color="blue", label="Training Loss")
    plt.plot(history.history["val_Test Loss"], color="red", label="Validation Loss")
    plt.legend()
    plt.show()

    _, p_train = model.predict(dg_train)
    _, p_test = model.predict(dg_test)

    for i in range(1, 10 * 10):
        plt.subplot(10, 10, i)
        plt.plot(p_train[0, :100, i, 0], color="red", label="Prediction")
        plt.plot(y_train[:100, i], color="blue", labels="True")

    plt.figure()
    for i in range(1, 10 * 10):
        plt.subplot(10, 10, i)
        plt.plot(p_test[0, :100, i, 0], color="red", label="Prediction")
        plt.plot(y_test[:100, i], color="blue", label="True")
    plt.legend(loc="best")

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
