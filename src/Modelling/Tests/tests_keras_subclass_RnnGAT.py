import os
from src.Data.data import TimeSeriesBatchGenerator, to_multiindex, StockTimeSeries
from src.Modelling import models, losses
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
import io
from datetime import datetime, timedelta


def plot_preds(preds, true, name: str, dates=None,  tickers=None, interval=10) -> plt.figure:
    if dates is not None:
        assert isinstance(dates[0], datetime) or isinstance(dates[0], pd.Timestamp)
    fig, ax = plt.subplots(10, 10, figsize=(45, 20))
    fig.suptitle(name)
    counter = 0
    t, n = preds.shape
    for ir in range(10):
        for jc in range(10):
            if dates is not None:
                ax[ir, jc].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax[ir, jc].xaxis.set_major_locator(mdates.DayLocator(interval=t // interval))
                ax[ir, jc].plot(dates[-len(preds):],
                                true[-len(preds):, counter].flatten(),
                                color="red", label="True")
                ax[ir, jc].plot(dates[-len(preds):],
                                preds[-len(preds):, counter].flatten(),
                                color="blue", label="Predicted")
                ax[ir, jc].tick_params("x", labelrotation=30)
            else:
                ax[ir, jc].plot(true[:, counter].flatten(), color="red", label="True")
                ax[ir, jc].plot(preds[:, counter].flatten(), color="blue", label="Predicted")
            if tickers is not None:
                ax[ir, jc].set_title(tickers[counter])
            ax[ir, jc].legend(loc="upper left")
            counter += 1
    plt.tight_layout()
    return fig, ax


def plot_train_test_callback(model: tf.keras.Model,
                             train_series: TimeSeriesBatchGenerator,
                             test_series: TimeSeriesBatchGenerator,
                             log_dir: os.path,
                             train_dates=None,
                             test_dates=None,
                             tickers=None):
    file_writer = tf.summary.create_file_writer(os.path.join(log_dir, "Train Evolution Plot"))
    buffer_test = io.BytesIO()
    buffer_train = io.BytesIO()

    def on_epoch_end(epoch, log):
        _, p_train = model.predict(train_series)
        _, p_test = model.predict(test_series)
        true_train = train_series.y
        true_test = test_series.y
        fig_train, ax_train = plot_preds(p_train[0, :, :, 0], true_train[:, :, 0], "Training", train_dates, tickers)
        fig_test, ax_test = plot_preds(p_test[0, :, :, 0], true_test[:, :, 0], "Test", test_dates, tickers)
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


def main(seq_len=60, len_train=160, len_test=160, epochs=200):
    df = pd.read_csv("../../../data/Processed/df.csv")
    df = to_multiindex(df)
    st = StockTimeSeries(df, "NYSE")
    start = datetime.fromisoformat("2004-09-24")
    end_train = start + timedelta(days=len_train)
    start_test = end_train + timedelta(days=1)
    end = start_test + timedelta(days=len_test)
    st = st[start:end]
    st.remove_non_business_days(inplace=True)
    st.ptc_change(inplace=True)
    (dg_train, train_dates), (dg_test, test_dates), _ = st.to_time_series_batch_generator(start,
                                                                                          end_train,
                                                                                          start_test,
                                                                                          end,
                                                                                          seq_len=seq_len, adj=True)

    tickers = st.stocks
    n_tickers = len(tickers)
    model = models.RNNGAT(n_tickers, 0.1, 0.1, 4, 15, 1, stateful=True)
    #model = model.get_model(5, batch_size=1)
    o, p = model.predict(dg_train)  # build model and initialize weights
    model.compile(loss=losses.custom_mse, optimizer="adam")  # compile loss
    log_dir = os.path.join(os.getcwd(), "Analysis", "Subclass", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir, update_freq=1, profile_batch='10, 15')
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=plot_train_test_callback(model,
                                                                                          dg_train,
                                                                                          dg_test,
                                                                                          log_dir,
                                                                                          train_dates,
                                                                                          test_dates,
                                                                                          tickers))
    history = model.fit(dg_train, validation_data=dg_test, epochs=epochs, callbacks=[tb_callback, cm_callback])

    plt.figure()
    plt.plot(history.history["Training Loss"], color="blue", label="Training Loss")
    plt.plot(history.history["val_Test Loss"], color="red", label="Validation Loss")
    plt.legend()
    plt.show()

    _, p_train = model.predict(dg_train)
    _, p_test = model.predict(dg_test)

    y_train = dg_train.x[1]
    for i in range(1, 10 * 10):
        plt.subplot(10, 10, i)
        plt.plot(p_train[0, :100, i, 0], color="red", label="Prediction")
        plt.plot(y_train[:100, i], color="blue", labels="True")

    plt.figure()
    y_test = dg_test.x[1]
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
