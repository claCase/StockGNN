import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from src.Modelling.models import build_RNNGAT
from src.Modelling.utils import train, diff_log
from src.Data import data
import argparse
import os
import pickle as pkl


def main(gpu=True,
         epochs=200,
         save=False,
         show=True
         ):
    tickers_path = os.path.join(os.getcwd(), "../../../data", "Tickers")
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

    processed = os.path.join(os.getcwd(), "../../../data", "Processed")
    if not os.path.exists(os.path.join(processed, "time_series_matrix.npy")):
        df = data.tickers_df(tickers_path)
        df.to_csv(os.path.join(processed, "df.csv"))
        matrix, mapping, columns = data.df_to_matrix(df, save_path=processed, name="time_series")
    else:
        matrix = np.load(os.path.join(processed, "time_series_matrix.npy"))
        with open(os.path.join(processed, "time_series_index_mappings.pkl"), "rb") as file:
            mapping = pkl.load(file)
        with open(os.path.join(processed, "time_series_columns_mapping.pkl"), "rb") as file:
            columns = pkl.load(file)
        with open(os.path.join(processed, "time_series_unique_index.pkl"), "rb") as file:
            idx_mapping = pkl.load(file)

    diff_matrix = diff_log(matrix)

    kwargs_cell = {"dropout": 0.01,
                   "activation": "relu",
                   "recurrent_dropout": 0.01,
                   "regularizer": "l2",
                   "layer_norm": False,
                   "gatv2": True,
                   "concat_heads": False,
                   "return_attn_coef": False,
                   "attn_heads": 4,
                   "channels": 15}
    model = build_RNNGAT(*diff_matrix.shape[1:], kwargs_cell=kwargs_cell)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001))

    a = np.ones(shape=(*diff_matrix.shape[:-1], diff_matrix.shape[-2]))
    t0 = 0
    t1 = 60
    dt = 60

    profiler_dir = os.path.join(os.getcwd(), "Analysis", "Functional")
    options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,
                                                       python_tracer_level=1,
                                                       device_tracer_level=1)
    loss_hist = train(model,
                      x=[tf.constant(diff_matrix[t0:t1][None, :]), tf.constant(a[t0:t1][None, :])],
                      y=tf.constant(diff_matrix[t0 + 1:t1 + 1, :, -2]),
                      x_val=[tf.constant(diff_matrix[t0 + dt:t1 + dt][None, :]),
                             tf.constant(a[t0 + dt:t1 + dt][None, :])],
                      y_val=diff_matrix[t0 + dt + 1:t1 + dt + 1, :, -2],
                      epochs=epochs,
                      log_dir=profiler_dir,
                      profiler_options=options)
    if show:

        plt.figure()
        plt.plot(np.log(loss_hist["Training Loss"]), label="Traning Loss")
        plt.plot(np.log(loss_hist["Test Loss"]), label="Test Loss")
        plt.legend()
        plt.suptitle("Loss History")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")

        o, p = model.predict([tf.constant(diff_matrix[t0:t1][None, :]),
                              tf.constant(a[t0:t1][None, :])]
                             )
        fig, ax = plt.subplots(10, 10)
        fig.suptitle("Predictions On TRAINING SET", fontsize=30)
        cntr = 0
        for ir in range(10):
            for jc in range(10):
                ax[ir, jc].plot(diff_matrix[t0 + 1:t1 + 1, cntr, -2], color="red", label="True")
                ax[ir, jc].plot(p[0, :, cntr].flatten(), color="blue", label="Predicted")
                ax[ir, jc].legend(loc="best")
                cntr += 1

        o_test, p_test = model.predict([tf.constant(diff_matrix[t0 + dt:t1 + dt][None, :]),
                                        tf.constant(a[t0 + dt:t1 + dt][None, :])]
                                       )
        fig, ax = plt.subplots(10, 10)
        fig.suptitle("Predictions On TEST SET", fontsize=30)
        cntr = 0
        for ir in range(10):
            for jc in range(10):
                ax[ir, jc].plot(diff_matrix[t0 + dt + 1:t1 + dt + 1, cntr, -2], color="red", label="True")
                ax[ir, jc].plot(p_test[0, :, cntr].flatten(), color="blue", label="Predicted")
                ax[ir, jc].legend(loc="best")
                cntr += 1
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    gpu = args.gpu
    epochs = args.epochs
    save = args.save
    show = args.show

    main(gpu=gpu, epochs=epochs, show=show, save=save)
