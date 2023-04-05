import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from src.models import build_RNNGAT
from src.losses import custom_mse
from src import data
import argparse
import os
import pickle as pkl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--epochs", default=500)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    gpu = args.gpu
    profile = args.profile
    epochs = args.epochs

    tickers_path = os.path.join(os.getcwd(), "data", "Tickers")
    if gpu:
        physical_devices = tf.config.list_physical_devices('GPU')
        print(physical_devices[0])
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        physical_devices = tf.config.list_physical_devices('CPU')
        tf.config.set_visible_devices(physical_devices)

    processed = os.path.join(os.getcwd(), "data", "Processed")
    if not os.path.exists(os.path.join(processed, "time_series_matrix.npy")):
        df = data.tickers_df(tickers_path)
        df.to_csv(os.path.join(processed, "df.csv"))
        matrix, mapping = data.df_to_matrix(df)
        np.save(os.path.join(processed, "time_series_matrix.npy"), matrix)
        with open(os.path.join(processed, "mappings.pkl"), "wb") as file:
            pkl.dump(mapping, file)
    else:
        matrix = np.load(os.path.join(processed, "time_series_matrix.npy"))
        with open(os.path.join(processed, "mappings.pkl"), "rb") as file:
            maping = pkl.load(file)
    diff_matrix = data.diff_log(matrix)

    kwargs_cell = {"dropout": 0.01,
                   "activation": "relu",
                   "recurrent_dropout": 0.01,
                   "hidden_size_out": 15,
                   "regularizer": "l2",
                   "layer_norm": False,
                   "gatv2": True,
                   "concat_heads": False,
                   "return_attn_coef": False}
    model = build_RNNGAT(*diff_matrix.shape[1:], kwargs_cell=kwargs_cell)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001))  # ,loss=custom_mse,metrics=[custom_mse])

    a = np.ones(shape=(*diff_matrix.shape[:-1], diff_matrix.shape[-2]))
    optim = tf.keras.optimizers.Adam(0.001)
    loss_hist = []
    t0 = 0
    t1 = 40
    dt = 25

    profiler_dir = os.path.join(os.getcwd(), "Profiler")
    options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,
                                                       python_tracer_level=1,
                                                       device_tracer_level=1)


    def train(model, x, y, epochs=500, log_dir="./", profiler_options=None):
        @tf.function
        def step(x, y):
            with tf.GradientTape() as tape:
                # with tf.profiler.experimental.Trace("inference", step_num=i):
                o, p = model(x)
                l = tf.reduce_mean(
                    tf.keras.losses.mse(tf.reshape(y, (-1, 1)), tf.reshape(p, (-1, 1))))
            # with tf.profiler.experimental.Trace("train", step_num=i):
            grads = tape.gradient(l, model.trainable_variables)
            optim.apply_gradients(zip(grads, model.trainable_variables))
            return l

        # tf.profiler.experimental.start(log_dir, options=profiler_options)
        for i in range(epochs):
            l = step(x, y)
            loss_hist.append(l)
            tf.print(f"Epoch {i}: {l}")
        # tf.profiler.experimental.stop()


    train(model,
          x=[tf.constant(diff_matrix[t0:t1][None, :]), tf.constant(a[t0:t1][None, :])],
          y=tf.constant(diff_matrix[t0 + 1:t1 + 1, :, -2]),
          log_dir=profiler_dir,
          profiler_options=options)

    o, p = model.predict([tf.constant(diff_matrix[t0:t1][None, :]),
                          tf.constant(a[t0:t1][None, :])]
                         )
    plt.figure()
    plt.plot(np.log(loss_hist))
    plt.suptitle("Loss History")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    fig, ax = plt.subplots(10, 10)
    fig.suptitle("Predictions On TRAINING SET (Blue) vs True (Red)")
    cntr = 0
    for ir in range(10):
        for jc in range(10):
            ax[ir, jc].plot(diff_matrix[t0:t1, cntr, -2], color="red")
            ax[ir, jc].plot(p[0, :, cntr].flatten(), color="blue")
            cntr += 1

    o_test, p_test = model.predict([tf.constant(diff_matrix[t0 + dt:t1 + dt][None, :]),
                                    tf.constant(a[t0 + dt:t1 + dt][None, :])]
                                   )
    fig, ax = plt.subplots(10, 10)
    fig.suptitle("Predictions On TEST SET (Blue) vs True (Red)")
    cntr = 0
    for ir in range(10):
        for jc in range(10):
            ax[ir, jc].plot(diff_matrix[t0 + dt + 1:t1 + dt + 1, cntr, -2], color="red")
            ax[ir, jc].plot(p_test[0, :, cntr].flatten(), color="blue")
            cntr += 1
    plt.show()
