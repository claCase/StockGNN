import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from src.models import build_RNNGAT
from src import data
import argparse
import os
import pickle as pkl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    gpu = args.gpu
    tickers_path = os.path.join(os.getcwd(), "data", "Tickers")
    if gpu:
        physical_devices = tf.config.list_physical_devices('GPU')
        print(physical_devices[0] )
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
    a = np.ones(shape=(*diff_matrix.shape[:-1], diff_matrix.shape[-2]))
    optim = tf.keras.optimizers.Adam(0.001)
    loss_hist = []
    for i in range(500):
        with tf.GradientTape() as tape:
            o, p = model([tf.constant(diff_matrix[:20][None, :]), tf.constant(a[:20][None, :])])
            l = tf.reduce_mean(tf.keras.losses.mse(diff_matrix[1:21, :, -2].reshape(-1, 1), tf.reshape(p, (-1, 1))))
            loss_hist.append(l)
        if not i % 200:
            fig, ax = plt.subplots(10, 10)
            cntr = 0
            for ir in range(10):
                for jc in range(10):
                    ax[ir, jc].plot(diff_matrix[1:21, cntr, -2], color="red")
                    ax[ir, jc].plot(p[0, :, cntr].numpy().flatten(), color="blue")
                    cntr += 1
            plt.show()
        grads = tape.gradient(l, model.trainable_variables)
        optim.apply_gradients(zip(grads, model.trainable_variables))
        print(f"loss {i}: {l.numpy()}")

    plt.figure()
    plt.plot(loss_hist)
    plt.suptitle("Loss History")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    fig, ax = plt.subplots(10, 10)
    fig.suptitle("Predictions (Blue) vs True (Red)")
    cntr = 0
    for ir in range(10):
        for jc in range(10):
            ax[ir, jc].plot(diff_matrix[1:21, cntr, -2], color="red")
            ax[ir, jc].plot(p[0, :, cntr].numpy().flatten(), color="blue")
            cntr += 1
    plt.show()