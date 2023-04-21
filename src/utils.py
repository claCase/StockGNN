import numpy as np
import tensorflow as tf
import os
import datetime


def train(model, x, y,
          x_val=None, y_val=None,
          epochs=500,
          log_dir="./",
          profiler_options=None):
    if not model._compile_was_called:
        optim = tf.keras.optimizers.Adam(0.001)

    @tf.function
    def step(x, y):
        with tf.GradientTape() as tape:
            with tf.profiler.experimental.Trace("inference", step_num=i):
                o, p = model(x)
            with tf.profiler.experimental.Trace("loss", step_num=i):
                l = tf.reduce_mean(
                    tf.keras.losses.mse(tf.reshape(y, (-1, 1)), tf.reshape(p, (-1, 1))))
        with tf.profiler.experimental.Trace("train", step_num=i):
            grads = tape.gradient(l, model.trainable_variables)
            if model._compile_was_called:
                model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            else:
                optim.apply_gradients(zip(grads, model.trainable_variables))
        return l

    loss_hist = []
    train_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, "History", "Train", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    validation_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, "History", "Validation", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    tf.profiler.experimental.start(os.path.join(log_dir, "Profiler"),
                                   options=profiler_options)
    for i in range(epochs):
        l = step(x, y)
        loss_hist.append(l)
        with train_writer.as_default(step=i):
            tf.summary.scalar("MeanSquaredError_train", l)
        if x_val is not None and y_val is not None:
            o_val, p_val = model.predict(x_val)
            l_val = tf.reduce_mean(
                tf.keras.losses.mse(tf.reshape(y_val, (-1, 1)), tf.reshape(p_val, (-1, 1))))
            with validation_writer.as_default(step=i):
                tf.summary.scalar("MeanSquaredError_test", l_val)
        tf.print(f"Epoch {i}: {l}")
        if i == 10:
            tf.profiler.experimental.stop()
    return loss_hist


def normalize_adj(adj, symmetric=True, power=-0.5, diagonal=False):
    '''
    Normalization of Adjacency Matrix
    :param adj: BatchxNxN
    :return: adj_normed: BatchxNxN
    '''
    if diagonal:
        adj_diag = tf.reduce_mean(adj, -1) * 0.5 + tf.reduce_mean(adj, -2) * 0.5
        adj_diag = tf.linalg.diag(adj_diag)
        adj = adj + adj_diag
    d = tf.reduce_sum(adj, -1)
    d_inv = tf.pow(d, power)
    d_inv = tf.where(tf.math.is_inf(d_inv), 0.0, d_inv)
    d_inv = tf.linalg.diag(d_inv)
    if symmetric:
        return tf.einsum("...ij, ...jk, ...ko -> ...io", d_inv, adj, d_inv)
    else:
        return tf.einsum("...ij, ...jk -> ...ik", d_inv, adj)


def node_degree(adj, symmetric=True):
    if symmetric:
        return tf.linalg.diag(tf.reduce_sum(adj, -1))
    else:
        return tf.linalg.diag(tf.reduce_sum(adj, -1), tf.reduce_sum(adj, -2))


def laplacian(adj):
    return node_degree(adj) - adj


def normalized_laplacian(adj, symmetric=True):
    I = tf.linalg.eye(adj.shape[-1], dtype=adj.dtype)[None, :]

    return I - normalize_adj(adj, symmetric)


def add_identity(adj):
    I = tf.linalg.eye(adj.shape[-1])[None, :]
    return adj + I


def power_adj_matrix(adj, power):
    powers = np.empty((power, *adj.shape))
    for i in range(power):
        powers[i, :, :] = np.linalg.matrix_power(adj, i)
    return powers


def laplacian_eigenvectors(adj, type="normalized", **kwargs):
    assert type in {"normalized", "standard"}
    if "symmetric" in kwargs.keys():
        symmetric = kwargs.symmetric
    else:
        symmetric = None

    if type == "normalized":
        lap = normalized_laplacian(adj, symmetric)
    else:
        lap = laplacian(adj)
    if isinstance(adj, tf.Tensor):
        vec, val = tf.linalg.eig(lap)
        return vec, val, lap
    if isinstance(adj, np.ndarray):
        val, vec = np.linalg.eig(lap)
        return val[0], vec[0], lap.numpy()


def outer_eigenvectors(eig):
    if isinstance(eig, tf.Tensor):
        outer = tf.einsum("...ki,...zi->...kzi", eig, eig)
        outer = tf.transpose(outer, perm=(1, 2, 0))
    elif isinstance(eig, np.ndarray):
        outer = np.einsum("...ki,...zi->...kzi", eig, eig)
        outer = np.swapaxes(outer, 2, 0)
    else:
        raise TypeError(f"Eigenvectors matrix of type {type(eig)} in not in types (tf.Tensor, np.ndarray)")

    return outer


def diff_log(matrix: np.array):
    t0 = np.nan_to_num(np.log(matrix[:-1]), nan=0.0, neginf=0.0)
    t1 = np.nan_to_num(np.log(matrix[1:]), nan=0.0, neginf=0.0)
    diff = t1 - t0
    return diff


def filter_value(matrix: np.array, value: float, axis=-1):
    filter_ = ~(matrix == value).all(axis=axis)
    return matrix[filter_], filter_
