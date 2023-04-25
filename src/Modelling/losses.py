import tensorflow as tf
from src.Modelling.utils import normalize_adj, node_degree

klos = tf.keras.losses


def custom_mse(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, 1))
    y_pred = tf.reshape(y_pred, (-1, 1))
    return tf.reduce_mean(klos.mse(y_true, y_pred))


def min_cut(A, S, normalize=True):
    """
    S: NxC
    A: NxN
    """
    if normalize:
        A = normalize_adj(A, symmetric=True)
    D = node_degree(A, symmetric=True)
    num = tf.linalg.trace(tf.einsum("...ji,...jk,...kl->...il", S, A, S))
    den = tf.linalg.trace(tf.einsum("...ji,...jk,...kl->...il", S, D, S))
    mc = -num / den  # mincut loss

    k = tf.cast(S.shape[-1], S.dtype)
    I = tf.eye(k, dtype=S.dtype) / tf.sqrt(k)
    innerS = tf.einsum("...ji,...jk->...ik", S, S)
    normS = tf.linalg.norm(innerS, axis=(-1, -2))
    ort = tf.linalg.norm(innerS / normS[..., None, None] - I, axis=(-1, -2))  # orthogonality constraint

    return mc + ort
