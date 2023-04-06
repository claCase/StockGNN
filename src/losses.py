import tensorflow as tf

klos = tf.keras.losses

def custom_mse(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, 1))
    y_pred = tf.reshape(y_pred, (-1, 1))
    return tf.reduce_mean(klos.mse(y_true, y_pred))