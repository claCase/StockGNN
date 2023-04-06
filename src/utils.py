import numpy as np
import tensorflow as tf
import os
import datetime


def train(model, x, y, epochs=500, log_dir="./", profiler_options=None):
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
    train_writer = tf.summary.create_file_writer(log_dir, "History")#os.path.join(log_dir, "History", str(datetime.datetime.now())[:-7]))
    tf.profiler.experimental.start(log_dir,#os.path.join(log_dir, str(datetime.datetime.now())[:-7]),
                                   options=profiler_options)
    for i in range(epochs):
        l = step(x, y)
        loss_hist.append(l)
        with train_writer.as_default(step=i):
            tf.summary.scalar("MeanSquaredError", l)
        tf.print(f"Epoch {i}: {l}")
        if i == 30:
            tf.profiler.experimental.stop()
    return loss_hist
