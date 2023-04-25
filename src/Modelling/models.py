import tensorflow as tf
from src.Modelling.layers import NestedGRUGATCell, NestedGRUAttentionCell, NestedGRUGATCellSingle, GATv2Layer
from src.Modelling.losses import custom_mse

# from src.losses import custom_mse

m = tf.keras.models
l = tf.keras.layers
act = tf.keras.activations


def build_RNNGAT(nodes,
                 input_features,
                 *args,
                 kwargs_cell={"dropout": 0.01,
                              "activation": "relu",
                              "recurrent_dropout": 0.01,
                              "hidden_size_out": 15,
                              "regularizer": "l2",
                              "layer_norm": False,
                              "gatv2": True,
                              "concat_heads": False,
                              "return_attn_coef": False},
                 kwargs_out={"units": 1, "activation": "tanh"},
                 single=True):
    i1 = tf.keras.Input(shape=(None, nodes, input_features), batch_size=1)
    i2 = tf.keras.Input(shape=(None, nodes, nodes), batch_size=1)
    if single:
        cell = NestedGRUGATCellSingle(nodes, *args, **kwargs_cell)
    else:
        cell = NestedGRUGATCell(nodes, *args, **kwargs_cell)
    rnn = l.RNN(cell, return_sequences=True, return_state=True)
    pred_layer = l.Dense(**kwargs_out)
    # pred_layer = l.Dense(1, **kwargs_out)
    o, h = rnn((i1, i2))
    if kwargs_cell.get("retun_attn_coef"):
        p = pred_layer(o[0])
    else:
        p = pred_layer(o)
    model = tf.keras.models.Model([i1, i2], [o, p])
    return model


def build_RNNAttention(nodes,
                       input_features,
                       kwargs_cell={"dropout": 0.01,
                                    "activation": "relu",
                                    "recurrent_dropout": 0.01,
                                    "hidden_size_out": 15,
                                    "regularizer": "l2",
                                    "layer_norm": False},
                       kwargs_out={"activation": "tanh"}):
    i1 = tf.keras.Input(shape=(None, nodes, input_features), batch_size=None)
    i2 = tf.keras.Input(shape=(None, nodes, nodes), batch_size=None)
    cell = NestedGRUAttentionCell(nodes, **kwargs_cell)
    rnn = l.RNN(cell, return_sequences=True, return_state=True)
    pred_layer = l.Dense(1, **kwargs_out)
    o, h = rnn((i1, i2))
    p = pred_layer(o)
    return tf.keras.models.Model([i1, i2], [o, p])


class RNNGAT(m.Model):
    def __init__(self,
                 nodes,
                 dropout,
                 recurrent_dropout,
                 attn_heads,
                 channels,
                 out_channels,
                 concat_heads=False,
                 add_bias=True,
                 activation="relu",
                 output_activation="tanh",
                 regularizer=None,
                 return_attn_coef=False,
                 layer_norm=False,
                 initializer="glorot_normal",
                 gatv2=True,
                 single_gnn=True,
                 return_sequences=True,
                 return_state=True,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 time_major=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.nodes = nodes
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.attn_heads = attn_heads
        self.channels = channels
        self.concat_heads = concat_heads
        self.add_bias = add_bias
        self.activation = activation
        self.output_activation = output_activation
        self.regularizer = regularizer
        self.return_attn_coef = return_attn_coef
        self.layer_norm = layer_norm
        self.initializer = initializer
        self.gatv2 = gatv2
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.time_major = time_major
        self.single_gnn = single_gnn
        self.out_channels = out_channels

    def build(self, input_shape):
        x, a = input_shape

        if self.single_gnn:
            rnn_cell = NestedGRUGATCellSingle(
                nodes=self.nodes,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                attn_heads=self.attn_heads,
                channels=self.channels,
                concat_heads=self.concat_heads,
                add_bias=self.add_bias,
                activation=self.activation,
                regularizer=self.regularizer,
                return_attn_coef=self.return_attn_coef,
                layer_norm=self.layer_norm,
                initializer=self.initializer,
                gatv2=self.gatv2
            )
        else:
            rnn_cell = NestedGRUGATCell(nodes=self.nodes,
                                        dropout=self.dropout,
                                        recurrent_dropout=self.recurrent_dropout,
                                        attn_heads=self.attn_heads,
                                        channels=self.channels,
                                        concat_heads=self.concat_heads,
                                        add_bias=self.add_bias,
                                        activation=self.activation,
                                        regularizer=self.regularizer,
                                        return_attn_coef=self.return_attn_coef,
                                        layer_norm=self.layer_norm,
                                        initializer=self.initializer,
                                        gatv2=self.gatv2
                                        )

        self.rnn = l.RNN(rnn_cell,
                         return_sequences=self.return_sequences,
                         return_state=self.return_state,
                         go_backwards=self.go_backwards,
                         stateful=self.stateful,
                         unroll=self.unroll,
                         time_major=self.time_major)
        self.rnn.build((x, a))
        rnn_out_shape = self.rnn.compute_output_shape((x, a))
        self.pred_out = l.Dense(self.out_channels, self.output_activation)
        self.pred_out.build(rnn_out_shape[0])

    @tf.function
    def call(self, inputs, states=None, training=None, mask=None):
        x, a = inputs
        o, h = self.rnn((x, a), initial_state=states, training=training)
        if self.return_attn_coef:
            p = self.pred_out(o[0])
        else:
            p = self.pred_out(o)
        return o, p

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            o, y_pred = self(x, training=True)
            l = self.compute_loss(y=y, y_pred=y_pred)
        grads = tape.gradient(l, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(l)
        return {"Training Loss": self.loss_tracker.result()}

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        return self.compiled_loss(y, y_pred)

    def test_step(self, data):
        x, y = data
        o, y_pred = self(x)
        loss = self.compiled_loss(y, y_pred)
        return {"Test Loss": loss}

    def get_config(self):
        config = {
            "nodes": self.nodes,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "attn_heads": self.attn_heads,
            "channels": self.channels,
            "concat_heads": self.concat_heads,
            "add_bias": self.add_bias,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "regularizer": self.regularizer,
            "return_attn_coef": self.return_attn_coef,
            "layer_norm": self.layer_norm,
            "initializer": self.initializer,
            "gatv2": self.gatv2,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "go_backwards": self.go_backwards,
            "stateful": self.stateful,
            "unroll": self.unroll,
            "time_major": self.time_major,
            "single_gnn": self.single_gnn,
            "out_channels": self.out_channels
        }
        config.update(tf.keras.utils.serialize_keras_object(self.pred_out))
        config.update(tf.keras.utils.serialize_keras_object(self.rnn))
        config["loss"] = tf.keras.losses.serialize(self.loss)
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def from_config(cls, config, custom_objects=None):
        return cls(**config)

    def get_model(self, input_features):
        i1 = tf.keras.Input(shape=(None, self.nodes, input_features))
        i2 = tf.keras.Input(shape=(None, self.nodes, self.nodes))
        o, h = self.rnn((i1, i2))
        if self.return_attn_coef:
            p = self.pred_out(o[0])
        else:
            p = self.pred_out(o)
        return m.Model((i1, i2), (o, p))


class GATv2Model(m.Model):
    def __init__(self, hidden_sizes=[10, 10, 7], channels=[10, 10, 7], heads=[10, 10, 1], dropout=0.2, activation="elu",
                 *args,
                 **kwargs):
        super(GATv2Model, self).__init__(**kwargs)
        self.channels = channels
        self.hidden_sizes = hidden_sizes
        self.heads = heads
        self.dropout = l.Dropout(dropout)
        assert len(channels) == len(heads) == len(hidden_sizes)
        self.n_layers = len(hidden_sizes)
        self.activation = activation
        self.gat_layers = [GATv2Layer(heads=hh, channels=c, activation=activation, *args) for c, hh in
                           zip(self.channels, self.heads)]
        self.dense_layers = [l.Dense(h, activation) for h in self.hidden_sizes]

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        for d, g in zip(self.dense_layers, self.gat_layers):
            x = d(x)
            x = self.dropout(x)
            x = g([x, a])
        return x


if __name__ == "__main__":
    import numpy as np

    B = 1
    N = 100
    f = 14
    T = 50
    x = np.random.normal(size=(B, T, N, f))
    a = np.random.normal(size=(B, T, N, N))


    @tf.function
    def f():
        tf.compat.v1.disable_eager_execution()
        model = RNNGAT(nodes=N,
                       dropout=0.1,
                       recurrent_dropout=0.1,
                       attn_heads=4,
                       channels=6,
                       out_channels=1,
                       concat_heads=False,
                       add_bias=True,
                       activation="relu",
                       output_activation="tanh",
                       regularizer=None,
                       return_attn_coef=True,
                       layer_norm=False,
                       initializer="glorot_normal",
                       gatv2=True,
                       single_gnn=True,
                       return_sequences=True,
                       return_state=True,
                       go_backwards=False,
                       stateful=False,
                       unroll=False,
                       time_major=False,
                       mincut=5)

        o, p = model([x, a], training=True)
        model.compile(loss=custom_mse)
        model.fit([x[:, :10], a[:, :10]], x[:, :10, :, 0])


    f()
