import tensorflow as tf
from src.layers import NestedGRUGATCell, NestedGRUAttentionCell, GATv2Layer

m = tf.keras.models
l = tf.keras.layers


def build_RNNGAT(nodes,
                 f,
                 kwargs_cell={"dropout": 0.01,
                              "activation": "relu",
                              "recurrent_dropout": 0.01,
                              "hidden_size_out": 15,
                              "regularizer": "l2",
                              "layer_norm": False,
                              "gatv2": True,
                              "concat_heads": False,
                              "return_attn_coef": False},
                 kwargs_out={"activation": "tanh"}
                 ):
    i1 = tf.keras.Input(shape=(None, nodes, f), batch_size=1)
    i2 = tf.keras.Input(shape=(None, nodes, nodes), batch_size=1)
    cell = NestedGRUGATCell(nodes, **kwargs_cell)
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)
    pred_layer = tf.keras.layers.Dense(1, **kwargs_out)
    o, h = rnn((i1, i2))
    if kwargs_cell.get("retun_attn_coef"):
        p = pred_layer(o[0])
    else:
        p = pred_layer(o)
    return tf.keras.models.Model([i1, i2], [o, p])


def build_RNNAttention(nodes,
                       out,
                       kwargs_cell={"dropout": 0.01,
                                    "activation": "relu",
                                    "recurrent_dropout": 0.01,
                                    "hidden_size_out": 15,
                                    "regularizer": "l2",
                                    "layer_norm": False},
                       kwargs_out={"activation": "tanh"}):
    i1 = tf.keras.Input(shape=(None, nodes, out), batch_size=1)
    i2 = tf.keras.Input(shape=(None, nodes, nodes), batch_size=1)
    cell = NestedGRUAttentionCell(nodes, **kwargs_cell)
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)
    pred_layer = tf.keras.layers.Dense(1, **kwargs_out)
    o, h = rnn((i1, i2))
    p = pred_layer(h)
    return tf.keras.models.Model([i1, i2], [o, p])


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
