import tensorflow as tf
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, _config_for_enable_caching_device, \
    _caching_device
from spektral.layers import GATConv
from spektral.layers.ops import unsorted_segment_softmax

l = tf.keras.layers
act = tf.keras.activations
init = tf.keras.initializers


class NestedGRUGATCell(DropoutRNNCellMixin, l.Layer):
    def __init__(self,
                 nodes,
                 dropout,
                 recurrent_dropout,
                 hidden_size_out,
                 activation,
                 regularizer=None,
                 layer_norm=False,
                 gatv2=False,
                 attn_heads=4,
                 return_attn_coef=False,
                 concat_heads=False,
                 **kwargs):
        super(NestedGRUGATCell, self).__init__(**kwargs)
        self.tot_nodes = nodes
        self.hidden_size_out = hidden_size_out
        self.activation = activation
        self.recurrent_dropout = recurrent_dropout
        self.dropout = dropout
        self.regularizer = regularizer
        self.layer_norm = layer_norm
        self.return_attn_coef = return_attn_coef
        self.state_size = tf.TensorShape((self.tot_nodes, self.hidden_size_out))
        if return_attn_coef:
            self.output_size = [tf.TensorShape((self.tot_nodes, self.hidden_size_out)),
                                tf.TensorShape((attn_heads, self.tot_nodes, self.tot_nodes)),
                                tf.TensorShape((attn_heads, self.tot_nodes, self.tot_nodes)),
                                tf.TensorShape((attn_heads, self.tot_nodes, self.tot_nodes))
                                ]
        else:
            self.output_size = tf.TensorShape((self.tot_nodes, self.hidden_size_out))

        if self.layer_norm:
            self.ln = l.LayerNormalization()
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        if gatv2:
            gat = GATv2Layer
        else:
            gat = GATConv
        self.gat_u = gat(channels=hidden_size_out, attn_heads=4, concat_heads=concat_heads, dropout_rate=0,
                         activation=self.activation, return_attn_coef=return_attn_coef)
        self.gat_r = gat(channels=hidden_size_out, attn_heads=4, concat_heads=concat_heads, dropout_rate=0,
                         activation=self.activation, return_attn_coef=return_attn_coef)
        self.gat_c = gat(channels=hidden_size_out, attn_heads=4, concat_heads=concat_heads, dropout_rate=0,
                         activation=self.activation, return_attn_coef=return_attn_coef)

    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        self.b_u = self.add_weight(shape=(self.hidden_size_out,), initializer=init.Zeros, name="b_u",
                                   regularizer=self.regularizer, caching_device=default_caching_device)
        self.b_r = self.add_weight(shape=(self.hidden_size_out,), initializer=init.Zeros, name="b_r",
                                   regularizer=self.regularizer, caching_device=default_caching_device)
        self.b_c = self.add_weight(shape=(self.hidden_size_out,), initializer=init.zeros, name="b_c",
                                   regularizer=self.regularizer, caching_device=default_caching_device)
        self.W_u = self.add_weight(shape=(self.hidden_size_out, self.hidden_size_out),
                                   initializer="he_normal", name="W_u_p",
                                   regularizer=self.regularizer, caching_device=default_caching_device)
        self.W_r = self.add_weight(shape=(self.hidden_size_out, self.hidden_size_out),
                                   initializer="he_normal", name="W_r_p",
                                   regularizer=self.regularizer, caching_device=default_caching_device)
        self.W_c = self.add_weight(shape=(self.hidden_size_out, self.hidden_size_out),
                                   initializer="he_normal", name="W_c_p",
                                   regularizer=self.regularizer, caching_device=default_caching_device)

    def call(self, inputs, states, training, *args, **kwargs):
        x, a = tf.nest.flatten(inputs)
        h = states[0]
        if 0 < self.recurrent_dropout < 1:
            h_mask = self.get_recurrent_dropout_mask_for_cell(inputs=h, training=training, count=1)
            h = h * h_mask
        if 0 < self.dropout < 1:
            x_mask = self.get_dropout_mask_for_cell(inputs=x, training=training, count=1)
            x = x * x_mask

        if self.return_attn_coef:
            u_gat, u_attn = self.gat_u([tf.concat([x, h], -1), a])
            r_gat, r_attn = self.gat_r([tf.concat([x, h], -1), a])
        else:
            u_gat = self.gat_u([tf.concat([x, h], -1), a])
            r_gat = self.gat_r([tf.concat([x, h], -1), a])
        u = tf.nn.sigmoid(self.b_u + u_gat @ self.W_u)
        r = tf.nn.sigmoid(self.b_r + r_gat @ self.W_r)
        if self.return_attn_coef:
            c_gat, c_attn = self.gat_c([tf.concat([x, r * h], -1), a])
        else:
            c_gat = self.gat_c([tf.concat([x, r * h], -1), a])
        c = tf.nn.tanh(self.b_c + c_gat @ self.W_c)
        h_prime = u * h + (1 - u) * c
        if self.layer_norm:
            h_prime = self.ln(h_prime)
        if self.return_attn_coef:
            return (h_prime, u_attn, r_attn, c_attn), h_prime
        else:
            return h_prime, h_prime

    def get_config(self):
        config = {"nodes": self.tot_nodes,
                  "hidden_size_out": self.hidden_size_out,
                  "dropout": self.dropout,
                  "recurrent_dropout": self.recurrent_dropout,
                  "regularizer": self.regularizer
                  }
        config.update(_config_for_enable_caching_device(self))
        base_config = super(NestedGRUGATCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NestedGRUAttentionCell(DropoutRNNCellMixin, l.Layer):
    def __init__(self,
                 nodes,
                 dropout,
                 recurrent_dropout,
                 hidden_size_out,
                 activation,
                 regularizer=None,
                 layer_norm=False,
                 concat_heads=False,
                 **kwargs):
        super(NestedGRUAttentionCell, self).__init__(**kwargs)
        self.tot_nodes = nodes
        self.hidden_size_out = hidden_size_out
        self.activation = activation
        self.recurrent_dropout = recurrent_dropout
        self.dropout = dropout
        self.regularizer = regularizer
        self.layer_norm = layer_norm
        self.state_size = tf.TensorShape((self.tot_nodes, self.hidden_size_out))
        self.output_size = tf.TensorShape((self.tot_nodes, self.hidden_size_out))
        if self.layer_norm:
            self.ln = l.LayerNormalization()
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)

        self.gat_u = SelfAttention(channels=hidden_size_out, attn_heads=4, concat_heads=concat_heads, dropout_rate=0,
                                   activation=self.activation)
        self.gat_r = SelfAttention(channels=hidden_size_out, attn_heads=4, concat_heads=concat_heads, dropout_rate=0,
                                   activation=self.activation)
        self.gat_c = SelfAttention(channels=hidden_size_out, attn_heads=4, concat_heads=concat_heads, dropout_rate=0,
                                   activation=self.activation)

    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        self.b_u = self.add_weight(shape=(self.hidden_size_out,), initializer=init.Zeros, name="b_u",
                                   regularizer=self.regularizer, caching_device=default_caching_device)
        self.b_r = self.add_weight(shape=(self.hidden_size_out,), initializer=init.Zeros, name="b_r",
                                   regularizer=self.regularizer, caching_device=default_caching_device)
        self.b_c = self.add_weight(shape=(self.hidden_size_out,), initializer=init.zeros, name="b_c",
                                   regularizer=self.regularizer, caching_device=default_caching_device)
        self.W_u = self.add_weight(shape=(self.hidden_size_out, self.hidden_size_out),
                                   initializer="he_normal", name="W_u_p",
                                   regularizer=self.regularizer, caching_device=default_caching_device)
        self.W_r = self.add_weight(shape=(self.hidden_size_out, self.hidden_size_out),
                                   initializer="he_normal", name="W_r_p",
                                   regularizer=self.regularizer, caching_device=default_caching_device)
        self.W_c = self.add_weight(shape=(self.hidden_size_out, self.hidden_size_out),
                                   initializer="he_normal", name="W_c_p",
                                   regularizer=self.regularizer, caching_device=default_caching_device)

    def call(self, inputs, states, training, *args, **kwargs):
        x, a = tf.nest.flatten(inputs)
        h = states[0]
        if 0 < self.recurrent_dropout < 1:
            h_mask = self.get_recurrent_dropout_mask_for_cell(inputs=h, training=training, count=1)
            h = h * h_mask
        if 0 < self.dropout < 1:
            x_mask = self.get_dropout_mask_for_cell(inputs=x, training=training, count=1)
            x = x * x_mask
        u = tf.nn.sigmoid(self.b_u + self.gat_u([tf.concat([x, h], -1), a]) @ self.W_u)
        r = tf.nn.sigmoid(self.b_r + self.gat_r([tf.concat([x, h], -1), a]) @ self.W_r)
        c = tf.nn.tanh(self.b_c + self.gat_c([tf.concat([x, r * h], -1), a]) @ self.W_c)
        h_prime = u * h + (1 - u) * c
        if self.layer_norm:
            h_prime = self.ln(h_prime)
        return h_prime, h_prime

    def get_config(self):
        config = {"nodes": self.tot_nodes,
                  "hidden_size_out": self.hidden_size_out,
                  "dropout": self.dropout,
                  "recurrent_dropout": self.recurrent_dropout,
                  "regularizer": self.regularizer
                  }
        config.update(_config_for_enable_caching_device(self))
        base_config = super(NestedGRUGATCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BatchBilinearDecoderDense(l.Layer):
    """
    inputs:
        - X of shape batch x N x d
        - A of shape batch x N x N
    outputs: A of shape batch x N x N
    """

    def __init__(self, activation="relu", qr=False, regularizer="l2", zero_diag=True):
        super(BatchBilinearDecoderDense, self).__init__()
        self.activation = activation
        self.regularizer = regularizer
        self.qr = qr
        self.zero_diag = zero_diag

    def build(self, input_shape):
        x = input_shape
        self.R = self.add_weight(
            shape=(x[-1], x[-1]),
            initializer="glorot_normal",
            regularizer=self.regularizer,
            name="bilinear_matrix",
        )
        self.diag = tf.constant(1 - tf.linalg.diag([tf.ones(x[-2])]))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        if self.qr:
            Q, W = tf.linalg.qr(x, full_matrices=False)
            W_t = tf.einsum("...jk->...kj", W)
            Q_t = tf.einsum("...jk->...kj", Q)
            Z = tf.matmul(tf.matmul(W, self.R), W_t)
            A = tf.matmul(tf.matmul(Q, Z), Q_t)
            A = act.get(self.activation)(A)
        else:
            x_t = tf.einsum("...jk->...kj", x)
            mat_left = tf.matmul(x, self.R)
            A = act.get(self.activation)(tf.matmul(mat_left, x_t))
        if self.zero_diag:
            return A * self.diag
        return A


class BilinearDecoderSparse(l.Layer):
    def __init__(self, activation="relu", diagonal=False, qr=False, **kwargs):
        super(BilinearDecoderSparse, self).__init__(**kwargs)
        self.initializer = init.GlorotNormal()
        self.diagonal = diagonal
        self.activation = activation
        self.qr = qr

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        if self.diagonal:
            self.R_kernel = tf.Variable(
                initial_value=self.initializer(shape=(X_shape[-1]))
            )
            self.R_kernel = tf.linalg.diag(self.R_kernel)
        else:
            self.R_kernel = tf.Variable(
                initial_value=self.initializer(shape=(X_shape[-1], X_shape[-1]))
            )

    def call(self, inputs, **kwargs):
        X, A = inputs
        """if self.qr:
            Q, W = tf.linalg.qr(X, full_matrices=False)
            Z = tf.matmul(tf.matmul(W, self.R), W, transpose_b=True)
            A = tf.matmul(tf.matmul(Q, Z), Q, transpose_b=True)"""
        i, j = A.indices[:, 0], A.indices[:, 1]
        e1 = tf.gather(X, i)
        e2 = tf.gather(X, j)
        left = tf.einsum("ij,jk->ik", e1, self.R_kernel)
        right = tf.einsum("ij,ij->i", left, e2)
        if self.activation:
            A_pred = act.get(self.activation)(right)
        A_pred = tf.sparse.SparseTensor(A.indices, A_pred, A.shape)
        return X, A_pred


class SelfAttention(l.Layer):
    def __init__(
            self,
            channels=10,
            attn_heads=5,
            dropout_rate=0.5,
            activation="relu",
            concat_heads=False,
            return_attn=False,
            renormalize=False,
            initializer="glorot_normal"
    ):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.attn_heads = attn_heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.concat_heads = concat_heads
        self.return_attn = return_attn
        self.renormalize = renormalize
        self.initializer = init.get(initializer)

    def build(self, input_shape):
        """
        Inputs: X, A
            - X: shape(NxTxd)
            #- A: shape(TxNxN)
            - A: shape(NxTxT)
        """
        x, a = input_shape
        self.q_w = self.add_weight(name="query", shape=(self.attn_heads, x[-1], self.channels),
                                   initializer=self.initializer)
        self.k_w = self.add_weight(name="key", shape=(self.attn_heads, x[-1], self.channels),
                                   initializer=self.initializer)
        self.v_w = self.add_weight(name="value", shape=(self.attn_heads, x[-1], self.channels),
                                   initializer=self.initializer)
        if self.dropout_rate:
            self.drop = l.Dropout(self.dropout_rate)

    def call(self, inputs, *args, **kwargs):
        """
        query=key=value:
            - n: nodes if time series or batch size
            - t: time dim if time series or number of nodes if n=batch size
            - d: input embedding dimension
            - o: output embedding dimension
            - h: number of heads
        x=input embedding of shape NxTxd
        a=input adjacency matrix of shape NxTxT
            -
        """
        x, a = inputs
        query = tf.einsum("ntd,hdo->ntho", x, self.q_w)
        key = tf.einsum("ntd,hdo->ntho", x, self.k_w)
        value = tf.einsum("ntd,hdo->ntho", x, self.v_w)
        qk = tf.einsum("ntho,nzho->nhtz", query, key)  # NxHxTxT
        qk /= tf.sqrt(tf.cast(self.channels, tf.float32))
        qk += tf.transpose([tf.where(a == 0.0, -1e10, 0.0)] * self.attn_heads, perm=(1, 0, 2, 3))  # NxHxTxT
        soft_qk = tf.nn.softmax(qk, axis=-1)
        if self.dropout_rate:
            soft_qk = self.drop(soft_qk)
            if self.renormalize:
                soft_qk = tf.nn.softmax(soft_qk, axis=-1)
        x_prime = tf.einsum("nhtz,nzho->nhto", soft_qk, value)
        if self.concat_heads:
            x_prime = tf.transpose(x_prime, (0, 2, 1, 3))  # NxTxHxO
            x_prime = tf.reshape(x_prime, (*tf.shape(x_prime)[:-2], -1))  # NxTxHO
        else:
            x_prime = tf.reduce_mean(x_prime, axis=1)
            x_prime = tf.squeeze(x_prime)  # NxTxO
        if self.return_attn:
            return x_prime, soft_qk
        return x_prime


class GATv2Layer(l.Layer):
    def __init__(self,
                 attn_heads,
                 channels,
                 concat_heads=False,
                 add_bias=True,
                 activation="relu",
                 dropout_rate=0.2,
                 residual=False,
                 initializer=init.glorot_normal,
                 regularizer=None,
                 return_attn_coef=False,
                 **kwargs):
        super(GATv2Layer, self).__init__(**kwargs)
        self.heads = attn_heads
        self.channels = channels
        self.concatenate_output = concat_heads
        self.add_bias = add_bias
        self.activation = activation
        self.residual = residual
        self.initializer = initializer
        self.regularizer = regularizer
        self.return_attention = return_attn_coef
        self.dropout = l.Dropout(dropout_rate)

    def build(self, input_shape):
        x, a = input_shape
        self.w_shape = (self.heads, self.channels, self.channels)
        self.attn_shape = (self.heads, self.channels)
        self.W_self = self.add_weight(name=f"kern_features_self",
                                      shape=self.w_shape,
                                      initializer=self.initializer,
                                      regularizer=self.regularizer,
                                      trainable=True
                                      )
        self.W_ngb = self.add_weight(name=f"kern_features_ngb",
                                     shape=self.w_shape,
                                     initializer=self.initializer,
                                     regularizer=self.regularizer,
                                     trainable=True
                                     )
        self.attn = self.add_weight(name=f"kern_attention",
                                    shape=self.attn_shape,
                                    initializer=self.initializer,
                                    regularizer=self.regularizer,
                                    trainable=True
                                    )
        self.kernel = self.add_weight(name="kernel_feature",
                                      shape=(x[-1], self.channels))
        if self.add_bias:
            self.bias = self.add_weight(name="kern_bias",
                                        shape=(1, *self.attn_shape))
            self.bias0 = self.add_weight(name="kern_bias_i",
                                       shape=(self.channels,))

    def call(self, inputs, training=None, mask=None):
        '''
        When the adjacency matrix is a Sparse Tensor batch size is not supported
        :param inputs: Tuple of features NxF and Sparse Adjacency Matrix NxN, in dense mode a tuple of BxNxF and Dense
        Adjacency Matrix BxNxN
        :param training: Whether in training mode
        :param mask: Not Used
        :return: Updated Features of shape Nx(HF) or NF
        '''
        x, a = inputs
        assert a.shape[-1] == a.shape[-2]
        x = x @ self.kernel
        if self.add_bias:
            x = x + self.bias0
        x = act.get(self.activation)(x)

        if isinstance(a, tf.sparse.SparseTensor):
            tf.assert_rank(a, 2)
            N = tf.shape(x, out_type=a.indices.dtype)[-2]
            i, j = a.indices[:, 0], a.indices[:, 1]
            x_i_prime = tf.einsum("NF,HFO->NHO", x, self.W_self)
            x_i = tf.gather(x_i_prime, i, axis=0)
            x_j_prime = tf.einsum("NF,HFO->NHO", x, self.W_ngb)
            x_j = tf.gather(x_j_prime, j, axis=0)
            x_ij_prime = x_i + x_j  # EHO
            if self.add_bias:
                x_ij_prime = x_ij_prime + self.bias
            x_ij_prime = act.get(self.activation)(x_ij_prime)
            a_ij = tf.einsum("EHO,HO->EH", x_ij_prime, self.attn)
            a_soft_ij = unsorted_segment_softmax(a_ij, j, N)
            a_soft_ij = self.dropout(a_soft_ij)
            out = a_soft_ij[..., None] * x_i[:, None]  # EH
            out = tf.math.unsorted_segment_sum(out, j, N)  # NHF
            if self.concatenate_output:
                out = tf.reshape(out, (-1, self.attn_shape[0] * self.attn_shape[1]))
            else:
                out = tf.math.reduce_mean(out, -2)
            if self.return_attention:
                return out, a_soft_ij
            else:
                return out
        else:
            x_i = tf.einsum("...NF,HFO->...HON", x, self.W_self)
            x_j = tf.einsum("...NF,HFO->...HON", x, self.W_ngb)
            x_ij = x_i[..., None, :] + x_j[..., None]  # BHONN
            if self.add_bias:
                x_ij = x_ij + self.bias[:, :, :, None, None]
            x_ij_activated = act.get(self.activation)(x_ij)
            e_ij = tf.einsum("...HONK,HO->...HNK", x_ij_activated, self.attn)
            a_mask = tf.where(a == 0, -10e9, 0.0)
            a_mask = tf.repeat(a_mask[:, None, ...], self.heads, 1)  # BHNN
            a_soft_ij = tf.nn.softmax(a_mask + e_ij)
            a_soft_ij = self.dropout(a_soft_ij)
            x_prime = tf.einsum("...HNK,...NF->...KHF", a_soft_ij, x)
            if self.concatenate_output:
                out = tf.reshape(x_prime, (*x.shape[:2], self.heads * x.shape[-1]))  # BxNx(FH)
            else:
                out = tf.reduce_mean(x_prime, 2)  # BxNxF (reduce over heads)
            if self.return_attention:
                return out, a_soft_ij
            else:
                return out