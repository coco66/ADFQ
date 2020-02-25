import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

def SAB(X, dim_out, scope='SAB'):
    # Set Attention Block
    with tf.compat.v1.variable_scope(scope):
        out = MAB(X, X, dim_out)
        out = ff(out)
    return out

def PMA(X, dim_out, num_seeds=1, scope='PMA'):
    # Pooling by Multihead Attention
    with tf.compat.v1.variable_scope(scope):
        S = tf.get_variable("W", shape=[1, num_seeds, dim_out],
                            initializer=tf.contrib.layers.xavier_initializer())
        S_ = tf.tile(S, [tf.shape(X)[0],1,1])
        out = MAB(S_, X, dim_out)
        # Rm dim=1 [N,1,d_obs] to out = [N,d_obs]
        out = tf.squeeze(out, axis=1)
    return out

def MAB(Q, K, dim_out, num_heads=4, scope='MAB'):
    # Multihead Attention Block
    with tf.compat.v1.variable_scope(scope):
        # Linear projection
        _Q = layers.fully_connected(Q, num_outputs=dim_out, activation_fn=None, scope='fc_Q')
        _K = layers.fully_connected(K, num_outputs=dim_out, activation_fn=None, scope='fc_K')
        _V = layers.fully_connected(K, num_outputs=dim_out, activation_fn=None, scope='fc_V')
        # Split and concat
        Q_ = tf.concat(tf.split(_Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_obs/h)
        K_ = tf.concat(tf.split(_K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(_V, num_heads, axis=2), axis=0)

        outputs = scaled_dot_product_attention(Q_, K_, V_)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        # Residual connection
        outputs += _Q
        # Normalization
        # outputs = layer_norm(outputs)
        outputs = layers.layer_norm(outputs, center=True, scale=True, scope='ln_MAP')

    return outputs


def scaled_dot_product_attention(Q, K, V, scope='Attention'):
    # Attention mechanism
    with tf.variable_scope(scope):
        d_k = Q.get_shape().as_list()[-1]
        # Q*K^T
        out = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        # Scale
        out /= np.sqrt(d_k)
        # Softmax over d_k
        out = tf.nn.softmax(out, axis=2)
        # Visualize attention
        # attention = tf.transpose(out, [0, 2, 1])
        # tf.summary.image("attention", tf.expand_dims(attention[:1], -1))
        # Weighted sum
        out = tf.matmul(out, V)
        #Can add dropout here

    return out

def ff(inputs, hidden=64, dim_out=64, scope='feed_forward'):
    #position-wise feed forward net. See 3.3
    with tf.compat.v1.variable_scope(scope):
        #Layers
        outputs = layers.fully_connected(inputs, num_outputs=hidden, 
                                            activation_fn=tf.nn.relu, scope='fc_1')
        outputs = layers.fully_connected(outputs, num_outputs=dim_out, 
                                            activation_fn=None, scope='fc_2')
        #Residual connection
        # outputs += inputs
        #Normalize
        # outputs = layer_norm(outputs)
        outputs = layers.layer_norm(outputs, center=True, scale=True, scope='ln_ff')
    return outputs

def layer_norm(inputs, epsilon = 1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
    return outputs