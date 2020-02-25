import tensorflow as tf
import tensorflow.contrib.layers as layers
from baselines0.setdeepq.modules import *

"""Default model for setdeepq"""
class SetTransformer:
    """ Based on tf implementation of Attention is All You Need: 
        https://www.github.com/kyubyong/transformer and
        PyTorch implementation of Set Transformer:
        https://github.com/juho-lee/set_transformer
    """
    def __init__(self):
        pass

    def encoder(self, X, dim_out=64, reuse=False):
        with tf.compat.v1.variable_scope('encoder', reuse=reuse):
            #Embeddings for input into blocks (needs consistent shape)
            X_embed = layers.fully_connected(X, num_outputs=dim_out, 
                                                activation_fn=None, scope='embedding')
            Z = SAB(X_embed, dim_out=dim_out)
            Z = SAB(Z, dim_out=dim_out)
        return Z

    def decoder(self, Z, num_actions, dim_out=64, reuse=False):
        with tf.compat.v1.variable_scope('decoder', reuse=reuse):
            out = PMA(Z, dim_out=dim_out)
            out = ff(out, dim_out=num_actions)
        return out

    def forward(self, X, num_actions, scope='SetTransformer', reuse=False):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            Z = self.encoder(X)
            q_out = self.decoder(Z, num_actions)
            return q_out




""" MLP that takes in sets, if using adjustments need to be made in build_graph"""
def _mlp(hiddens, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
        q_out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        # import pdb;pdb.set_trace()
        # q_out = tf.squeeze(q_out[:,0,:], axis=1)
        return q_out[:,0,:]


def mlp(hiddens=[], layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp(hiddens, layer_norm=layer_norm, *args, **kwargs)


"""Set Transofrmer in lambda format, if using adjustments need to be made in build_graph"""
def _setTransformer(X, num_actions, scope, reuse=False, dim_out=64):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        with tf.compat.v1.variable_scope('encoder'):
            #Embeddings for input into blocks (needs consistent shape)
            X_embed = layers.fully_connected(X, num_outputs=dim_out, 
                                                activation_fn=None, scope='embedding')
            Z = SAB(X_embed, dim_out=dim_out)
            Z = SAB(Z, dim_out=dim_out)
        with tf.compat.v1.variable_scope('decoder'):
            out = PMA(Z, dim_out=dim_out)
            out = ff(out, dim_out=num_actions)
    return out

def setTransformer():
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _setTransformer(*args, **kwargs)