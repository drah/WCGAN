from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import *

def discriminator_i(image, options, reuse=False, name="discriminator_i"):
    """
    image -> conv -> score
    Args:
        image: (batch_size, 64, 64, 3)

    Returns:
        regions_score: (batch_size, 2, 2, 1)
    """

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, name='d_h3_conv'), 'd_bn3'))
        h4 = conv2d(h3, 1, name='d_h4_conv')
        return h4

def discriminator_t(seq, seq_len, options, reuse=False, name="discriminator_t"):
    """
    text -> conv -> score
    Args:
        seq: (batch_size, len, vocab_size)
        seq_len: (batch_size)

    Returns:
        score: (batch_size, 8, 1, 1)
    """
    batch_size = seq.get_shape().as_list()[0]

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        net = seq
        net = tf.expand_dims(net, 2)
        net = lrelu(instance_norm(tf.layers.conv2d(net, options.df_dim*2, (2,1)), 'd_bn_1'))
        net = lrelu(instance_norm(tf.layers.conv2d(net, options.df_dim*4, (2,1)), 'd_bn_2'))
        net = lrelu(instance_norm(tf.layers.conv2d(net, options.df_dim*8, (2,1)), 'd_bn_3'))
        net = tf.layers.conv2d(net, 1, (3,1))
        return net

def residule_block(x, dim, ks=3, s=1, name='res'):
    p = int((ks - 1) / 2)
    y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
    y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
    y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
    y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
    return y + x

def generator_it(image, seq, seq_len, options, reuse=False, name="generator_it"):
    """
    image -> conv -> fully -> lstm -> text

    Args:
        image: (batch_size, 64, 64, 3)
        seq: (batch_size, max_len, vocab_size)
        seq_len: (batch_size) or None

    Returns:
        generated_text: (batch_size, max_len, vocab_size)

    """
    batch_size = image.get_shape().as_list()[0]
    timesteps = seq.get_shape().as_list()[1]
    vocab_size = seq.get_shape().as_list()[2]

    seq = tf.argmax(seq, 2)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(
                instance_norm(
                    conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))

        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        net = tf.nn.avg_pool(r9, [1,2,2,1], [1,2,2,1], 'VALID')
        net = tf.reshape(net, [batch_size, -1])
        net = tf.layers.dense(net, 128, tf.nn.relu)

        with tf.variable_scope("rnn_generate_text"):
            with tf.variable_scope("cell"):
                cell = tf.contrib.rnn.LSTMCell(options.gf_dim*4)

            encoded_image = net
            state_tuple = tf.contrib.rnn.LSTMStateTuple(
                    c=encoded_image, h=tf.zeros_like(encoded_image))

            seq = tf.transpose(seq, [1,0])
            seq = tf.unstack(seq)
            pw = xavier_w("pw", [options.gf_dim*4, 42])
            pb = zero_w("pb", [42])
            outputs, _ = tf.contrib.legacy_seq2seq.embedding_rnn_decoder(
                    seq, state_tuple, cell, 42, options.gf_dim*4, (pw,pb), True)

            net = outputs
            net = tf.stack(net)
            net = tf.transpose(net, [1,0,2])

            with tf.variable_scope("proj_softmax"):
                net = tf.expand_dims(net, 2)
                net = tf.layers.conv2d(net, vocab_size, (1,1), activation=None)
                net = tf.nn.softmax(net)
                net = tf.squeeze(net, 2)

        return net

def generator_ti(seq, seq_len, options, reuse=False, name="generator_ti"):
    """
    text -> lstm -> fully -> conv_t -> image
    Args:
        seq: (batch_size, max_len, vocab_size)
        seq_len: (batch_size) or None

    Returns:
        generated_image: (batch_size, max_len, vocab_size)

    """
    batch_size = seq.get_shape().as_list()[0]
    timesteps = seq.get_shape().as_list()[1]
    vocab_size = seq.get_shape().as_list()[2]

    seq = embed(seq, vocab_size, 128)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        with tf.variable_scope("rnn_encode_text"):
            with tf.variable_scope("cell"):
                cell = tf.contrib.rnn.LSTMCell(options.gf_dim*4)

            init_state = cell.zero_state(batch_size, tf.float32)
            _, state = tf.nn.dynamic_rnn(
                    cell, seq, seq_len, initial_state=init_state, dtype=tf.float32)
            net = tf.concat([state.h, state.c], 1)

        net = tf.layers.dense(net, 4*4*options.gf_dim*4)
        net = tf.reshape(net, [batch_size, 4, 4, options.gf_dim*4])

        d1 = deconv2d(net, options.gf_dim*4, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))

        d2 = deconv2d(d1, options.gf_dim*4, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))

        d3 = deconv2d(d2, options.gf_dim*4, 3, 2, name='g_d3_dc')
        d3 = tf.nn.relu(instance_norm(d3, 'g_d3_bn'))

        d3 = tf.pad(d3, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT")

        r1 = residule_block(d3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d4 = deconv2d(r9, options.gf_dim*4, 3, 2, name='g_d4_dc')
        d4 = tf.nn.relu(instance_norm(d4, 'g_d4_bn'))

        pred = conv2d(d4, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c')
        pred = tf.nn.tanh(instance_norm(pred, 'g_pred_bn'))
        pred = pred[:,1:-1,1:-1,:]

        return pred


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(in_ - target), [1,2,3]))

def xent_criterion(in_, target):
    return -tf.reduce_mean(tf.reduce_sum(target*tf.log(in_), [1,2]))

def w(name, shape, dtype, init, regu=None):
    w = tf.get_variable(name, shape, dtype, init, regu)
    return w

def tn_w(name, shape, seed=None, regu=None, scaling=0.1):
    init = tf.truncated_normal_initializer(0.0, scaling, seed)
    return w(name, shape, tf.float32, init, regu)

def xavier_w(name, shape):
    init = tf.contrib.layers.xavier_initializer()
    return w(name, shape, tf.float32, init)

def zero_w(name, shape):
    init = tf.zeros_initializer()
    return w(name, shape, tf.float32, init)

def embed(x, vocab_size, embed_size):
    x = tf.argmax(x, 2)
    try:
        with tf.variable_scope("generator_it"):
            w = tn_w("rnn_generate_text/embedding_rnn_decoder/embedding", [vocab_size, embed_size])
        print("create embeddings")
    except:
        with tf.variable_scope("generator_it", reuse=True):
            w = tn_w("rnn_generate_text/embedding_rnn_decoder/embedding", [vocab_size, embed_size])
        print("reuse embeddings")
    e = tf.nn.embedding_lookup(w, x)
    return e

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable(
                "scale",
                [depth],
                initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable(
                "offset",
                [depth],
                initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

