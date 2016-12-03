import tensorflow as tf

xi = tf.contrib.layers.xavier_initializer
xic = tf.contrib.layers.xavier_initializer_conv2d


def fc(name, x, _in, out, act=tf.nn.relu, omit_bias=False, trainable=True):
    with tf.variable_scope(name):
        w = tf.get_variable(name="fc_w", shape=[_in, out], initializer=xi(), trainable=trainable)
        b = None
        if omit_bias:
            xw = tf.matmul(x, w)
        else:
            b = tf.get_variable(name="fc_b", shape=[out], initializer=tf.constant_initializer(0), trainable=trainable)
            xw = tf.matmul(x, w) + b

    if act is not None:
        return act(xw, name=name), w, b
    else:
        return xw, w, b


def conv2d(name, x, maps_in, maps_out, size=3, stride=1, act=tf.nn.relu, omit_bias=False, padding="SAME", trainable=True):
    with tf.variable_scope(name):
        w = tf.get_variable(name="conv2d_w", shape=[size, size, maps_in, maps_out], initializer=xic(), trainable=trainable)
        b = None
        if omit_bias:
            c = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
        else:
            b = tf.get_variable(name="conv2d_b", shape=[maps_out], initializer=tf.constant_initializer(0), trainable=trainable)
            c = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding) + b

    if act is not None:
        return act(c, name=name), w, b
    else:
        return c, w, b

def max_pool(name, x):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def load_or_init(model_dir, sess):
    init = tf.initialize_all_variables()

    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver = tf.train.Saver()
    summary_writer = tf.train.SummaryWriter(model_dir)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restored %s" % ckpt.model_checkpoint_path)
    else:
        sess.run(init)

    return saver, summary_writer

def summary_float(step, name, value, summary_writer):
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=name, simple_value=float(value))])
    summary_writer.add_summary(summary, global_step=step)