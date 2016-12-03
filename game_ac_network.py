# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import tf_common as tfc
import constants
import pickle

FLAGS = tf.app.flags.FLAGS

class Network(object):
    def __init__(self, name="agent"):
        self.name = name

        with tf.device(constants.device):
            with tf.variable_scope(name):
                self.create_pnn()

    def debug(self, sess):
        fd = {self.s: np.ones((1, 84, 84, 4))}

        a = sess.run(self.pi, feed_dict=fd)
        b = sess.run(self.pi, feed_dict={self.s: np.zeros((1, 84, 84, 4))})

        exit()

    def create_pnn(self):
        #self.s = tf.placeholder("float", [None, FLAGS.screen_height, FLAGS.screen_width, constants.history_frames], "state")
        self.s = tf.placeholder("float", [None, 84, 84, constants.history_frames],
                                "state")
        self.train_vars = []
        self.var_dict = {}
        self.all_vars = []
        self.col_hiddens = []

        for i in range(len(constants.tasks)):
            print(">>>>>>>>>>>>>>>>>>>>>>")
            p, v, col_vars, col_h = create_column(constants.tasks, i, self.s, self.col_hiddens)

            vvv = []#col_vars[:-4]
            if i == len(constants.tasks)-1:
                vvv = col_vars
                #print([var.name for var in vvv])

            self.all_vars.extend(col_vars)

            for var in vvv:
                if var in tf.trainable_variables():
                    self.train_vars.append(var)

            for col_var in col_vars:
                n = col_var.name
                n = n[n.index('/'):]
                self.var_dict[n] = col_var

            self.col_hiddens.append(col_h)

            if i == len(constants.tasks)-1:
                print("setting policy and value tensors.")
                self.pi = p
                self.v = v
            print("<<<<<<<<<<<<<<<<<<<<<")

        self.columns = len(self.col_hiddens)
        self.layers = len(self.col_hiddens[0])

        print([v.name for v in self.train_vars])
        print("%i trainable weight variables." % len(self.train_vars))

    def run_policy_and_value(self, sess, s_t):
        pi_out, v_out = sess.run([self.pi, self.v], feed_dict={self.s: [s_t]})
        return (pi_out[0], v_out[0])

    def run_policy(self, sess, s_t):
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]

    def run_value(self, sess, s_t):
        v_out = sess.run(self.v, feed_dict={self.s: [s_t]})
        return v_out[0]

    def get_train_vars(self):
        return self.train_vars

    def evaluate_vars(self, sess):
        for v in self.train_vars:
            print(v.name)
            print(sess.run(v))
            print("="*20)

    def prepare_loss(self, entropy_beta):
        with tf.device(constants.device):
            # taken action (input for policy)
            self.a = tf.placeholder("float", [None, FLAGS.action_size])

            # temporary difference (R-V) (input for policy)
            self.td = tf.placeholder("float", [None])

            # avoid NaN with clipping when value in pi becomes zero
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

            # policy entropy
            entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)

            # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
            policy_loss = - tf.reduce_sum(
                tf.reduce_sum(tf.mul(log_pi, self.a), reduction_indices=1) * self.td + entropy * entropy_beta)

            # R (input for value)
            self.r = tf.placeholder("float", [None])

            # value loss (output)
            # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
            value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

            # gradienet of policy and value are summed up
            self.total_loss = policy_loss + value_loss

    def sync_from(self, src_netowrk, name=None):
        src_vars = src_netowrk.all_vars
        dst_vars = self.all_vars

        sync_ops = []

        with tf.device(constants.device):
            with tf.op_scope([], name, "GameACNetwork") as name:
                for (src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)

    def save(self, sess, path):
        weights = {}

        for name, var in self.var_dict.items():
            weights[name] = sess.run(var)

        pickle.dump(weights, open(path, "wb"))

    def load(self, sess, path):
        weights = pickle.load(open(path, "rb"))

        print("CURRENT MODEL VARIABLES: " + str([v for v in self.var_dict.keys()]))
        print("LOADING WEIGHTS FOR: " + str(weights.keys()))

        for suffix, values in weights.items():
            #if "p_" not in suffix and "v_" not in suffix:
                #print(suffix)
            var_name = self.name + suffix
            sess.run(tf.assign(self.var_dict[suffix], values))
            print("loaded values for: %s" % var_name)
            #else:
            #    print("!!!skipping")



    def get_grads(self):  # to be implemented later
        # print(self.col_hiddens[k][i])
        grads = [[None for i in range(self.layers)] for k in range(self.columns)]
        for k in range(self.columns):
            for i in range(self.layers):
                norm = self.col_hiddens[k][i]#/tf.reduce_sum(self.col_hiddens[k][i])
                g = tf.gradients(tf.log(self.pi), norm)[0]

                grads[k][i] = g
        return grads

    def sample_fisher(self, sess, state, grads):
        dpdh = []

        for k in range(self.columns):
            print(k)
            col_dpdh = []
            for i in range(self.layers):
                print(i)
                dpdh_mat = np.power(sess.run(grads[k][i], feed_dict={self.s: [state]}), 2.0)
                if len(dpdh_mat.shape) == 4:
                    dpdh_mat = np.sum(dpdh_mat, (0, 1, 2, 3))
                else:
                    # print(grad)
                    dpdh_mat = np.sum(dpdh_mat, (0, 1))
                # self.get_current_dpdh(sess, i, k, state)
                # print(dpdh_mat.shape)
                # print(self.col_hiddens[k][i])
                # print(dpdh_mat)

                col_dpdh.append(dpdh_mat)
            dpdh.append(col_dpdh)

        # fishers = []

        # for k in range(self.columns):
        #    lyrs = []
        #    for i in range(self.layers):
        #        f = np.dot(dpdh[k][i], dpdh[k][i].T)
        #        lyrs.append(f)
        #    fishers.append(lyrs)

        return dpdh  # fishers

def create_column(col_names, self_i, state, col_hiddens):
    print("creating column %i" % self_i)

    arch = [
        [8, constants.history_frames, 16, 4],  # size, in, out, stride
        [4, 16, 32, 2],
        [256],
        -1
    ]

    train_vars = []
    lats = [] #k, i
    c_lats = []

    if self_i > 0:
        with tf.variable_scope("laterals"):
            print("creating lateral connections to column %i" % self_i)

            for col_i in range(self_i):
                hiddens = col_hiddens[col_i]

                print("##" + str(len(col_hiddens[col_i])))

                col_lats = []

                print("creating laterals %i -> %i" % (col_i, self_i))

                with tf.variable_scope("%s_to_%s" % (col_names[col_i], col_names[self_i])):
                    for layer_i in range(len(hiddens)):
                        layer_lats = []
                        print("###" + str(layer_i))
                        dest_h_shape = arch[layer_i + 1]

                        with tf.variable_scope("layer%ito%i" % (layer_i, layer_i+1)):
                            orig_h = hiddens[layer_i]#tf.stop_gradient(hiddens[layer_i]) #origin

                            print("layer %i -> %i" % (layer_i, layer_i + 1))

                            if dest_h_shape == -1: # to policy and value layer
                                with tf.variable_scope("policy"):
                                    lat_h_p, lat_vars_p = lateral_connection(orig_h, [FLAGS.action_size], self_i)
                                with tf.variable_scope("value"):
                                    lat_h_v, lat_vars_v = lateral_connection(orig_h, [1], self_i)

                                layer_lats.append(lat_h_p)
                                layer_lats.append(lat_h_v)
                                train_vars.extend(lat_vars_p)
                                train_vars.extend(lat_vars_v)
                            else:
                                lat_h, lat_vars = lateral_connection(orig_h, dest_h_shape, self_i, arch[layer_i + 1])

                                layer_lats.append(lat_h)
                                train_vars.extend(lat_vars)

                            col_hiddens[col_i][layer_i] = orig_h

                        col_lats.append(layer_lats)
                lats.append(col_lats)

        #print("columns: %i" % (len(lats) + 1))
        #print("hidden layers: %i" % (len(lats[0])))
        #print("hidden shapes: %s" % col_hiddens[0])

        #concatenate same-layer lateral connections
        for i in range(len(lats[0])):
            if arch[i+1] == -1:
                to_policy_list = [lats[k][i][0] for k in range(len(lats))]
                to_value_list = [lats[k][i][1] for k in range(len(lats))]
                to_policy = tf.reduce_sum(to_policy_list, 0)
                to_value = tf.reduce_sum(to_value_list, 0)

                c_lats.append([to_policy, to_value])

                print("summing ->policy and ->value layers")
                print(to_policy_list)
                print("=>")
                print(to_policy)
                print("&")
                print(to_value_list)
                print("=>")
                print(to_value)
            else:
                h_list = [lats[k][i][0] for k in range(len(lats))]

                if len(arch[i+1]) > 1:
                    c = tf.reduce_sum(h_list, 0)
                    c_lats.append(c)
                    print("summing convolutional layers")
                    print(h_list)
                    print("=>")
                    print(c)
                else:
                    c = tf.reduce_sum(h_list, 0)
                    c_lats.append(c)
                    print("summing fully connected layers")
                    print(h_list)
                    print("=>")
                    print(c)

            print("~~~")

    print("done summing layers")
    #print("c lats:")
    #print(c_lats)

    def add_lat(layer, i, act=tf.nn.relu):

        if self_i <= 0:
            if act is None:
                return layer[0], layer[1], layer[2]
            else:
                return act(layer[0]), layer[1], layer[2]
        elif len(i) == 1:
            print("adding %s and %s" % (layer[0], c_lats[i[0]]))
            return act(layer[0]+c_lats[i[0]]), layer[1], layer[2]
        else:
            if act is None:
                print("(value) adding %s and %s" % (layer[0], c_lats[i[0]][i[1]]))
                return layer[0] + c_lats[i[0]][i[1]], layer[1], layer[2]
            else:
                print("(policy) adding %s and %s" % (layer[0], c_lats[i[0]][i[1]]))
                return act(layer[0]+c_lats[i[0]][i[1]]), layer[1], layer[2]

    train = self_i == len(constants.tasks)-1
    print("column trainable: %s" % train)

    with tf.variable_scope(col_names[self_i]):
        #resized = tf.image.resize_images(state, 84, 84)

        c1, w1, b1 = tfc.conv2d("c1", state, arch[0][1], arch[0][2], size=arch[0][0], stride=arch[0][3], trainable=train)
        c2, w2, b2 = add_lat(tfc.conv2d("c2", c1, arch[1][1], arch[1][2], size=arch[1][0], stride=arch[1][3], act=None, trainable=train), [0])

        c2_size = np.prod(c2.get_shape().as_list()[1:])
        c2_flat = tf.reshape(c2, [-1, c2_size])

        if self_i <= 0:
            h_fc1, w3, b3 = tfc.fc("fc1", c2_flat, c2_size, arch[2][0], trainable=train)
        else:
            h_fc1, w3, b3 = tfc.fc("fc1", c2_flat, c2_size, arch[2][0], act=None, trainable=train)

            lat = c_lats[1]
            print("adding %s and %s" % (h_fc1, lat))
            lat_size = np.prod(lat.get_shape().as_list()[1:])
            lat_flat = tf.reshape(lat, [-1, lat_size])
            h_fc1 = tf.nn.relu(h_fc1 + lat_flat)

        pi, wp, bp = add_lat(tfc.fc("p_fc", h_fc1, arch[2][0], FLAGS.action_size, act=None, trainable=train), [2, 0], tf.nn.softmax)
        v_, wv, bv = add_lat(tfc.fc("v_fc", h_fc1, arch[2][0], 1, act=None, trainable=train), [2, 1], None)

        v = tf.reshape(v_, [-1])

        train_vars.extend([w1, b1, w2, b2, w3, b3, wp, bp, wv, bv])

        col_vars = pi, v, train_vars, [c1, c2, h_fc1]

        print("policy: %s" % pi)
        print("last fc: %s" % h_fc1)
        print("wp: %s" % wp.name)

        print("created column %i." % self_i)

        return col_vars

def lateral_connection(orig_hidden, dest_shape, self_i, current_op_shape=None):


    print("adapter origin: %s" % orig_hidden.name)
    train = self_i == len(constants.tasks)-1
    #print(self_i)
    #print(len(constants.tasks)-1)
    print("lateral trainable: %s" % train)
    nonlinear = True

    omit_b = True

    a = tf.get_variable(name="adapter", shape=[1], initializer=tf.constant_initializer(1), trainable=train)
    ah = tf.mul(a, orig_hidden)

    if nonlinear:
        if len(orig_hidden.get_shape().as_list()) == 4:
            maps_in = ah.get_shape().as_list()[3]
            nic = int(maps_in / (2.0 * (self_i)))
            lateral, w1, b1 = tfc.conv2d("V", ah, maps_in, nic, size=1, stride=1, trainable=train)  # reduction (keep bias)

            print("1) conv 1x1: %s" % w1.get_shape())

            if len(dest_shape) > 1:   # conv layer to conv layer
                lateral, w2, _ = tfc.conv2d("U", lateral, nic, current_op_shape[2], size=current_op_shape[0],
                                           stride=current_op_shape[3], act=None, omit_bias=omit_b, padding="SAME", trainable=train)
                print("2) conv 1x1: %s" % w2.get_shape())
                print("end result: %s" % lateral.name)

                return lateral, [w1, b1, w2]

            else:  # conv layer to fc layer
                c_size = np.prod(lateral.get_shape().as_list()[1:])
                c_flat = tf.reshape(lateral, [-1, c_size])
                lateral, w2, _ = tfc.fc("U", c_flat, c_size, dest_shape[0], act=None, omit_bias=omit_b, trainable=train)
                print("2) flattened conv fc: %s" % w2.get_shape())
                print("end result: %s" % lateral.name)

                return lateral, [w1, b1, w2]

        else:  # fc layer to fc layer
            n_in = ah.get_shape().as_list()[1]
            ni = int(n_in / (2.0 * (self_i)))
            lateral, w1, b1 = tfc.fc("V", ah, n_in, ni, trainable=train)  # reduction (keep bias)
            print("1) fc: %s" % w1.get_shape())
            lateral, w2, _ = tfc.fc("U", lateral, ni, dest_shape[0], act=None, omit_bias=omit_b, trainable=train) # to be added to next hidden
            print("2) fc: %s" % w2.get_shape())
            print("end result: %s" % lateral.name)

            return lateral, [w1, b1, w2]
    else:
        if len(orig_hidden.get_shape().as_list()) == 4:
            maps_in = ah.get_shape().as_list()[3]

            if len(dest_shape) > 1:   # conv layer to conv layer
                lateral, w2, _ = tfc.conv2d("U", ah, maps_in, current_op_shape[2], size=current_op_shape[0],
                                           stride=current_op_shape[3], act=None, omit_bias=omit_b, padding="SAME", trainable=train)
                return lateral, [w2]

            else:  # conv layer to fc layer
                c_size = np.prod(ah.get_shape().as_list()[1:])
                c_flat = tf.reshape(ah, [-1, c_size])
                lateral, w2, _ = tfc.fc("U", c_flat, c_size, dest_shape[0], act=None, omit_bias=True, trainable=train)
                return lateral, [w2]

        else:  # fc layer to fc layer
            n_in = ah.get_shape().as_list()[1]
            lateral, w2, _ = tfc.fc("U", ah, n_in, dest_shape[0], act=None, omit_bias=True, trainable=train) # to be added to next hidden
            return lateral, [w2]
