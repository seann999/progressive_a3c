# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os
import time
import constants

from game_ac_network import Network
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier
import tf_common as tfc

flags = tf.app.flags
FLAGS = flags.FLAGS

class Trainer:
    def __init__(self):
        pass

    def signal_handler(self, signal, frame):
        print('Stop requested.')
        self.stop_requested = True

    def train_function(self, parallel_index):
        training_thread = self.training_threads[parallel_index]
        # set start_time
        start_time = time.time() - self.wall_t
        training_thread.set_start_time(start_time)

        while True:
            if self.stop_requested:
                break
            if self.global_t > FLAGS.global_t_max:
                break

            diff_global_t = training_thread.process(self.sess, self.global_t, self.summary_writer,
                                                    self.summary_op, self.score_input)

            if self.global_t % FLAGS.save_every > (self.global_t + diff_global_t) % FLAGS.save_every:# or\
             #       (self.global_t < 6000000 and (self.global_t % 1000000 > (self.global_t + diff_global_t) % 1000000)):
                self.save()


            #print(np.mean(self.sess.run(self.global_network.all_vars["/laterals/pong_to_pong_inv/layer2to3/value/V/fc_w:0"])))

            self.global_t += diff_global_t

    def start(self):
        self.global_t = 0

        if not os.path.exists(os.path.join(FLAGS.model_dir, "images")):
            os.makedirs(os.path.join(FLAGS.model_dir, "images"))

        constants.device = "/cpu:0"
        if FLAGS.use_gpu:
            constants.device = "/gpu:0"

        initial_learning_rate = FLAGS.init_lr

        self.stop_requested = False

        # prepare session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                                     allow_soft_placement=True))

        self.global_network = Network(name="core_%s" % constants.task_name)

        self.training_threads = []

        learning_rate_input = tf.placeholder("float")

        grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                                      decay=FLAGS.rmsp_alpha,
                                      momentum=0.0,
                                      epsilon=FLAGS.rmsp_epsilon,
                                      clip_norm=FLAGS.grad_norm_clip,
                                      device=constants.device)

        print("CREATING AGENTS")

        for i in range(FLAGS.threads):
            training_thread = A3CTrainingThread(i, self.global_network, initial_learning_rate,
                                                learning_rate_input,
                                                grad_applier, FLAGS.global_t_max,
                                                device=constants.device, sess=self.sess, name="agent_%s_%i" % (constants.task_name, i))
            self.training_threads.append(training_thread)

        init = tf.initialize_all_variables()
        self.sess.run(init)



        # summary for tensorboard
        self.score_input = tf.placeholder(tf.int32)
        tf.scalar_summary("score", self.score_input)

        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(FLAGS.model_dir, self.sess.graph_def)

        # init or load checkpoint with saver
        self.saver = tf.train.Saver()

        checkpoint = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path
                               if FLAGS.checkpoint is None else FLAGS.checkpoint)
            print("checkpoint loaded:", checkpoint.model_checkpoint_path)
            tokens = checkpoint.model_checkpoint_path.split("-")
            # set global step
            self.global_t = int(tokens[-1])
            print(">>> global step set: ", self.global_t)
            # set wall time
            self.wall_t_fname = FLAGS.model_dir + '/' + 'wall_t.' + str(self.global_t)
            with open(self.wall_t_fname, 'r') as f:
                self.wall_t = float(f.read())
        else:
            print("Could not find old checkpoint")
            # set wall time
            self.wall_t = 0.0

        if FLAGS.transfer_model is not None:
            self.global_network.load(self.sess, FLAGS.transfer_model)

        train_threads = []
        for i in range(FLAGS.threads):
            train_threads.append(threading.Thread(target=self.train_function, args=(i,)))

        signal.signal(signal.SIGINT, self.signal_handler)

        # set start time
        self.start_time = time.time() - self.wall_t

        for t in train_threads:
            t.start()

        print('Press Ctrl+C to stop')
        signal.pause()

        print('Now saving data. Please wait')

        for t in train_threads:
            t.join()

        self.save()

    def save(self):
        if not os.path.exists(FLAGS.model_dir):
            os.mkdir(FLAGS.model_dir)

        # write wall time
        self.wall_t = time.time() - self.start_time
        self.wall_t_fname = FLAGS.model_dir + '/' + 'wall_t.' + str(self.global_t)
        with open(self.wall_t_fname, 'w') as f:
            f.write(str(self.wall_t))

        if not os.path.exists(FLAGS.model_dir + "/core"):
            os.mkdir(FLAGS.model_dir + '/core')
        self.saver.save(self.sess, FLAGS.model_dir + '/' + 'checkpoint', global_step=self.global_t)

        self.global_network.save(self.sess, FLAGS.model_dir + '/core/' + 'checkpoint-%i.p' % self.global_t)

if __name__ == "__main__":
    flags.DEFINE_integer("local_t_max", 5, "local t max")
    flags.DEFINE_float("rmsp_alpha", 0.99, "rmsprop alpha")
    flags.DEFINE_float("rmsp_epsilon", 0.1, "rmsprop epsilon")
    flags.DEFINE_string("model_dir", "summaries/test0", "model dir")
    flags.DEFINE_float("init_lr", 1e-3, "initial learning rate")
    flags.DEFINE_integer("threads", 8, "trainer threads to run in parallel")
    flags.DEFINE_integer("action_size", 4, "action size of game")
    flags.DEFINE_float("gamma", 0.99, "reward discount factor")
    flags.DEFINE_float("entropy_beta", 0.01, "entropy regularization coefficient")
    flags.DEFINE_integer("global_t_max", 1e8, "max iterations")
    flags.DEFINE_float("grad_norm_clip", 40.0, "gradient clipping")
    flags.DEFINE_boolean("use_gpu", False, "use gpu")
    flags.DEFINE_boolean("save_frames", False, "save frame images")
    flags.DEFINE_integer("save_every", 500000, "save model every n steps")
    flags.DEFINE_string("checkpoint", None, "load specific checkpoint")
    flags.DEFINE_integer("screen_width", 227, "screen width")
    flags.DEFINE_integer("screen_height", 227, "screen height")

    #flags.DEFINE_string("task_name", "foo", "name of task")
    flags.DEFINE_string("column_names", "foo", "names of columns(tasks)")
    flags.DEFINE_string("transfer_model", None, "model to transfer from with progressive neural networks")
    #flags.DEFINE_integer("history_frames", 4, "history frames")

    constants.tasks = FLAGS.column_names.split(",")
    constants.task_name = constants.tasks[-1]

    Trainer().start()

