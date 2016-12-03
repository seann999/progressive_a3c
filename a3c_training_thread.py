# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys
import gym
import os
import scipy.misc

from accum_trainer import AccumTrainer
from game_ac_network import Network
import cv2

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

#USE_ALE = False
FLAGS = tf.app.flags.FLAGS

#pong_actions = [1, 2, 3]

class A3CTrainingThread(object):
    def __init__(self,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 grad_applier,
                 max_global_time_step,
                 device,
                 sess,
                 name="agent"):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        #if USE_LSTM:
        #    self.local_network = GameACLSTMNetwork(ACTION_SIZE, thread_index, device)
        #else:

        self.local_network = Network(name=name)

        self.local_network.prepare_loss(FLAGS.entropy_beta)

        # TODO: don't need accum trainer anymore with batch
        self.trainer = AccumTrainer(device)
        self.local_network.vars = self.trainer.prepare_minimize(self.local_network.total_loss,
                                      self.local_network.get_train_vars())

        self.accum_gradients = self.trainer.accumulate_gradients()
        self.reset_gradients = self.trainer.reset_gradients()

        self.apply_gradients = grad_applier.apply_gradients(
            global_network.get_train_vars(),
            self.trainer.get_accum_grad_list())

        self.sync = self.local_network.sync_from(global_network)

        #if USE_ALE:
        #    self.game_state = GameState(113 * thread_index)
        #else:
        self.game = gym.make('Lis-v2')
        self.game.configure(str(5000 + thread_index))
        # game initialization
        # observation = env.reset()
        self.observation, reward, end_episode, _ = self.game.step(1)
        #self.observation = self.preprocess([self.observation])
        self.history = [self.rgb2gray(self.observation) for _ in range(4)]#FLAGS.history_frames
        self.observation = np.dstack(self.history)

        self.local_t = 0

        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0

        # variable controling log output
        self.prev_local_t = 0

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (
        self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, pi_values):
        values = []
        sum = 0.0
        for rate in pi_values:
            sum = sum + rate
            value = sum
            values.append(value)

        r = random.random() * sum
        for i in range(len(values)):
            if values[i] >= r:
                return i
        # fail safe
        return len(values) - 1

    def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
        summary_str = sess.run(summary_op, feed_dict={
            score_input: score
        })
        summary_writer.add_summary(summary_str, global_t)

    def set_start_time(self, start_time):
        self.start_time = start_time

    def rgb2gray(self, rgb, i=0):
        if FLAGS.save_frames:
            if self.thread_index == 0 and len(os.listdir(os.path.join(FLAGS.model_dir, "images"))) < 1000:
                scipy.misc.imsave("%s/%i.png" % (os.path.join(FLAGS.model_dir, "images"), i), rgb["image"][0])

        img = np.asarray(rgb["image"][0])[..., :3]
        img = np.dot(img, [0.299, 0.587, 0.114])
        img = scipy.misc.imresize(img, (84, 84)) / 255.0
        #flip H
        #
        #img = np.fliplr(img)



        return img
        #return -np.dot(img, [0.299, 0.587, 0.114]) / 255.0 + 1.0

    def preprocess(self, frames, name=0):
        if len(frames) == 1:
            gray = self.rgb2gray(frames[0])
            return np.dstack([gray, gray, gray, gray])

        return np.dstack([self.rgb2gray(frame) for frame in frames])

    def action2string(self, action):
        moveX, moveZ, turn = 0, 0, 0

        """if action == 0:
            moveX = -10
        elif action == 1:
            moveX = 10
        elif action == 2:
            moveZ = -10
        elif action == 3:
            moveZ = 10
        elif action == 4:
            turn = 10
        elif action == 5:
            turn = -10
        elif action == 6:
            pass"""
        if action == 0:
            turn = -10
        elif action == 1:
            turn = 10
        elif action == 2:
            moveZ = 10
        elif action == 3:
            pass

        return "%s %s %s" % (moveX, moveZ, turn)

    def get_frame(self, index):
        if index > len(self.history):
            return self.history[-1]
        else:
            return self.history[-index]

    def process(self, sess, global_t, summary_writer, summary_op, score_input):
        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        # reset accumulated gradients
        sess.run(self.reset_gradients)

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t

        #if USE_LSTM:
        #    start_lstm_state = self.local_network.lstm_state_out

        # t_max times loop
        for i in range(FLAGS.local_t_max):
            #if USE_ALE:
            #    pi_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
            #else:
            pi_, value_ = self.local_network.run_policy_and_value(sess, self.observation)

            #if self.thread_index == 0:
                #print(pi_)
                #cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                #cv2.imshow("img", self.observation)
                #cv2.waitKey(1)

            """if self.thread_index == 0 and len(os.listdir(os.path.join(FLAGS.model_dir, "images"))) < 1000:
                ft = sess.run(self.local_network.col_hiddens[0][0], feed_dict={self.local_network.s: [self.observation]})
                print(ft.shape)

                scipy.misc.imsave("%s/%i-obs.png" % (os.path.join(FLAGS.model_dir, "images"), global_t + i),
                                  self.observation[:, :, 3])

                for m in range(8):
                    img = ft[0, :, :, m]
                    img = img - np.amin(img)
                    img /= np.amax(img)
                    img *= 255.0
                    scipy.misc.imsave("%s/%i-feature-%i.png" % (os.path.join(FLAGS.model_dir, "images"), global_t + i, m),
                                      img)
"""

            action = self.choose_action(pi_)

            states.append(self.observation)
            actions.append(action)
            values.append(value_)

            if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
                print("pi={}".format(pi_))
                print(" V={}".format(value_))

            #if USE_ALE:
            #self.game_state.process(action)
            #reward = self.game_state.reward
            #end_episode = self.game_state.terminal
            #else:

            #for i in range(FLAGS.skip_frames):
            new_obs, reward, end_episode, _ = self.game.step(self.action2string(action))

            if len(self.history) > 10:
                del self.history[0]

            self.history.append(self.rgb2gray(new_obs, global_t + self.local_t))#, "%i-a%i" % (global_t, action)

            def create_history():
                return np.dstack([self.get_frame(1), self.get_frame(2), self.get_frame(3), self.get_frame(4)])

            new_observation = create_history()


            # process game
            #self.game_state.process(action)

            # receive game result
            #reward = self.game_state.reward
            terminal = end_episode#self.game_state.terminal

            self.episode_reward += reward

            # clip reward
            rewards.append(np.clip(reward, -1, 1))

            self.local_t += 1

            #if USE_ALE:
                # s_t1 -> s_t
            #    self.game_state.update()
            #else:


            if terminal:
                terminal_end = True
                print("score={}".format(self.episode_reward))

                self._record_score(sess, summary_writer, summary_op, score_input,
                                   self.episode_reward, global_t)

                self.episode_reward = 0

                #if USE_ALE:
                self.game.reset()
                #else:
                #self.history = [self.rgb2gray(self.game.step(0))]
                #self.observation = create_history()
                #if USE_LSTM:
                #    self.local_network.reset_state()
                break
            else:
                self.observation = new_observation

        R = 0.0
        if not terminal_end:
            #if USE_ALE:
            #    R = self.local_network.run_value(sess, self.game_state.s_t)
            #else:
            R = self.local_network.run_value(sess, self.observation)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # compute and accmulate gradients
        for (ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + FLAGS.gamma * R
            td = R - Vi
            a = np.zeros([FLAGS.action_size])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        sess.run(self.accum_gradients,
                 feed_dict={
                     self.local_network.s: batch_si,
                     self.local_network.a: batch_a,
                     self.local_network.td: batch_td,
                     self.local_network.r: batch_R})

        cur_learning_rate = self._anneal_learning_rate(global_t)

        sess.run(self.apply_gradients,
                 feed_dict={self.learning_rate_input: cur_learning_rate})

        if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
            self.prev_local_t += PERFORMANCE_LOG_INTERVAL
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t
