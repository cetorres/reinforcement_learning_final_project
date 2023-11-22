'''
University of Colorado at Colorado Springs
PhD in Computer Science

Class: CS 4080-5080 - Reinforcement Learning - Fall 2021
Professor: Jugal Kalita
Student: Carlos Eugenio Lopes Pires Xavier Torres
Student ID: 110320128
E-mail: clopespi@uccs.edu
Date: October 20, 2021

Class Project - Midterm
Product Recommendation in Online Advertising with Reinforcement Learning
'''

import numpy as np
from numpy.random import choice
import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def build_dqn(alpha, n_actions, input_dims, delta_huber):
    # model = keras.Sequential([
    #     keras.layers.Dense(layer1_dims, input_shape=(input_dims,), activation='relu'),
    #     keras.layers.Dense(layer2_dims, activation='relu'),
    #     keras.layers.Dense(n_actions, activation='linear')])
    # model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=alpha))
    model = keras.Sequential()
    model.add(keras.layers.Dense(256, activation='relu', input_dim=input_dims))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(n_actions, activation='softmax'))
    model.compile(loss=keras.losses.Huber(delta=delta_huber), optimizer=Adam(learning_rate=alpha), metrics=['accuracy'])
    model.summary()
    return model


class Agent():
    def __init__(self, alpha=0.001, gamma=0.001, n_actions=0, epsilon=0, batch_size=0,
                input_dims=0, epsilon_dec=1e-3, epsilon_end=0.01, product_views=0,
                mem_size=1_000_000, model_file='dqn_model.h5', delta_huber=1.0):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.eps_max = epsilon
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = model_file
        self.n_actions = n_actions
        self.product_views = product_views
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, delta_huber)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            # action = np.random.choice(self.action_space)
            product_prob = self.product_views / sum(self.product_views)
            action = choice(self.n_actions, p=product_prob)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(next_states)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1)*dones

        loss = self.q_eval.train_on_batch(states, q_target)
        # self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        return loss

    def decay_epsilon(self, t):
        self.epsilon = max(self.eps_min, min(self.eps_max, 1.0 - math.log10((t+1) / self.eps_dec)))

    def save_model(self):
        self.q_eval.save(self.model_file)
        print(f"Model saved: {self.model_file}")

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        print(f"Model loaded: {self.model_file}")
        self.q_eval.summary()

