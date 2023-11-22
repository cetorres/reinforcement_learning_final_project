'''
University of Colorado at Colorado Springs
PhD in Computer Science

Class: CS 4080-5080 - Reinforcement Learning - Fall 2021
Professor: Jugal Kalita
Student: Carlos Eugenio Lopes Pires Xavier Torres
Student ID: 110320128
E-mail: clopespi@uccs.edu
Date: October 20, 2021

Class Project - Final
Product Recommendation in Online Advertising with Reinforcement Learning
'''

from dqn_agent import Agent
import numpy as np
import gym
from recogym import env_1_args
import tensorflow as tf
from tqdm import tqdm
from plot_results import plot, save_plot_image
from os.path import exists
import time

'''
Hyperparameters
'''
REPLAY_MEMORY = 1_000_000
GAMMA = 0.95
ALPHA = 0.001
EPISILON = 0.8 #1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.001 #0.01
BATCH_SIZE = 64
DELTA_HUBER = 1.5

'''
Training parameters
'''
NUM_OFFLINE_USERS = 1_000
NUM_EPISODES = 200
NUM_PRODUCTS = 10
STATE_SIZE = 2 #30
DQN_MODEL_FILE = 'dqn_model.h5'
RESULT_CHART_FILE = f'project_agent_result_{time.time()}.png'
PLOT_UPPER_LIMIT = 10.0
PLOT_LOWER_LIMIT = 0.1


def main():
    tf.compat.v1.disable_eager_execution()
    env = gym.make('reco-gym-v1')
    num_clicks = 0
    num_events = 0
    total_clicks = 0
    total_events = 0
    ctr = 0
    ctr_plot = []
    total_ctr_plot = []
    env.init_gym(env_1_args)
    cur_episode = 0
    
    # Pre-train on 1000 users offline.
    product_views = np.zeros(NUM_PRODUCTS)
    for _ in tqdm(range(NUM_OFFLINE_USERS), unit='users', desc='Generating off-line product views'):
        # Reset env and set done to False.
        env.reset()
        done = False
        observation, reward, done = None, 0, False
        while not done:
            action, observation, reward, done, _ = env.step_offline(observation, reward, done)
            if observation:
                for session in observation.sessions():
                    product_views[session['v']] += 1

    print('product_views', product_views)

    # Create DQN agent
    agent = Agent(gamma=GAMMA, epsilon=EPISILON, alpha=ALPHA,
                input_dims=STATE_SIZE,
                n_actions=NUM_PRODUCTS, product_views=product_views,
                mem_size=REPLAY_MEMORY, batch_size=BATCH_SIZE, delta_huber=DELTA_HUBER,
                epsilon_dec=EPSILON_DECAY, epsilon_end=EPSILON_MIN)
    
    # Load model if exists
    if exists(DQN_MODEL_FILE):
        print('Loading saved model...')
        agent.load_model()

    # Episodes iteration
    progressbar = tqdm(total=NUM_EPISODES, unit='episodes', desc='Training')
    while cur_episode < NUM_EPISODES:
        done = False
        env.reset()
        state, _, done, _ = env.step(None)
        state = [0, 0]
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            agent.learn()

            # Used for calculating click-through rate.
            num_clicks += (1 if reward == 1 and reward is not None else 0)
            num_events += 1

        if num_events > 0:
            ctr = num_clicks / num_events * 100
            if ctr <= PLOT_UPPER_LIMIT and ctr >= PLOT_LOWER_LIMIT:
                ctr_plot.append(ctr)
                total_clicks += num_clicks
                total_events += num_events
                total_ctr = total_clicks / total_events * 100
                total_ctr_plot.append(total_ctr)
                cur_episode += 1
                progressbar.update(1)

        agent.decay_epsilon(cur_episode)
        num_clicks = 0
        num_events = 0

    progressbar.close()
    
    # Save model
    agent.save_model()
    # Plot results
    plot(ctr_plot, total_ctr_plot)
    save_plot_image(RESULT_CHART_FILE)
    # Show final result
    final_ctr = total_clicks / total_events * 100
    print(f"Click-Through Rate: {final_ctr:.2f}%")


def preprocess_state(state):
    state = state.sessions()
    if len(state) > 0:
        state = state[-1:][0]
        state = [state['u'], state['v']]
    else:
        state = [0, 0]
    return state


def preprocess_state2(state):
    state = state.sessions()
    if len(state) > 0:
        base = np.full(STATE_SIZE, -1)
        state_ = [d['v'] for d in state]
        for i in range(min(len(state_), STATE_SIZE)):
            base[i] = state_[i]
        state = base
    else:
        state = np.full(STATE_SIZE, -1)
    return state

if __name__ == '__main__':
    main()
