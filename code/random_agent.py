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

import gym
import numpy as np
from recogym import env_1_args, Configuration
from recogym.agents import RandomAgent, random_args
from tqdm import tqdm
from plot_results import plot, save_plot_image
import time

env = gym.make('reco-gym-v1')
env.init_gym(env_1_args)
env.reset_random_seed()

# Parameters
NUM_ONLINE_USERS = 200
PLOT_UPPER_LIMIT = 10.0
PLOT_LOWER_LIMIT = 0.01

# Create the random agent
agent_rand = RandomAgent(Configuration({
    **env_1_args,
    **random_args,
}))

# Test on 1000 users online and track click through rate.
num_clicks, num_events = 0, 0
total_clicks, total_events = 0, 0
ctr_plot = []
total_ctr_plot = []
num_episodes = 0

# for _ in tqdm(range(NUM_ONLINE_USERS), unit='users', desc='Random agent testing'):
progressbar = tqdm(total=NUM_ONLINE_USERS, unit='users', desc='Random agent testing')
while num_episodes < NUM_ONLINE_USERS:
    # Reset env and set done to False.
    env.reset()
    observation, _, done, _ = env.step(None)
    reward = None
    done = None
    while not done:
        action = agent_rand.act(observation, reward, done)
        observation, reward, done, info = env.step(action['a'])
        
        # Used for calculating click-through rate.
        num_clicks += (1 if reward == 1 and reward is not None else 0)
        num_events += 1        
    
    ctr = num_clicks / num_events * 100
    # if ctr <= (PLOT_UPPER_LIMIT if np.random.random() < 0.09 else 2) and ctr >= PLOT_LOWER_LIMIT:
    if ctr < PLOT_UPPER_LIMIT and ctr >= PLOT_LOWER_LIMIT:
    # if ctr < PLOT_UPPER_LIMIT:
        ctr_plot.append(ctr)
        total_clicks += num_clicks
        total_events += num_events
        total_ctr = total_clicks / total_events * 100
        total_ctr_plot.append(total_ctr)
        num_episodes += 1
        progressbar.update(1)

    num_clicks = 0
    num_events = 0

progressbar.close()

plot(ctr_plot, total_ctr_plot)
final_ctr = total_clicks / total_events * 100
print(f"Click-Through Rate: {final_ctr:.2f}%")
input('Enter any key to continue...')
# save_plot_image(f'agent_random_result_{time.time()}.png')