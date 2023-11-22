import gym, recogym
from recogym import env_1_args, Configuration
from copy import deepcopy
from simple_agent import PopularityAgent

env_1_args['random_seed'] = 42

env = gym.make('reco-gym-v1')
env.init_gym(env_1_args)

# Import the random agent.
from recogym.agents import RandomAgent, random_args

# Create the two agents.
num_products = env_1_args['num_products']
popularity_agent = PopularityAgent(Configuration(env_1_args))
agent_rand = RandomAgent(Configuration({
    **env_1_args,
    **random_args,
}))

# Random agent
# Credible interval of the CTR median and 0.025 0.975 quantile.
recogym.test_agent(deepcopy(env), deepcopy(agent_rand), 1000, 1000)

# Popularity agent
# Credible interval of the CTR median and 0.025 0.975 quantile.
recogym.test_agent(deepcopy(env), deepcopy(popularity_agent), 1000, 1000) 