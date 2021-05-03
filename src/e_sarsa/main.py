### Reference: https://www.geeksforgeeks.org/expected-sarsa-in-reinforcement-learning/

import random
import time
import gym
import numpy as np

from expected_sarsa import ExpectedSarsaAgent
from matplotlib import pyplot as plt

from state import number_states

# Using the gym library to create the environment
env = gym.make('CartPole-v1')

# Fix random seed for reproducible results
random.seed(0)
np.random.seed(0)
env.seed(0)

# Defining some of the required parameters
max_steps = 500
alpha = 0.4
gamma = 1

time_start = time.perf_counter()

"""
    The two parameters below is used to calculate
    the reward
"""
episodeReward = 0
scores = []

# Defining all the three agents

# define averages for rewards after 100 episodes
averages = []
average = 0 # average for 100 episodes

# Now we run all the episodes and calculate the reward obtained at the end of the episode

k = 1

expectedSarsaAgent = ExpectedSarsaAgent(
        0, alpha, gamma, number_states,
        env.action_space.n, env.action_space)

while average < 195:
    # Initialize the necessary parameters before
    # the start of the episode
    epsilon = 1 / k
    expectedSarsaAgent.epsilon = epsilon
    done = False
    t = 0
    state1 = env.reset()
    action1 = expectedSarsaAgent.choose_action(state1)
    episodeReward = 0

    # while t < max_steps:
    while not done:

        # Getting the next state, reward, and other parameters
        state2, reward, done, info = env.step(action1)

        # Choosing the next action
        action2 = expectedSarsaAgent.choose_action(state2)

        # Learning the Q-value
        expectedSarsaAgent.update(state1, state2, reward, action1, action2)

        state1 = state2
        action1 = action2

        # Updating the respective vaLues
        t += 1

    # Append the sum of reward at the end of the episode
    # scores.append(episodeReward)
    scores.append(t)
    average = sum(scores[-100:]) / len(scores[-100:])
    averages.append(average)
    k += 1

time_end = time.perf_counter()

execution_time = time_end - time_start

print('\n-- Expected Sarsa --')
print(f'Number of episodes before convergence: {len(scores)}')
print(f'Total execution time: {execution_time:.4f}s')
print(f'Execution time per 1000 time steps: {execution_time * 1000 / sum(scores):.4f}s\n')

def plot_res(values, averages, title=''):
    ''' Plot the reward curve and histogram of results over time.'''
    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].plot(averages, label='average')
    ax[0].axhline(195, c='red',ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x,p(x),"--", label='trend')
    except:
        print('')

    # Plot the histogram of results
    ax[1].hist(values)
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()

plot_res(scores, averages, 'Expected Sarsa')

env.close()
