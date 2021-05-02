from functools import reduce
import random
import time
from typing import NamedTuple

import gym
from matplotlib import pyplot as plt
import numpy as np

env = gym.make('CartPole-v1')

# Fix random seed for reproducible results
random.seed(0)
np.random.seed(0)
env.seed(0)

alpha = 0.4
# No need for discount since episodes always terminate
discount = 1

class StateInfo(NamedTuple):
    # Minimum (approximately) observed value for a piece of state
    min: float
    # Observable range (max - min = range)
    range: float
    # Number of discrete states to divide the space into
    size: int

stateInfos = []
# Cart position
stateInfos.append(StateInfo(-2.4, 4.8, 3))
# Cart velocity
stateInfos.append(StateInfo(-4, 8, 4))
# Pole angle
stateInfos.append(StateInfo(-0.418, 0.836, 4))
# Pole angular velocity
stateInfos.append(StateInfo(-3, 6, 5))

# Total number of states, e.g. 3 * 4 * 4 * 5 = 240
number_states = reduce(lambda n, info: n * info.size, stateInfos, 1)

# Gets discrete space index (0 to 239) from continous space values
def get_state(observation):
    s = 0
    size = number_states
    for i, info in enumerate(stateInfos):
        s_i = min(max(int((observation[i] - info.min) / info.range * info.size), 0), info.size - 1)
        size //= info.size
        s += s_i * size
    return s

# Choose an action with epsilon-greedy policy
def choose_action(q, state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    return np.argmax(q[state])

def sarsa(output = False):
    scores = []
    averages = []
    average = 0
    q = np.zeros((number_states, 2))
    k = 1
    while average < 195:
        epsilon = 10 / k
        done = False
        t = 0
        observation = env.reset()
        s = get_state(observation)
        a = choose_action(q, s, epsilon)
        while not done:
            t += 1
            if output:
                env.render()
            if t == 200:
                break
            observation, reward, done, info = env.step(a)
            s_prime = get_state(observation)
            a_prime = choose_action(q, s_prime, epsilon)
            future = 0
            if not done:
                future = q[s_prime, a_prime]
            q[s, a] += alpha * (reward + discount * future - q[s, a])
            s, a = s_prime, a_prime
        if output:
            print(f'{k}: Episode finished after {t} timesteps.')
        scores.append(t)
        # Average of the last 100 episodes
        average = sum(scores[-100:]) / len(scores[-100:])
        averages.append(average)
        k += 1
    return scores, averages

def q_learning(output = False):
    scores = []
    averages = []
    average = 0
    q = np.zeros((number_states, 2))
    k = 1
    while average < 195:
        epsilon = 1 / k
        done = False
        t = 0
        observation = env.reset()
        s = get_state(observation)
        while not done:
            t += 1
            if output:
                env.render()
            if t == 200:
                break
            a = choose_action(q, s, epsilon)
            observation, reward, done, info = env.step(a)
            s_prime = get_state(observation)
            future = 0
            if not done:
                future = np.amax(q[s_prime])
            q[s, a] += alpha * (reward + discount * future - q[s, a])
            s = s_prime
        if output:
            print(f'{k}: Episode finished after {t} timesteps.')
        scores.append(t)
        # Average of the last 100 episodes
        average = sum(scores[-100:]) / len(scores[-100:])
        averages.append(average)
        k += 1
    return scores, averages

def n_step_sarsa(n, output = False):
    if n < 1:
        return
    scores = []
    averages = []
    average = 0
    q = np.zeros((number_states, 2))
    s = np.zeros(n + 1, dtype=int)
    a = np.zeros(n + 1, dtype=int)
    r = np.zeros(n + 1)
    m = lambda t: t % (n + 1)
    k = 1
    while average < 195:
        epsilon = 1 / k
        T = 200
        t = 0
        observation = env.reset()
        s[0] = get_state(observation)
        a[0] = choose_action(q, s[0], epsilon)
        while t - n + 1 < T:
            if t < T:
                if output:
                    env.render()
                # Don't apply learning algorithm if maximum timestep is reached
                if t + 1 == 200:
                    break
                observation, r[m(t + 1)], done, info = env.step(a[m(t)])
                if done:
                    T = t + 1
                else:
                    s[m(t + 1)] = get_state(observation)
                    a[m(t + 1)] = choose_action(q, s[m(t + 1)], epsilon)
            tau = t - n + 1
            if tau >= 0:
                g = 0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    g += discount ** (i - tau - 1) * r[m(i)]
                if tau + n < T:
                    g += discount ** n * q[s[m(tau + n)], a[m(tau + n)]]
                q[s[m(tau)], a[m(tau)]] += alpha * (g - q[s[m(tau)], a[m(tau)]])
            t += 1
        if output:
            print(f'{k}: Episode finished after {T} timesteps.')
        scores.append(T)
        # Average of the last 100 episodes
        average = sum(scores[-100:]) / len(scores[-100:])
        averages.append(average)
        k += 1
    return scores, averages

time_start = time.perf_counter()
scores, averages = n_step_sarsa(8)
time_end = time.perf_counter()

execution_time = time_end - time_start
print('\n-- n-step Sarsa --')
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

plot_res(scores, averages, 'n-step Sarsa')

env.close()
