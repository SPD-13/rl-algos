from functools import reduce
import random
from typing import NamedTuple

import gym
import numpy as np

env = gym.make('CartPole-v1')

alpha = 0.4
# No need for discount since episodes always terminate
discount = 1

#position_range = env.observation_space.high[0] - env.observation_space.low[0]
#position_min = env.observation_space.low[0]
#angle_range = env.observation_space.high[2] - env.observation_space.low[2]
#angle_min = env.observation_space.low[2]

# Max cart velocity ~= 4.370761097745926
# Min cart velocity ~= -0.4184858819204502
# Max pole angular velocity ~= 0.9178660833775616
# Min pole angular velocity -3.1944354323964013

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
    action = 0 if q[state, 0] > q[state, 1] else 1
    if random.random() < epsilon:
        action = (action + 1) % 2
    return action

# Choose an action with greedy policy
def choose_action_max(q, state, epsilon):
    action = 0 if q[state, 0] > q[state, 1] else 1
    return action

def sarsa():
    q = np.zeros((number_states, 2))
    for k in range(1000):
        epsilon = 1 / (k + 1)
        done = False
        t = 0
        observation = env.reset()
        s = get_state(observation)
        a = choose_action(q, s, epsilon)
        while not done:
            t += 1
            env.render()
            observation, reward, done, info = env.step(a)
            s_prime = get_state(observation)
            a_prime = choose_action(q, s_prime, epsilon)
            a_max = choose_action_max(q, s_prime, epsilon)
            q[s, a] += alpha * (reward + discount * q[s_prime, a_prime] - q[s, a])
            s, a = s_prime, a_prime
        print(f'{k + 1}: Episode finished after {t} timesteps.')

def nStepSarsa(n):
    if n < 1:
        return
    q = np.zeros((number_states, 2))
    s = np.zeros(n + 1, dtype=int)
    a = np.zeros(n + 1, dtype=int)
    r = np.zeros(n + 1)
    m = lambda t: t % (n + 1)
    for k in range(1000):
        epsilon = 1 / (k + 1)
        T = 500
        t = 0
        observation = env.reset()
        s[0] = get_state(observation)
        a[0] = choose_action(q, s[0], epsilon)
        while t - n + 1 < T:
            if t < T:
                env.render()
                observation, r[m(t + 1)], done, info = env.step(a[m(t)])
                if done:
                    T = t + 1
                    # Don't apply learning algorithm if maximum timestep is reached
                    if T == 500:
                        t = T + n - 1
                        break
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
        print(f'{k + 1}: Episode finished after {t - n + 1} timesteps.')

#sarsa()
nStepSarsa(8)
env.close()
