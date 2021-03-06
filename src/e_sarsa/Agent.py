### Reference: https://www.geeksforgeeks.org/expected-sarsa-in-reinforcement-learning/

import numpy as np
from state import get_state
from state import number_states

class Agent:
    """
    The Base class that is implemented by
    other classes to avoid the duplicate 'choose_action'
    method
    """

    def choose_action(self, state, epsilon):
        action = 0
        if np.random.uniform(0, 1) < epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Q[get_state(state), :])
        return action