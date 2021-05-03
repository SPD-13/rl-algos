from typing import NamedTuple
from functools import reduce

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