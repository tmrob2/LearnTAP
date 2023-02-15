from typing import Optional, Tuple
import random
import numpy as np


class Agent:
    def __init__(self, idx, initial_loc: Optional[Tuple] = None,
        maximum_penalty = 100):
        if initial_loc is None:
            # select a random x in range(11)
            initial_loc = (0, random.randint(0, 11))
        xinit, yinit = initial_loc
        self.current_state = np.array([xinit, yinit], dtype=np.int32)
        self.initial_loc = np.array(list(initial_loc), dtype=np.int32)
        if idx == 0:
            self.active = True
        else:
            self.active = False
        self.agent_idx = idx
        self.maximum_penalty = maximum_penalty
        self.initial_penalty = 0
        self.penalty = 0

    def reset(self):
        if self.agent_idx == 0:
            self.active = True
        else:
            self.active = False
        self.current_state = self.initial_loc
        self.penalty = 0

    def receive_penalty(self, penalty):
        self.penalty += penalty