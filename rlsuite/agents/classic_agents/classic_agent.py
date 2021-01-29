from abc import abstractmethod

import numpy as np
import random
import math
from rlsuite.agents.agent import Agent


class ClassicAgent(Agent):
    """
    """
    def __init__(self, num_of_actions, dimensions, gamma=0.999, epsilon=1):

        super().__init__(num_of_actions, gamma, epsilon)
        # q_table has the dimensions of each variable (state) and an extra one for every possible action from each state
        self.q_table = np.zeros(dimensions + [num_of_actions])
        self.ada_divisor = 25  # np.prod(self.q_table.shape[:-1])

    def adjust_exploration(self, i_episode):
        """  """
        self.epsilon = max(0.01, min(0.3, 1.0 - math.log10((i_episode + 1) / self.ada_divisor)))
        #  /= np.sqrt(i_episode + 1)

    def choose_action(self, cur_state, train=True):
        """  """
        if random.uniform(0, 1) < self.epsilon and train:
            return random.randrange(self.num_of_actions)
        else:
            return np.argmax(self.q_table[cur_state])

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError

    # TODO implement those methods so as q_table is saved in a pickle and restored
    def save_checkpoint(self, filename):
        pass
    #     with open(filename, 'wb') as f:
    #         pickle.dump([X_train, y_train], f)

    def load_checkpoint(self, filename):
        pass
    #     with open(filename, 'rb') as f:
    #         var_you_want_to_load_into = pickle.load(f)
