import math
from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, num_of_actions, gamma=0.999, epsilon=1):
        """
        """
        self.num_of_actions = num_of_actions  # TODO remove this it can be deduced.
        self.gamma = gamma  # usually 0.8 - 0.99
        self.epsilon = epsilon

    @abstractmethod
    def choose_action(self, cur_state):
        raise NotImplementedError

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError
