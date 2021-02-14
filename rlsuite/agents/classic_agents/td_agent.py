from abc import abstractmethod
import math
from rlsuite.agents.classic_agents.classic_agent import ClassicAgent


class TDAgent(ClassicAgent):
    def __init__(self, num_of_actions, dimensions, lr=0.1, gamma=0.999, epsilon=1):
        """
        """
        super().__init__(num_of_actions, dimensions, gamma, epsilon)
        self.lr = lr

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError

    def adjust_lr(self, i_episode):
        self.lr = max(0.01, min(1.0, 1.0 - math.log10((i_episode + 1) / self.ada_divisor)))
