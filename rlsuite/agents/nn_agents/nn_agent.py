import torch
from rlsuite.agents.agent import Agent


class NeuralAgent(Agent):

    def __init__(self, num_of_actions, network, criterion, optimizer, gamma=0.999, eps_decay=0.0005, eps_start=1,
                 eps_end=0.01, use_gpu=False):
        super().__init__(num_of_actions, gamma, epsilon=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def choose_action(self, cur_state):
        pass

    def update(self, *args):
        pass

    # TODO consider if this abstraction is useful, there are differences between the nn agents
