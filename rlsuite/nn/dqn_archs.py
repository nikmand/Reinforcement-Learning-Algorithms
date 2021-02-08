import torch
import torch.nn as nn


class VanillaDQN(nn.Module):
    """ Vanilla implementation of DQN. A single stream is used to estimate the action value function Q(s,a). """

    def __init__(self, output_dim, actions_dim):
        super().__init__()
        self.output = nn.Linear(output_dim, actions_dim)

    def forward(self, x):
        return self.output(x)


class Dueling(nn.Module):
    """ The action value function Q(s,a) can be decomposed to state value V(s) and advantage A(s,a) for each action.
    With Dueling, we want to separate the estimation of V(s) and A(s,a) by using two streams. """

    def __init__(self, output_dim, actions_dim):
        super().__init__()
        # TODO consider refactoring the network arch, streams' layers should be configurable
        self.value_stream = nn.Sequential(
            # nn.Linear(output_dim, 12),
            # nn.Dropout(p=0.1),
            # nn.ReLU(),
            # nn.Linear(12, 1)
            nn.Linear(output_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            # nn.Linear(output_dim, 12),
            # nn.Dropout(p=0.1),
            # nn.ReLU(),
            # nn.Linear(12, actions_dim)
            nn.Linear(output_dim, actions_dim)
        )

    def forward(self, x):
        """ The outcome of the first layers of the neural net is passed to both the value and advantage stream.
        Then the results of the two streams are merged and the action value function Q(s,a) is returned. """
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)

        q_values = values + (advantages - advantages.mean())  # broadcasting happens on values

        return q_values
