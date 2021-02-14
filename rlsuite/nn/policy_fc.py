import torch
import torch.nn as nn
from rlsuite.nn.dqn_archs import VanillaDQN


class PolicyFC(nn.Module):
    # TODO consider renaming class and file because:
    #  1) as target net has exactly same structure
    #  2) consider also that it is used in more than one algorithm
    def __init__(self, features_dim, layers_dim, actions_dim, dqn_arch=VanillaDQN, dropout=0.1):
        super().__init__()
        layers_input_dims = [features_dim] + layers_dim[:-1]
        layers_output_dims = layers_dim
        fc_output_dim = layers_dim[-1]
        self.fc_layers = nn.Sequential(*[nn.Sequential(nn.Linear(input_dim, output_dim),
                                                       # nn.Dropout(p=dropout),  # weird connection in graphs
                                                       nn.ELU())
                                         for input_dim, output_dim in zip(layers_input_dims, layers_output_dims)])
        self.dqn_arch_layers = dqn_arch(fc_output_dim, actions_dim)

    def forward(self, features):
        fc_outputs = self.fc_layers(features)
        x = self.dqn_arch_layers(fc_outputs)

        return x

    def save_checkpoint(self, filename):
        torch.save(self.state_dict(), filename)

    def load_checkpoint(self, filename):
        weights = torch.load(filename)
        self.load_state_dict(weights)
