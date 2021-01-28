import torch.nn as nn
from rlsuite.nn.dqn_archs import ClassicDQN


class PolicyFC(nn.Module):
    # TODO consider renaming class and file, it is just a FC network and it can be used anywhere
    def __init__(self, features_dim, layers_dim, actions_dim, dqn_arch=ClassicDQN, dropout=0.1):
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
