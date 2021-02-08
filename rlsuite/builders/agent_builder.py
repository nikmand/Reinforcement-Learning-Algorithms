from rlsuite.builders.factories import dqn_agent_factory, arch_factory
from rlsuite.nn.policy_fc import PolicyFC
import logging.config
from rlsuite.utils.constants import LOGGER
from rlsuite.examples.cartpole.cartpole_constants import LOGGER_PATH

# TODO properly configure logger path
logging.config.fileConfig(LOGGER_PATH)
log = logging.getLogger(LOGGER)


# TODO can we have a general abstract builder ?
class AgentBuilder:

    def __int__(self):
        pass


class DQNAgentBuilder:
    """ Builder takes over the construction of a DQN Agent object. """

    def __init__(self, num_of_observations, num_of_actions, gamma, eps_decay, eps_start, eps_end):
        self.network = None
        self.optimizer = None
        self.criterion = None
        self.agent = None
        self.num_of_observations = num_of_observations
        self.num_of_actions = num_of_actions
        self.gamma = gamma
        self.eps_decay = eps_decay
        self.eps_start = eps_start
        self.eps_end = eps_end

    def build_optimizer(self, optimizer, lr):
        self.optimizer = optimizer(self.network.parameters(), lr)

        return self

    def set_criterion(self, criterion):
        self.criterion = criterion

        return self

    def build_network(self, layers_dims, arch):  # NOTE consider having a builder for Nets
        arch = arch_factory(arch)
        self.network = PolicyFC(self.num_of_observations, layers_dims, self.num_of_actions, arch, dropout=0)

        log.debug("Number of net parameters: {}".format(sum(x.numel() for x in self.network.parameters())))

        return self

    def build(self, agent_algorithm):
        self.agent = dqn_agent_factory(agent_algorithm, self.network, self.criterion, self.optimizer,
                                       self.num_of_actions, self.gamma, self.eps_decay, self.eps_start, self.eps_end)

        return self.agent
