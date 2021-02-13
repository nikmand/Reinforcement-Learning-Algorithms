import torch

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

    def load_checkpoint(self, checkpoint_path):
        log.info("Loading weights from checkpoint.")

        try:
            weights = torch.load(checkpoint_path)
            self.network.load_state_dict(weights)
        except AttributeError:
            log.warning("No file was provided, checkpointing is aborted.")
        except  FileNotFoundError:
            log.warning("File was not found, checkpointing is aborted.")

        return self

    # TODO this implementation is architecture dependent, make it general or implement also the Vanilla DQN case.
    #   Weights used to be loaded from checkpoint, adapt this so it refers to the network weights.
    def init_weights(self):
        raise NotImplementedError
        # log.info("Weights of last layer heads will be reinitialized.")

        # weights["output.value_stream.0.weight"] = torch.rand(1, weights["output.value_stream.0.weight"].size(1),
        #                                                      requires_grad=True)
        # weights["output.value_stream.0.bias"] = torch.rand(1, requires_grad=True)
        #
        # weights["output.advantage_stream.0.weight"] = torch.rand(num_of_actions,
        #                                                          weights["output.advantage_stream.0.weight"].size(1),
        #                                                          requires_grad=True)
        # weights["output.advantage_stream.0.bias"] = torch.rand(num_of_actions, requires_grad=True)

    def build(self, agent_algorithm):
        self.agent = dqn_agent_factory(agent_algorithm, self.network, self.criterion, self.optimizer,
                                       self.num_of_actions, self.gamma, self.eps_decay, self.eps_start, self.eps_end)

        return self.agent


# def __init__(self, dataset_name):
#     self.dataset_name = dataset_name
#     path = os.path.join('data', dataset_name)
#
#     if dataset_name in ['cifar', 'mimgnet', 'aircraft', 'quickdraw', 'vgg_flower']:
#       # meta-training set
#       x = np.load(os.path.join(path, 'train.npy'), encoding='bytes')
#       self.C_mtr = len(x)
#       self.N_mtr = [len(xx) for xx in x]
#       self.x_mtr = [np.reshape(x[i], [self.N_mtr[i], -1]) for i in range(self.C_mtr)]
#
#     # meta-validation set
#     if not dataset_name in ['traffic', 'fashion-mnist']:
#       x = np.load(os.path.join(path, 'valid.npy'), encoding='bytes')
#       self.C_mval = len(x)
#       self.N_mval = [len(xx) for xx in x]
#       self.x_mval = [np.reshape(x[i], [self.N_mval[i], -1]) for i in range(self.C_mval)]
#
#     # meta-test set
#     x = np.load(os.path.join(path, 'test.npy'), encoding='bytes')
#     self.C_mte = len(x)
#     self.N_mte = [len(xx) for xx in x]
#     self.x_mte = [np.reshape(x[i], [self.N_mte[i], -1]) for i in range(self.C_mte)]