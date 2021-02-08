from enum import Enum

LOGGER = 'simpleExample'

logger_path = ""


class NetArchitectures(str, Enum):
    DUELING = 'dueling'
    VANILLA = 'vanilla'


class TypesOfMemory(str, Enum):
    PER = 'per'
    VANILLA = 'vanilla'


class DeepRLAgents(str, Enum):
    ACTOR_CRITIC = 'actor_critic'
    POLICY = 'policy_grad'
    DQN = 'dqn'
    DDQN = 'ddqn'


class RLAlgorithms(Enum):
    Q_LEARNING = '0'
    SARSA = '1'
    DOUBLE_Q_LEARNING = '2'


# agent
LR = 'lr'
LAYERS_DIM = 'layers_dim'
TARGET_UPDATE = 'target_update'
BATCH_SIZE = 'batch_size'
GAMMA = 'gamma'
ARCH = 'arch'
ALGO = 'algo'
MEM_SIZE = 'mem_size'
MEM_PER = 'mem_type'
EPS_DECAY = 'eps_decay'
EPS_START = 'eps_start'
EPS_END = 'eps_end'
CHECKPOINT = 'checkpoint'
WEIGHTS = 'weights'
