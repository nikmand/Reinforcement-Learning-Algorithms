from rlsuite.agents.nn_agents.dqn_agents import DDQNAgent, DQNAgent
from rlsuite.nn.dqn_archs import Dueling, VanillaDQN
from rlsuite.utils.constants import TypesOfMemory, DeepRLAgents, NetArchitectures
from rlsuite.utils.memory import MemoryPER, Memory


# TODO: factory probably won't be the same for all algorithms as they may need different args
#   an extra abstraction is probably needed here.

def dqn_agent_factory(dqn_agent, network, criterion, optimizer, num_of_actions, gamma, eps_decay, eps_start, eps_end, gpu):
    """  """
    if dqn_agent == DeepRLAgents.DQN:
        agent_algorithm = DQNAgent
    elif dqn_agent == DeepRLAgents.DDQN:
        agent_algorithm = DDQNAgent
    else:
        raise ValueError("DQN Agent option {} is not supported".format(dqn_agent))

    agent = agent_algorithm(num_of_actions, network, criterion, optimizer, gamma, eps_decay, eps_start, eps_end, gpu)

    return agent


def memory_factory(mem_type, mem_size, batch_size):
    """  """
    if mem_type == TypesOfMemory.PER:
        memory = MemoryPER(mem_size, batch_size)
    elif mem_type == TypesOfMemory.VANILLA:
        memory = Memory(mem_size, batch_size)
    else:
        raise ValueError("Memory option {} is not supported".format(mem_type))

    return memory


def arch_factory(arch):
    """  """
    if arch == NetArchitectures.DUELING:
        dqn_arch = Dueling
    elif arch == NetArchitectures.VANILLA:
        dqn_arch = VanillaDQN
    else:
        raise ValueError("Architecture option {} is not supported".format(arch))

    return dqn_arch
