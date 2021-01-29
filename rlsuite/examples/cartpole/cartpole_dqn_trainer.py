import torch
import logging.config
import torch.optim as optim
import gym
from rlsuite.examples.cartpole import cartpole_constants
from rlsuite.examples.cartpole.cartpole_constants import LOGGER_PATH
from rlsuite.utils.memory import Memory, MemoryPER
from rlsuite.nn.policy_fc import PolicyFC
from rlsuite.nn.dqn_archs import ClassicDQN, Dueling
from rlsuite.agents.nn_agents.dqn_agents import DQNAgent, DDQNAgent
from rlsuite.utils.constants import LOGGER
from rlsuite.trainers.trainer import Trainer


# WIP Attempt to solve the cartpole problem by using the trainer abstraction

logging.config.fileConfig(LOGGER_PATH)
log = logging.getLogger(LOGGER)

if __name__ == "__main__":

    env = gym.make(cartpole_constants.environment)
    num_of_observations = env.observation_space.shape[0]
    num_of_actions = env.action_space.n

    lr = 1e-3
    layers_dim = [24, 48]
    dropout = 0
    gamma = 1
    eps_decay, eps_start, eps_end = 0.001, 1, 0
    mem_size = 15_000

    dueling = True  # Classic and Dueling DQN architectures are supported
    per = True
    double = True

    if dueling:
        dqn_arch = Dueling
    else:
        dqn_arch = ClassicDQN

    network = PolicyFC(num_of_observations, layers_dim, num_of_actions, dqn_arch, dropout)

    log.debug("Number of parameters in our model: {}".format(sum(x.numel() for x in network.parameters())))

    criterion = torch.nn.MSELoss(reduction='none')  # torch.nn.SmoothL1Loss()  # Huber loss
    optimizer = optim.Adam(network.parameters(), lr)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.9, patience=20)  # not used in update for now
    # ExponentialLR(optimizer, lr_decay)  # alternative scheduler
    # scheduler will reduce the lr by the specified factor when metric has stopped improving

    if per:
        memory = MemoryPER(mem_size)
    else:
        memory = Memory(mem_size)

    if double:
        agent = DDQNAgent(num_of_actions, network, criterion, optimizer, gamma, eps_decay, eps_start, eps_end)
    else:
        agent = DQNAgent(num_of_actions, network, criterion, optimizer, gamma, eps_decay, eps_start, eps_end)

    config = None
    trainer = Trainer(env, agent, memory, config)
    trainer.run()

    config_env, config_agent, config_misc = config_parser(args.config_file)

