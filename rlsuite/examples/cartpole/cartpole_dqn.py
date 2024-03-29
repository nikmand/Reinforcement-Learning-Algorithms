import torch
import torch.optim as optim
import gym
import numpy as np
import logging.config
from rlsuite.builders.agent_builder import DQNAgentBuilder
from rlsuite.builders.factories import memory_factory
from rlsuite.examples.cartpole import cartpole_constants
from rlsuite.examples.cartpole.cartpole_constants import check_termination, LOGGER_PATH, LOG_WEIGHTS
from rlsuite.utils.functions import plot_rewards, plot_rewards_completed, plot_epsilon, log_parameters_histograms
from rlsuite.utils.constants import LOGGER
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.config.fileConfig(LOGGER_PATH)
log = logging.getLogger(LOGGER)

if __name__ == "__main__":

    tensorboard_writer = None
    if cartpole_constants.USE_TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_writer = SummaryWriter()

    # TODO name of experiment can be formed to include hparams

    env = gym.make(cartpole_constants.environment)
    num_of_observations = env.observation_space.shape[0]
    num_of_actions = env.action_space.n

    # TODO better to use configuration files for such parameters
    lr = 1e-3
    layers_dim = [24, 48]  # 24, 48
    dropout = 0
    gamma = 1
    eps_decay, eps_start, eps_end = 0.001, 1, 0
    mem_size = 10_000
    target_net_update_period = 100  # target net is updated with the weights of policy net once every n updates (steps)
    batch_size = 32
    arch = 'dueling'
    mem_type = 'vanilla'
    agent_algorithm = 'ddqn'
    gpu = False

    criterion = torch.nn.MSELoss(reduction='none')  # torch.nn.SmoothL1Loss()  # Huber loss
    optimizer = optim.Adam
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.9, patience=20)  # not used in update for now
    # ExponentialLR(optimizer, lr_decay)  # alternative scheduler
    # scheduler will reduce the lr by the specified factor when metric has stopped improving

    memory = memory_factory(mem_type, mem_size, batch_size)

    agent = DQNAgentBuilder(num_of_observations, num_of_actions, gamma, eps_decay, eps_start, eps_end, gpu)\
        .set_criterion(criterion)\
        .build_network(layers_dim, arch) \
        .build_optimizer(optimizer, lr) \
        .build(agent_algorithm)

    steps_done = 0
    train_rewards, eval_rewards = {}, {}
    epsilon_at_end_of_each_episode = []

    for i_episode in range(cartpole_constants.max_episodes):

        # Initialize the environment and state
        state = env.reset()
        state = np.float32(state)

        episode_duration = 0
        episode_reward = 0
        episode_loss = 0

        done = False
        train = True
        agent.train_mode()

        if (i_episode + 1) % cartpole_constants.EVAL_INTERVAL == 0:
            train = False
            agent.eval_mode()

        while not done:
            episode_duration += 1
            # if not train:
            #     env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.float32(next_state)
            memory.store(state, action, next_state, reward, done)  # Store the transition in memory
            state = next_state
            episode_reward += reward

            # if memory.tree.n_entries < 500:
            #     # DQN paper starts from a partially loaded memory as in the beginning it just collects experiences
            #     # we do nothing until we collect a number of experiences
            #     continue
            if train:
                steps_done += 1
                try:
                    transitions, indices, is_weights = memory.sample()
                except ValueError:
                    continue
                loss, errors = agent.update(transitions, is_weights)  # Perform one step of optimization on the policy net
                episode_loss += loss
                agent.adjust_exploration(steps_done)  # rate is updated at every step - taken from the tutorial
                memory.batch_update(indices, errors)
                if steps_done % target_net_update_period == 0:
                    # Update the target network, the frequency of the update had crucial impact
                    agent.update_target_net()
                    if cartpole_constants.USE_TENSORBOARD and LOG_WEIGHTS:
                        log_parameters_histograms(tensorboard_writer, agent.target_net, i_episode, 'TargetNet')

        if train:
            train_rewards[i_episode] = episode_reward
            if cartpole_constants.USE_TENSORBOARD:
                tensorboard_writer.add_scalar('Agent/Loss', episode_loss / episode_duration, i_episode)
                tensorboard_writer.add_scalar('Agent/Reward Train', episode_reward, i_episode)
                tensorboard_writer.flush()
                if LOG_WEIGHTS:
                    log_parameters_histograms(tensorboard_writer, agent.policy_net, i_episode, 'PolicyNet')

        else:
            eval_rewards[i_episode] = episode_reward
            if cartpole_constants.USE_TENSORBOARD:
                tensorboard_writer.add_scalar('Agent/Reward Eval', episode_reward, i_episode)
                tensorboard_writer.flush()
            if check_termination(eval_rewards):
                log.info('Solved after {} episodes.'.format(len(train_rewards)))
                break

        if cartpole_constants.USE_TENSORBOARD:
            tensorboard_writer.add_scalar('Agent/Epsilon', agent.epsilon, i_episode)
            tensorboard_writer.add_scalar('Agent/Steps', steps_done, i_episode)
            tensorboard_writer.flush()
        else:
            plot_rewards(train_rewards, eval_rewards)
            epsilon_at_end_of_each_episode.append(agent.epsilon)
            plot_epsilon(epsilon_at_end_of_each_episode)

    else:
        log.info("Unable to reach goal in {} training episodes.".format(len(train_rewards)))

    figure = plot_rewards_completed(train_rewards, eval_rewards)
    # figure.show()
    if cartpole_constants.USE_TENSORBOARD:
        tensorboard_writer.add_figure('Plot', figure)

        state = np.float32(env.reset())
        tensorboard_writer.add_graph(agent.policy_net, torch.tensor(state, device=agent.device))

        # first dict with hparams, second dict with metrics
        tensorboard_writer.add_hparams({'lr': lr, 'gamma': gamma, 'HL Dims': str(layers_dim), 'Algorithm': agent_algorithm, 'Arch': arch,
                            'Mem Type': mem_type, 'Mem Size': mem_size, 'Target_upd_interval': target_net_update_period,
                            'Batch Size': batch_size, 'EPS_DECAY': eps_decay},
                                       {'episodes_needed': len(train_rewards)})
        tensorboard_writer.flush()
        tensorboard_writer.close()

    env.close()
