import gym
import logging.config
import matplotlib.pyplot as plt
from rlsuite.examples.cartpole.cartpole_constants import out_of_bounds, environment, var_freq, LOGGER_PATH, \
    max_episodes, EVAL_INTERVAL
from rlsuite.utils.constants import RLAlgorithms
from rlsuite.utils.quantization import Quantization
from rlsuite.agents.classic_agents.td_agents import QAgent, DoubleQAgent, SARSAgent
from rlsuite.utils.functions import plot_rewards
from itertools import count
from rlsuite.utils.constants import LOGGER

logging.config.fileConfig(LOGGER_PATH)
logger = logging.getLogger(LOGGER)

if __name__ == "__main__":

    env = gym.make(environment)
    dimensions_high_barriers = env.observation_space.high
    dimensions_low_barriers = env.observation_space.low
    num_of_actions = env.action_space.n

    logger.debug(dimensions_high_barriers)
    logger.debug(dimensions_low_barriers)

    dimensions_description = list(zip(dimensions_low_barriers, dimensions_high_barriers, var_freq))
    quantizator = Quantization(dimensions_description)

    logger.debug(quantizator.dimensions_bins)

    algorithm = RLAlgorithms.DOUBLE_Q_LEARNING

    if algorithm == RLAlgorithms.Q_LEARNING:
        agent = QAgent(num_of_actions, quantizator.dimensions)
    elif algorithm == RLAlgorithms.SARSA:
        agent = SARSAgent(num_of_actions, quantizator.dimensions)
    elif algorithm == RLAlgorithms.DOUBLE_Q_LEARNING:
        agent = DoubleQAgent(num_of_actions, quantizator.dimensions)
    else:
        raise NotImplementedError

    logger.debug(quantizator.dimensions)
    logger.debug(agent.q_table.shape)

    train_durations = {}
    eval_durations = {}
    means = []

    for i_episode in range(max_episodes):

        train = True
        if (i_episode + 1) % EVAL_INTERVAL == 0:
            train = False

        observation = env.reset()  #
        agent.adjust_exploration(i_episode)
        agent.adjust_lr(i_episode)

        state = quantizator.digitize(observation)
        action = agent.choose_action(state, train=train)

        for step in count():  # range(constants.max_steps):  # consider ending only on fail ?
            # env.render()
            observation, reward, done, info = env.step(action)  # takes the specified action
            if done:
                position = observation[0]
                rotation = observation[2]
                if train:
                    train_durations[i_episode] = (step + 1)
                else:
                    eval_durations[i_episode] = (step + 1)

                plot_rewards(train_durations, eval_durations)
                if out_of_bounds(position):
                    logger.debug("Terminated due to position")
                # print("Episode {} terminated after {} timesteps".format(i_episode, step + 1))
                break

            new_state = quantizator.digitize(observation)
            new_action = agent.choose_action(new_state, train=train)

            agent.update(state, action, reward, new_state, new_action)  # if q-learning new action is not going to be used

            state = new_state
            action = new_action
        else:
            # print("Episode {} finished successful!".format(i_episode))
            pass

    env.close()
    plt.show()
