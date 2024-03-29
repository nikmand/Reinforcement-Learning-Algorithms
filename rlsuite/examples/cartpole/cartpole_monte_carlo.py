import gym
from rlsuite.examples.cartpole import cartpole_constants
from rlsuite.examples.cartpole.cartpole_constants import check_termination, LOGGER_PATH
from rlsuite.agents.classic_agents.mc_agent import MCAgent
import logging.config
from rlsuite.utils.quantization import Quantization
from rlsuite.utils.functions import plot_rewards, plot_rewards_completed
import matplotlib.pyplot as plt
from rlsuite.utils.constants import LOGGER

# COMMENT it seems that monte carlo has high variance maybe we should reduce exploration
logging.config.fileConfig(LOGGER_PATH)
logger = logging.getLogger(LOGGER)
if __name__ == "__main__":

    env = gym.make(cartpole_constants.environment)
    train_durations = {}
    eval_durations = {}

    num_of_actions = env.action_space.n

    dimensions_high_barriers = env.observation_space.high
    dimensions_low_barriers = env.observation_space.low

    # if we want to exclude one dimension we can set freq=1

    dimensions_description = list(zip(dimensions_low_barriers, dimensions_high_barriers, cartpole_constants.var_freq))

    quantizator = Quantization(dimensions_description)

    agent = MCAgent(num_of_actions, quantizator.dimensions)

    for i_episode in range(cartpole_constants.max_episodes):
        # Initialize the environment and state
        done = False
        train = True
        if (i_episode + 1) % cartpole_constants.EVAL_INTERVAL == 0:
            train = False

        next_observation = env.reset()
        agent.adjust_exploration(i_episode)
        state_action_ls = []
        reward_ls = []

        t = 0
        while not done:
            t += 1
            # env.render()
            state = quantizator.digitize(next_observation)
            action = agent.choose_action(state, train=train)    # Select and perform an action
            next_observation, reward, done, _ = env.step(action)

            state_action_ls.append((state, action))
            reward_ls.append(reward)

        if train:
            train_durations[i_episode] = (t + 1)
            discounted_rewards = agent.calculate_rewards(reward_ls)
            agent.update(state_action_ls, discounted_rewards)
        else:
            eval_durations[i_episode] = (t + 1)
            if check_termination(eval_durations):
                logger.info('Solved after {} episodes.'.format(len(train_durations)))
                break
        plot_rewards(train_durations, eval_durations)

    else:
        logger.info("Unable to reach goal in {} training episodes.".format(len(train_durations)))

    plot_rewards_completed(train_durations, eval_durations)
    env.close()
    plt.show()
