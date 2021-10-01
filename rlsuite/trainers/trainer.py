import numpy as np
from abc import ABC, abstractmethod

from rlsuite.nn.dqn_archs import Dueling, VanillaDQN
from rlsuite.utils.constants import *
from rlsuite.utils.functions import log_parameters_histograms
from rlsuite.utils.memory import MemoryPER, Memory


class Trainer(ABC):
    """  """

    def __init__(self, env, agent, config, writer=None):

        # TODO agent and memory algorithms should be specified and their creation should happen here
        self.env = env
        self.writer = writer

        if config[MEM_PER] == 'per':
            memory = MemoryPER(config["mem_size"])
        else:
            memory = Memory(config["mem_size"])

        if config[ARCH] == 'dueling':
            dqn_arch = Dueling
        else:
            dqn_arch = VanillaDQN



        self.memory = memory
        self.max_episodes = config["max_ep"]
        self.eval_interval = config["eval_interval"]
        # TODO ckeck if evaluation is needed
        self.target_update_period = config["target_update"]
        self.batch_size = config["batch"]
        self.render = config["render"]

    def _should_eval(self, i_episode):
        should_eval = (i_episode + 1) % self.eval_interval == 0

        return should_eval

    def _should_update_target_net(self, steps_done):
        should_update_target_net = steps_done % self.target_update_period == 0

        return should_update_target_net

    # def _tensorboard_write_scalars(self):
    #     """  """
    #     if self.writer:
    #         self.writer.add_scalar('Agent/Loss', episode_loss / episode_duration, i_episode)
    #         self.writer.add_scalar('Agent/Reward Train', episode_reward, i_episode)
    #         self.writer.add_scalar('Agent/Epsilon', self.agent.epsilon, i_episode)
    #         self.writer.add_scalar('Agent/Steps', steps_done, i_episode)
    #         self.writer.flush()

    def _tensorboard_write_hparams(self, hyperparams, metrics):
        """
         :param hyperparams: Hyper-parameters of the experiment
         :param metrics: Metrics based on which we evaluate the experiment
         """

        # first dict with hparams, second dict with evaluation metrics

        # {'lr': lr, 'gamma': gamma, 'HL Dims': str(layers_dim), 'Double': double, 'Dueling': dueling,
        #  'PER': per, 'Mem Size': mem_size, 'Target_upd_interval': TARGET_NET_UPDATE_PERIOD,
        #  'Batch Size': BATCH_SIZE, 'EPS_DECAY': eps_decay}

        # {'episodes_needed': len(train_rewards)}

        self.writer.add_hparams(hyperparams, metrics)
        self.writer.flush()

    def _forward_step(self, episode_duration, state, episode_reward):
        """  """
        episode_duration += 1
        if self.render:
            self.env.render()
        action = self.agent.choose_action(state)
        next_state, reward, done, _ = self.env.step(action)
        next_state = np.float32(next_state)
        self.memory.store(state, action, next_state, reward, done)  # Store the transition in memory
        state = next_state
        episode_reward += reward

    def _backward_step(self, steps_done, episode_loss):
        """ A batch is sampled from the memory and one step of optimization is performed. """

        steps_done += 1
        try:
            transitions, indices, is_weights = self.memory.sample()
        except ValueError:
            return
        loss, errors = self.agent.update(transitions, is_weights)  # Perform one step of optimization on the policy net
        episode_loss += loss
        self.agent.adjust_exploration(steps_done)  # rate is updated at every step - taken from the tutorial
        self.memory.batch_update(indices, errors)
        if self._should_update_target_net(steps_done):
            # Update the target network
            self.agent.update_target_net()
            # if cartpole_constants.USE_TENSORBOARD and LOG_WEIGHTS:
            #     log_parameters_histograms(tensorboard_writer, agent.target_net, i_episode, 'TargetNet')

    def train_step(self):
        """  """

        train_rewards, eval_rewards = {}, {}
        epsilon_at_end_of_each_episode = []
        steps_done = 0
        for i_episode in range(self.max_episodes):

            state = self.env.reset()
            state = np.float32(state)

            episode_duration = 0
            episode_reward = 0
            episode_loss = 0

            done = False
            eval_episode = self._should_eval(i_episode)

            self.agent.train_mode()
            while not done:
                self._forward_step(episode_duration, state, episode_reward)

                self._backward_step(steps_done, episode_loss)

            train_rewards[i_episode] = episode_reward
            # self._tensorboard_write_scalars()
        #     if LOG_WEIGHTS:
        #         log_parameters_histograms(tensorboard_writer, self.agent.policy_net, i_episode, 'PolicyNet')
        #
        # figure = plot_rewards_completed(train_rewards, eval_rewards)
        # figure.show()

        # if cartpole_constants.USE_TENSORBOARD:
        #     tensorboard_writer.add_figure('Plot', figure)
        #
        #     state = np.float32(env.reset())
        #     tensorboard_writer.add_graph(agent.policy_net, torch.tensor(state, device=agent.device))
        #
        #     self._tensorboard_write_hparams()
        #     tensorboard_writer.close()

    def eval_step(self):
        self.agent.eval_mode()
        pass

    def run(self):

        self.train_step()



