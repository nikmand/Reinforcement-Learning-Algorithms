import numpy as np
from abc import ABC, abstractmethod


class Trainer(ABC):
    """  """

    def __init__(self, agent, env, memory, config):

        self.env = env
        self.agent = agent
        self.memory = memory

        self.max_episodes = config["max_ep"]
        self.eval_interval = config["eval_interval"]

    def _should_eval(self, i_episode):
        should_eval = (i_episode + 1) % self.eval_interval == 0

        return should_eval

    def train_step(self):
        self.agent.train_mode()
        pass

    def eval_step(self):
        self.agent.eval_mode()
        pass

    def run(self):
        for i_episode in range(self.max_episodes):

            state = self.env.reset()
            state = np.float32(state)

            episode_duration = 0
            episode_reward = 0
            episode_loss = 0

            done = False
            eval_episode = self._should_eval(i_episode)

            if eval_episode:
                self.eval_step()
            else:
                self.train_step()



