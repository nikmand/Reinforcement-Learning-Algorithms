import numpy as np
import random
from statistics import mean, stdev
from rlsuite.agents.classic_agents.classic_agent import ClassicAgent


class MCAgent(ClassicAgent):
    """ Implementation of Monte carlo without exploration start. The simulation starts from the same point every time
     and we explore the action space by using e-greedy. """

    def __init__(self, num_of_actions, dimensions, gamma=0.999, epsilon=1):
        """
        """
        super().__init__(num_of_actions, dimensions, gamma, epsilon)
        # we keep the number of times that we have visited each Q(s, a)
        self.num_of_visits = np.zeros(dimensions + [num_of_actions])

    def update(self, state_action_ls, rewards):
        """ This method runs at the end of every episode. We take into consideration only the first time that we
        encounter a state. For each (state, action) we have the corresponding value. We update the mean Q(s, a) by
        taking into consideration the value of the current episode. """

        visited_states = set()
        for (state, action), value in zip(state_action_ls, rewards):
            if (tuple(state), action) not in visited_states:  # first-visit MC
                # μ_k += 1/k(x_k - μ_(k-1)), we move our mean to the direction of the error
                index = state + (action,)
                self.num_of_visits[index] += 1
                coef = max(0.1, 1.0 / self.num_of_visits[index])
                self.q_table[index] += coef * (value - self.q_table[index])
                # self.q_table[state + (action,)] = (self.q_table[state + (action,)] * (self.num_of_visits[
                #     state + (action,)] - 1) + value) / self.num_of_visits[state + (action,)]
                visited_states.add((tuple(state), action))

    def calculate_rewards(self, rewards):
        """  """
        discounted_rewards = []
        value = 0

        for reward in rewards[::-1]:
            value = reward + self.gamma * value
            discounted_rewards.insert(0, value)

        discounted_mean = mean(discounted_rewards)
        discounted_std = stdev(discounted_rewards)
        discounted_rewards = [(discounted_reward - discounted_mean) / (discounted_std + 1e-9) for discounted_reward in discounted_rewards]
        # is it needed to standarize the values? the effect doesn't seem so important

        return discounted_rewards
