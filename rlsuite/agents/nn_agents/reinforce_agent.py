import torch
from rlsuite.agents.agent import Agent
from torch.distributions import Categorical
import torch.nn.functional as F


class Reinforce(Agent):
    """ Implementation of policy gradients agent. Can be used only in episodic environments. """

    def __init__(self, num_of_actions, network, optimizer, gamma=0.999, use_gpu=False):
        """  """
        super().__init__(num_of_actions, gamma)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.policy_net = network.to(self.device)
        self.optimizer = optimizer

    def choose_action(self, state, train=True):
        """ Our neural network based only on current state outputs the probabilities of taken every possible action.
        We form a distribution and we sample from it. By this way we achieve exploration as well. """

        state = torch.tensor(state, device=self.device)
        # each element of probs is the relative probability of sampling the class at that index
        probs = F.softmax(self.policy_net(state), dim=-1)

        m = Categorical(probs)  # Creates a categorical distribution parameterized by probs

        # MY MODIFICATION act greedily when evaluating,
        # it cancels the stochastic nature of the algorithm
        if train:
            action = m.sample()  # we choose an action based on their probability
        else:
            action = probs.argmax()

        log_prob = m.log_prob(action)
        return action.item(), log_prob, probs.max()  # action is a tensor so we return just the number

    def update(self, log_probs, discounted_rewards):
        """ Calculates the loss function for every step of the episode, by calculating the log_prob and the
         corresponding reward. We then take the sum of those values and we apply backpropagation."""

        policy_loss = []
        for log_prob, discounted_reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * discounted_reward)  # we add minus to turn score function into loss function

        self.optimizer.zero_grad()
        # transforms a list of single element tensors to a single tensor including all of them
        policy_loss = torch.stack(policy_loss)
        policy_loss = policy_loss.sum()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss

    def calculate_rewards(self, rewards):
        """ We start from the end of each episode and we calculate the discounted rewards. Moreover we standardize
         the rewards. By doing this we’re always encouraging and discouraging roughly half of the performed actions"""

        discounted_rewards = []
        discounted_reward = 0
        for r in rewards[::-1]:
            discounted_reward = r + self.gamma * discounted_reward
            # inserted at first place all other elements are shifted to the right
            discounted_rewards.insert(0, discounted_reward)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        # 1e-9 used to avoid division by zero
        # the standardization doesn't seems to have a great effect, but it is used by all tutorials

        return discounted_rewards

    def train_mode(self):
        self.policy_net.train()

    def eval_mode(self):
        self.policy_net.eval()

    def save_checkpoint(self, filename):
        raise NotImplementedError

