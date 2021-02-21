import torch
from rlsuite.agents.agent import Agent
from torch.distributions import Categorical
import torch.nn.functional as F
from abc import abstractmethod


class ActorCritic(Agent):

    def __init__(self, num_of_actions, network, criterion, optimizer, gamma=0.999, gpu=False):
        super().__init__(num_of_actions, gamma)
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
        self.actor_critic_net = network.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer

    def choose_action(self, state, train=True):
        state = torch.tensor(state, device=self.device)
        logits, state_value = self.actor_critic_net(state)
        probs = F.softmax(logits, dim=-1)

        m = Categorical(probs)
        if train:
            action = m.sample()  # we choose an action based on their probability
        else:
            action = probs.argmax()

        log_prob = m.log_prob(action)
        return action.item(), log_prob, state_value, probs.max()  # action is a tensor so we return just the number

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError

    def train_mode(self):
        self.actor_critic_net.train()

    def eval_mode(self):
        self.actor_critic_net.eval()

    # def save_checkpoint(self, filename):
    #     raise NotImplementedError


class ActorCriticMC(ActorCritic):

    def __init__(self, num_of_actions, network, criterion, optimizer, gamma=0.999, gpu=False):
        super().__init__(num_of_actions, network, criterion, optimizer, gamma, gpu)

    def update(self, log_probs, state_values, discounted_rewards):

        policy_loss, state_value_loss = [], []
        for log_prob, state_value, discounted_reward in zip(log_probs, state_values, discounted_rewards):
            advantage = discounted_reward - state_value.item()
            # print(state_value.type())
            # print(discounted_reward.type())
            # print(state_value.shape)
            # print(discounted_reward.shape)

            # calculate actor (policy) loss
            # gradient should not be computed for advantage
            policy_loss.append(-log_prob * advantage)

            # calculate critic (value) loss
            state_value_loss.append(self.criterion(state_value, torch.tensor([discounted_reward], device=self.device)))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(state_value_loss).sum()
        loss.backward()
        self.optimizer.step()

        return loss

    def calculate_rewards(self, rewards):
        """  """

        discounted_rewards = []
        discounted_reward = 0
        for r in rewards[::-1]:
            discounted_reward = r + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        # the standardization doesn't seems to have a great effect, but it is used by all tutorials

        return discounted_rewards


class ActorCriticTD(ActorCritic):

    def __init__(self, num_of_actions, network, criterion, optimizer, gamma=0.999, gpu=False):
        super().__init__(num_of_actions, network, criterion, optimizer, gamma, gpu)

    def update(self, log_prob, predicted_state_value, next_state, reward, done):
        """ Updates are based on single step that means high bias """

        next_state = torch.tensor(next_state, device=self.device)
        _, next_state_value = self.actor_critic_net(next_state)
        target_value = (reward + (1 - int(done)) * self.gamma * next_state_value).detach()
        td_delta = target_value - predicted_state_value
        policy_loss = -log_prob * td_delta
        state_value_loss = self.criterion(target_value, predicted_state_value)
        loss = policy_loss + state_value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
