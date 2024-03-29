import random
import numpy as np
from rlsuite.utils.sumtree import SumTree
from collections import deque


class Memory:

    def __init__(self, capacity, batch_size, init_capacity=0):
        self.batch_size = batch_size
        self.init_capacity = init_capacity
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def num_of_samples(self):
        """ Returns number of samples stored in memory. """
        return len(self.memory)

    def is_initialized(self):
        """ Checks whether the memory contains the minimum required number of samples. """
        return self.num_of_samples() >= self.init_capacity

    def store(self, *args):
        """Saves a transition."""
        self.memory.append(args)

    def sample(self):
        """ Returns a list of samples. """
        return random.sample(self.memory, self.batch_size), None, None  # returns a list of samples(tuples)

    def batch_update(self, tree_idx, abs_errors):
        """ As no priorities are used in simple memory, update has no effect """
        pass

    def flush(self):  # if we use flush in every episode it doesn't get trained at all
        """ In case we want to empty the memory.  """
        self.memory.clear()  # Used in experiments when a new instance of the problem comes up.


class MemoryPER:  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparam that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparam that sets a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    PER_absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity, batch_size, init_capacity=0):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.batch_size = batch_size
        self.init_capacity = init_capacity
        self.capacity = capacity
        self.tree = SumTree(capacity)

    def num_of_samples(self):
        """ Returns number of samples stored in memory. """
        return self.tree.n_entries

    def is_initialized(self):
        """ Checks whether the memory contains the minimum required number of samples. """
        return self.num_of_samples() >= self.init_capacity

    def store(self, *experience):
        """
        Store a new experience in our tree
        Each new experience have a score of max_prority (By training it will be improved)
        """
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])  # max priority of the leaves

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.PER_absolute_error_upper

        # print(max_priority)
        self.tree.add(max_priority, experience)  # set the max p for new p

    def sample(self):
        """
        - First, to sample a minibatch of n size, the range [0, priority_total] is / into n ranges.
        - Then a value is uniformly sampled from each range
        - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
        - Then, we calculate IS weights for each minibatch element
        """
        if self.tree.n_entries < self.batch_size:
            raise ValueError()

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / self.batch_size  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        leaves = self.tree.tree[-self.tree.capacity:]
        p_min = np.min(leaves[np.nonzero(leaves)]) / self.tree.total_priority
        max_weight = np.power(p_min * self.tree.n_entries, -self.PER_b)

        values = [self.tree.get_leaf(np.random.uniform(priority_segment * i, priority_segment * (i + 1))) for i in
                  range(self.batch_size)]
        indices, priorities, transitions = zip(*values)
        sampling_probabilities = priorities / self.tree.total_priority
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.PER_b)
        is_weights /= max_weight

        return transitions, indices, is_weights

    def batch_update(self, tree_idx, errors):
        """
        Update the priorities on the tree. Data on memory don't change.
        """
        errors += self.PER_e  # avoid 0
        errors = np.minimum(errors, self.PER_absolute_error_upper)  # not present in other implementation
        ps = np.power(errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def flush(self):
        """ Empties the memory """
        self.tree.clear()
