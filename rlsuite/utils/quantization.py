import numpy as np
import scipy.stats


class Quantization:
    """
        Help class that handles the quantization of the magnitude involved in the problem
    """

    def __init__(self, dimensions_description):
        """
             Initiates a list which keeps the bins intervals for each variable of the problem

        :param dimensions_description: An iterable which contains tuple entries (start, stop, num_bins) for each variable
        """

        self.dimensions_bins = [self.initiate_dimension(*dimension_description)
                                for dimension_description in dimensions_description]
        self.dimensions = [len(var_bin) - 1 for var_bin in self.dimensions_bins]  # num of bins for each variable

    @staticmethod
    def initiate_dimension(low_barrier, high_barrier, num_bins):
        """
            Splits the interval of possible values that a magnitude can take into bins

        :param low_barrier:
        :param high_barrier:
        :param num_bins:
        :return: an array that contains the numbers that constitute bins bounds
        """

        # TODO investigate if it is useful to split the interval based on a distribution
        # so that higher bin frequency is used for points which are more probable to come up

        intervals = np.linspace(low_barrier, high_barrier, num_bins + 1, endpoint=True)
        # intervals = scipy.stats.norm.ppf(intervals)
        return intervals

    def digitize(self, observations):
        """

        :param observations: list that contains continues values foreach variable of the problem
        :return: their quantized equivalents
        """

        assert len(observations) == len(self.dimensions_bins)  # validate that there are observations for every variable
        digitized = tuple(
            np.digitize(observation, var_bin) - 1 for observation, var_bin in zip(observations, self.dimensions_bins))

        # TODO check for values that fall outside of bins borders
        # If values in observations are beyond the bounds of bins, 0 or len(bins) is returned as appropriate.
        return digitized


if __name__ == 'main':
    pass
