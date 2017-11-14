"""
This module contains the Low Degree Algorithm.
"""
from itertools import combinations
import time
import numpy as np
from scipy.misc import comb as ncr
from pypuf.simulation.fourier_based.fourier_expansion import FourierExpansionSign, FourierCoefficient
from pypuf import tools


class LowDegreeAlgorithm:
    """
    Probabilistic algorithm to create a model of a Boolean function using a `training_set`. It approximates
    all Fourier coefficients of degree up to `degree`. If the training_set has size `get_training_set_size`
    and the function is epsilon/2-concentrated up to degree `degree` the algorithm returns a model that with
    probability 1-`delta` has accuracy 1-`epsilon`.
    """

    def __init__(self, training_set, degree, debug=False):
        """
        :param training_set: pypuf.tools.TrainingSet
                             The trainings set generated by tools.TrainingSet
        :param degree: int
                       The degree up to which the Fourier coefficients are approximated
        :param debug: boolean
                      If true, a progress message with ETA will be periodically printed to stdout
        """
        self.training_set = training_set
        self.n = len(training_set.challenges[0])
        self.monomial_count = 0
        for k in range(degree + 1):
            self.monomial_count += ncr(self.n, k)
        self.degree = degree
        self.fourier_coefficients = []
        self.debug = debug

    @staticmethod
    def get_training_set_size(n, degree, epsilon, delta):
        """
        This function calculates the training set size that is needed to satisfy the theoretical requirements of the
        Low Degree Algorithm such that the compliance of the epsilon and delta parameters is guaranteed.
        :param n: int
                  Input length
        :param degree: int
                       The degree up to which the Fourier coefficients are approximated
        :param epsilon: float
                        The maximum error rate of the model
        :param delta: float
                      The maximum failure rate of the algorithm, where epsilon is not satisfied
        :return:
        """
        monomial_count = 0
        for k in range(degree + 1):
            monomial_count += ncr(n, k)
        return int(4 * monomial_count * np.log(2 * monomial_count / delta) / epsilon)

    def learn(self):
        """
        Compute a model according to the given training set.
        Note that this function can take long to return.
        :return: The computed model.
        """
        processed = 0
        last = 0
        start_time = time.time()
        for i in range(self.degree + 1):
            for chi in self.low_degree_chi(i):
                self.fourier_coefficients.append(self.approx_fourier_coefficient(chi))
                if not self.debug:
                    continue
                processed += 1
                current = int(processed / self.monomial_count * 100)
                if current > last:
                    current_time = time.time()
                    time_diff = current_time - start_time
                    duration_left = np.round((100 - current) * time_diff / current)
                    last = current
                    print('\r%s percent complete. Estimated time left: %s s  ' % (current, duration_left), end='')
        if self.debug:
            print()

        return FourierExpansionSign(self.fourier_coefficients)

    def approx_fourier_coefficient(self, subset):
        """
        Approximate the Fourier coefficient of the function on `subset`
        :param subset: list of int
                       A {0,1}-array indicating the coefficient's index set

        :return float
                The approximated value of the coefficient
        """
        return FourierCoefficient(subset, tools.approx_fourier_coefficient(subset, self.training_set))

    def low_degree_chi(self, degree):
        """
        Returns an iterator for the sets s (represented as {0,1}-arrays that represent monomials with degree exactly
        `degree`.
        :param degree: int
                       The desired degree of the subsets
        :return iterator of arrays of length n
        """
        for indices in combinations(range(self.n), degree):
            yield np.array([1 if i in indices else 0 for i in range(self.n)], dtype=tools.RESULT_TYPE)
