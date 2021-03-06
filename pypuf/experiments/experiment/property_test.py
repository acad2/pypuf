"""This module can be used to characterize the properties of a puf class."""
from collections import OrderedDict
from typing import NamedTuple
from uuid import UUID

from numpy import array
from numpy.random import RandomState
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray, SimulationMajorityLTFArray
from pypuf.experiments.experiment.base import Experiment
from pypuf.property_test.base import PropertyTest
from pypuf.tools import sample_inputs


class Parameters(NamedTuple):
    """
    Parameter for PropertyTest experiment.
    """

    # PropertyTest.uniqueness_statistic or reliability_statistic
    # Function which is used to calculate a statistic.
    test_function: str

    # Number of challenges used.
    challenge_count: int

    # Number of calculations see test_function for more details.
    measurements: int

    # The seed which is used to initialize the pseudo-random number generator
    # which is used to generate challenges.
    challenge_seed: int

    # A Function: *kwargs -> list of pypuf.simulation.base.Simulation
    # This function is used to generate a list of simulation instances which are inspected.
    ins_gen_function: str

    # A collections.OrderedDict with keyword arguments
    # This keyword arguments are passed to ins_gen_function to generate
    # pypuf.simulation.base.Simulation instances and saved into the result log.
    param_ins_gen: OrderedDict


class Result(NamedTuple):
    """
    Result of PropertyTest experiment.
    """

    experiment_id: UUID
    mean: float
    median: float
    minimum: float
    maximum: float
    sample_variance: float
    measured_time: float
    samples_string: str


class ExperimentPropertyTest(Experiment):
    """
    This class can be used to test several puf simulations instances with the pypuf.property_test.base.PropertyTest
    class.
    """
    def __init__(self, progress_log_name, parameters):
        super().__init__(progress_log_name=progress_log_name, parameters=parameters)
        self.result = None

    def run(self):
        """Runs a property test."""
        ins_gen_function = getattr(self, self.parameters.ins_gen_function)
        instances = ins_gen_function(**self.parameters.param_ins_gen)
        n = self.parameters.param_ins_gen['n']
        challenge_prng = RandomState(self.parameters.challenge_seed)
        challenges = array(list(sample_inputs(n, self.parameters.challenge_count, random_instance=challenge_prng)))
        property_test = PropertyTest(instances, logger=self.progress_logger)
        test_function = getattr(PropertyTest, self.parameters.test_function)
        self.result = test_function(property_test, challenges, measurements=self.parameters.measurements)

    def analyze(self):
        """Summarize the results of the search process."""
        assert self.result is not None

        return Result(
            experiment_id=self.id,
            mean=self.result.get('mean', float("inf")),
            median=self.result.get('median', float("inf")),
            minimum=self.result.get('min', float("inf")),
            maximum=self.result.get('max', float("inf")),
            sample_variance=self.result.get('sv', float("inf")),
            measured_time=self.measured_time,
            samples_string=','.join(list(map(str, self.result.get('samples', []))))
        )

    @classmethod
    def create_ltf_arrays(cls, n=8, k=1, instance_count=10, transformation=LTFArray.transform_id,
                          combiner=LTFArray.combiner_xor, bias=None, mu=0, sigma=1, weight_random_seed=0x123):
        """
        This function can be used to create a list of LTFArrays.
        :param n: int
                  Number of stages of the PUF
        :param k: int
                  Number different LTFArrays
        :param instance_count: int
                               Number of simulations to be instantiated.
        :param transformation: A function: array of int with shape(N,k,n), int number of PUFs k -> shape(N,k,n)
                               The function transforms input challenges in order to increase resistance against attacks.
        :param combiner: A function: array of int with shape(N,k,n) -> array of in with shape(N)
                         The functions combines the outputs of k PUFs to one bit results,
                         in oder to increase resistance against attacks.
        :param bias: None, float or a two dimensional array of float with shape (k, 1)
                     This bias value or array of bias values will be appended to the weight_array.
                     Use a single value if you want the same bias for all weight_vectors.
        :param mu: float
                   Mean (“centre”) of the stage weight distribution of the PUF instance simulation.
        :param sigma: float
                      Standard deviation of the stage weight distribution of the PUF instance simulation.
        :param weight_random_seed: int
                                   The seed which is used to initialize the pseudo-random number generator
                                   which is used to generate the stage weights for the arbiter PUF simulation.
        :return: list of pypuf.simulation.arbiter_based.ltfarray.LTFArray
        """
        instances = []
        for seed_offset in range(instance_count):
            weight_array = LTFArray.normal_weights(n, k, mu, sigma,
                                                   random_instance=RandomState(weight_random_seed + seed_offset))
            instances.append(
                LTFArray(
                    weight_array=weight_array,
                    transform=transformation,
                    combiner=combiner,
                    bias=bias,
                )
            )
        return instances

    @classmethod
    def create_noisy_ltf_arrays(cls, n=8, k=1, instance_count=10, transformation=LTFArray.transform_id,
                                combiner=LTFArray.combiner_xor, bias=None, mu=0, sigma=1, weight_random_seed=0x123,
                                sigma_noise=0.5, noise_random_seed=0x321):
        """
        This function can be used to create a list of NoisyLTFArray.
        :param n: int
                  Number of stages of the PUF
        :param k: int
                  Number different LTFArrays
        :param instance_count: int
                               Number of simulations to be instantiated.
        :param transformation: A function: array of int with shape(N,k,n), int number of PUFs k -> shape(N,k,n)
                               The function transforms input challenges in order to increase resistance against attacks.
        :param combiner: A function: array of int with shape(N,k,n) -> array of in with shape(N)
                         The functions combines the outputs of k PUFs to one bit results,
                         in oder to increase resistance against attacks.
        :param bias: None, float or a two dimensional array of float with shape (k, 1)
                     This bias value or array of bias values will be appended to the weight_array.
                     Use a single value if you want the same bias for all weight_vectors.
        :param mu: float
                   Mean (“centre”) of the stage weight distribution of the PUF instance simulation.
        :param sigma: float
                      Standard deviation of the stage weight distribution of the PUF instance simulation.
        :param weight_random_seed: int
                                   The seed which is used to initialize the pseudo-random number generator
                                   which is used to generate the stage weights for the arbiter PUF simulation.
        :param sigma_noise: float
                            Standard deviation of the noise distribution.
        :param noise_random_seed: int
                                  The seed which is used to initialize the pseudo-random number generator
                                  which is used to generate the noise for the arbiter PUF simulation.
        :return: list of pypuf.simulation.arbiter_based.ltfarray.NoisyLTFArray
        """
        instances = []
        for seed_offset in range(instance_count):
            weight_array = LTFArray.normal_weights(n, k, mu, sigma,
                                                   random_instance=RandomState(weight_random_seed + seed_offset))
            instances.append(
                NoisyLTFArray(
                    weight_array=weight_array,
                    transform=transformation,
                    combiner=combiner,
                    sigma_noise=sigma_noise,
                    random_instance=RandomState(noise_random_seed + seed_offset),
                    bias=bias,
                )
            )
        return instances

    @classmethod
    def create_mv_ltf_arrays(cls, n=8, k=1, instance_count=10, transformation=LTFArray.transform_id,
                             combiner=LTFArray.combiner_xor, bias=None, mu=0, sigma=1, weight_random_seed=0x123,
                             sigma_noise=0.5, noise_random_seed=0x321, vote_count=3):
        """
        This function can be used to create a list of SimulationMajorityLTFArray.
        :param n: int
                  Number of stages of the PUF
        :param k: int
                  Number different LTFArrays
        :param instance_count: int
                               Number of simulations to be instantiated.
        :param transformation: A function: array of int with shape(N,k,n), int number of PUFs k -> shape(N,k,n)
                               The function transforms input challenges in order to increase resistance against attacks.
        :param combiner: A function: array of int with shape(N,k,n) -> array of in with shape(N)
                         The functions combines the outputs of k PUFs to one bit results,
                         in oder to increase resistance against attacks.
        :param bias: None, float or a two dimensional array of float with shape (k, 1)
                     This bias value or array of bias values will be appended to the weight_array.
                     Use a single value if you want the same bias for all weight_vectors.
        :param mu: float
                   Mean (“centre”) of the stage weight distribution of the PUF instance simulation.
        :param sigma: float
                      Standard deviation of the stage weight distribution of the PUF instance simulation.
        :param weight_random_seed: int
                                   The seed which is used to initialize the pseudo-random number generator
                                   which is used to generate the stage weights for the arbiter PUF simulation.
        :param sigma_noise: float
                            Standard deviation of the noise distribution.
        :param noise_random_seed: int
                                  The seed which is used to initialize the pseudo-random number generator
                                  which is used to generate the noise for the arbiter PUF simulation.
        :param vote_count: int
                           Number of evaluations which are used to choose a response with majority vote.
        :return: list of pypuf.simulation.arbiter_based.ltfarray.SimulationMajorityLTFArray
        """
        instances = []
        for seed_offset in range(instance_count):
            weight_array = LTFArray.normal_weights(n, k, mu, sigma,
                                                   random_instance=RandomState(weight_random_seed + seed_offset))
            instances.append(
                SimulationMajorityLTFArray(
                    weight_array=weight_array,
                    transform=transformation,
                    combiner=combiner,
                    sigma_noise=sigma_noise,
                    random_instance_noise=RandomState(noise_random_seed + seed_offset),
                    bias=bias,
                    vote_count=vote_count,
                )
            )
        return instances
