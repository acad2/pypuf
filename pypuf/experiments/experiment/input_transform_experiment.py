"""This module provides an experiment class which learns an instance of LTFArray with reliability based CMAES learner.
It is based on the work from G. T. Becker in "The Gap Between Promise and Reality: On the Insecurity of
XOR Arbiter PUFs". The learning algorithm applies Covariance Matrix Adaptation Evolution Strategies from N. Hansen
in "The CMA Evolution Strategy: A Comparing Review".
"""
import numpy as np

from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf import tools


class InputTransformExperiment(Experiment):
    """This class implements an experiment for executing the reliability based CMAES learner for XOR LTF arrays.
    It provides all relevant parameters as well as an instance of an LTF array to learn.
    Furthermore, the learning results are being logged into csv files.
    """

    def __init__(self, log_name, k, n, transform, combiner, num, seed_instance, seed_challenges):
        """Initialize an Experiment using the Reliability based CMAES Learner for modeling LTF Arrays
        :param log_name:        Log name, Prefix of the name of the experiment log file
        :param seed_instance:   PRNG seed used to create an LTF array instance to learn
        :param k:               Width, the number of parallel LTFs in the LTF array
        :param n:               Length, the number stages within the LTF array
        :param transform:       Transformation function, the function that modifies the input within the LTF array
        :param combiner:        Combiner, the function that combines particular chains' outputs within the LTF array
        :param seed_challenges: PRNG seed used to sample challenges
        :param num:             Challenge number, the number of binary inputs (challenges) for the LTF array
        """
        super().__init__(
            progress_log_name='%s.0x%x_%i_%i_%i' % (
                log_name,
                seed_instance,
                k,
                n,
                num,
            ),
        )
        # Instance of LTF array to learn
        self.seed_instance = seed_instance
        self.prng_i = np.random.RandomState(seed=self.seed_instance)
        self.k = k
        self.n = n
        self.transform = transform
        self.combiner = combiner
        # Training set
        self.seed_challenges = seed_challenges
        self.prng_c = np.random.RandomState(seed=self.seed_instance)
        self.num = num
        self.training_set = None
        self.training_set_fast = None
        # Basic objects
        self.instance = None
        self.learner = None
        self.model = None

    def prepare(self):
        self.instance = LTFArray(
            weight_array=LTFArray.normal_weights(self.n, self.k, random_instance=self.prng_i),
            transform=self.transform,
            combiner=self.combiner,
        )
        self.training_set = tools.TrainingSet(self.instance, self.num, self.prng_c)
        self.training_set_fast = tools.ChallengeResponseSet(
            challenges=self.transform(self.training_set.challenges, self.k),
            responses=self.training_set.responses
        )

    def run(self):
        """Initialize the instance, the training set and the learner
        to then run the Reliability based CMAES with the given parameters
        """
        self.learner = LogisticRegression(
            t_set=self.training_set_fast,
            n=self.n,
            k=self.k,
            transformation=LTFArray.transform_none,
            combiner=self.combiner,
            weights_mu=0,
            weights_sigma=1,
            weights_prng=self.prng_i,
            logger=self.progress_logger,
            iteration_limit=10000
        )
        self.model = self.learner.learn()
        self.model.transform = self.transform

    def analyze(self):
        """Analyze the learned model"""
        assert self.model is not None
        distance = 1.0 - tools.approx_dist(self.instance, self.model, min(1000, 2 ** self.n), self.prng_c)
        self.result_logger.info(
            '0x%x\t0x%x\t%i\t%i\t%i\t%f\t%s\t%s\t%s\t%s\t%s',
            self.seed_instance,
            self.seed_challenges,
            self.n,
            self.k,
            self.num,
            max(distance, 1 - distance),
            ','.join(map(str, self.individual_accs())),
            self.measured_time,
            self.transform.__name__,
            self.combiner.__name__,
            ','.join(map(str, self.model.weight_array.flatten() / np.linalg.norm(self.model.weight_array.flatten()))),
        )

    def individual_accs(self):
        """Calculate the accuracies of individual chains of the learned model"""
        transform = self.model.transform
        combiner = self.model.combiner
        accuracies = np.zeros(self.k)
        polarities = np.zeros(self.k)
        for i in range(self.k):
            chain_original = LTFArray(self.instance.weight_array[i, np.newaxis, :], transform, combiner)
            for j in range(self.k):
                chain_model = LTFArray(self.model.weight_array[j, np.newaxis, :], transform, combiner)
                accuracy = tools.approx_dist(chain_original, chain_model, min(10000, 2 ** self.n), self.prng_c)
                pol = 1
                if accuracy < 0.5:
                    accuracy = 1.0 - accuracy
                    pol = -1
                if accuracy > accuracies[i]:
                    accuracies[i] = accuracy
                    polarities[i] = pol
        accuracies *= polarities
        for i in range(self.k):
            if accuracies[i] < 0:
                accuracies[i] += 1
        return accuracies
