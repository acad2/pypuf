from numpy.random import RandomState
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.neural_networks.mlp import MultiLayerPerceptron
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf import tools


NUM_VALIDATION_SET = 10000


class ExperimentMLP(Experiment):
    """
    This Experiment uses the MLP learner on an LTFArray PUF simulation.
    """

    def __init__(self, log_name, N, n, k, transformation, combiner, preprocess, batch_size=1000, iteration_limit=1000,
                 seed_simulation=None, seed_challenges=None, seed_model=None, seed_accuracy=None):
        super().__init__(
            log_name='%s_MLP_0x%x_0x%x_0_%i_%i_%i_%s_%s' % (
                log_name,
                seed_model,
                seed_simulation,
                n,
                k,
                N,
                transformation.__name__,
                combiner.__name__,
            ),
        )
        self.N = N
        self.n = n
        self.k = k
        self.transformation = transformation
        self.combiner = combiner
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.iteration_limit = iteration_limit
        self.seed_simulation = seed_simulation
        self.prng_challenges = RandomState(seed=seed_challenges)
        self.seed_model = seed_model
        self.seed_accuracy = seed_accuracy
        self.training_set = None
        self.validation_set = None
        self.simulation = None
        self.learner = None
        self.model = None

    def prepare(self):
        """
        Prepare learning: initialize learner, prepare training set, etc.
        """
        self.simulation = LTFArray(
            weight_array=LTFArray.normal_weights(
                self.n,
                self.k,
                random_instance=RandomState(seed=self.seed_simulation)
            ),
            transform=self.transformation,
            combiner=self.combiner,
        )

        self.training_set = tools.TrainingSet(self.simulation, self.N, self.prng_challenges)
        self.validation_set = tools.TrainingSet(self.simulation, NUM_VALIDATION_SET, self.prng_challenges)

        self.learner = MultiLayerPerceptron(
            log_name=self.log_name,
            n=self.n,
            k=self.k,
            training_set=self.training_set,
            validation_set=self.validation_set,
            transformation=self.transformation if self.preprocess else None,
            iteration_limit=self.iteration_limit,
            batch_size=self.batch_size,
            seed_model=self.seed_model
        )
        self.learner.prepare()

    def run(self):
        """
        Runs the learner
        """
        self.model = self.learner.learn()

    def analyze(self):
        """
        Analyzes the learned result.
        """
        assert self.model is not None
        distance = 1.0 - tools.approx_dist(
                self.simulation,
                self.model,
                min(10000, 2 ** self.n),
                random_instance=RandomState(seed=self.seed_accuracy)
            )
        self.result_logger.info(
            '0x%x\t0x%x\t%i\t%i\t%i\t%f\t%s\t%s\t%s',
            self.seed_simulation,
            self.seed_model,
            self.n,
            self.k,
            self.N,
            max(distance, 1 - distance),
            self.measured_time,
            self.transformation.__name__,
            self.combiner.__name__,
        )
