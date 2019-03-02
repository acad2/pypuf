from numpy.random.mtrand import RandomState
from pypuf.experiments.experiment.base import Experiment
from pypuf.experiments.result import ExperimentResult
from pypuf.learner.neural_networks.mlp import MultiLayerPerceptron
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf import tools


NUM_VALIDATION_SET = 10000
FLOAT_PRECISION = '.4f'


class ExperimentMLP(Experiment):
    """
    This Experiment uses the MLP learner on an LTFArray PUF simulation.
    """

    def __init__(self, log_name, N, n, k, transformation, combiner, preprocess, batch_size=1000, iteration_limit=1000,
                 print_keras=False, seed_simulation=None, seed_challenges=None, seed_model=None, seed_accuracy=None):
        progress_log_name = '{0}_MLP_0x{1}_0x{2}_0_{3}_{4}_{5}_{6}_{7}'.format(
            log_name,
            seed_model,
            seed_simulation,
            n,
            k,
            N,
            transformation.__name__,
            combiner.__name__,
        ) if log_name else None
        super().__init__(progress_log_name)
        self.N = N
        self.n = n
        self.k = k
        self.transformation = transformation
        self.combiner = combiner
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.iteration_limit = iteration_limit
        self.print_keras = print_keras
        self.seed_simulation = seed_simulation
        self.seed_challenges = seed_challenges
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
            combiner=self.combiner
        )
        prng_challenges = RandomState(seed=self.seed_challenges)
        self.training_set = tools.TrainingSet(self.simulation, self.N, prng_challenges)
        self.validation_set = tools.TrainingSet(self.simulation, NUM_VALIDATION_SET, prng_challenges)
        self.learner = MultiLayerPerceptron(
            log_name=self.progress_log_name,
            n=self.n,
            k=self.k,
            training_set=self.training_set,
            validation_set=self.validation_set,
            transformation=self.transformation if self.preprocess else None,
            print_keras=self.print_keras,
            iteration_limit=self.iteration_limit,
            batch_size=self.batch_size,
            seed_model=self.seed_model
        )
        self.learner.prepare()

    def run(self):
        """
        Train the learner.
        """
        self.model = self.learner.learn()

    def analyze(self):
        """
        Calculate statistics of the trained model and write them into a result log file.
        """
        assert self.model is not None
        accuracy = 1.0 - tools.approx_dist(
                self.simulation,
                self.model,
                min(10000, 2 ** self.n),
                random_instance=RandomState(seed=self.seed_accuracy)
            )
        self.result_logger.info('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}'.format(
            hex(self.seed_simulation),
            hex(self.seed_model),
            self.n,
            self.k,
            self.N,
            format(max(accuracy, 1 - accuracy), FLOAT_PRECISION),
            format(self.learner.history.history['val_pypuf_accuracy'][-1], FLOAT_PRECISION),
            format(self.learner.history.history['pypuf_accuracy'][-1], FLOAT_PRECISION),
            len(self.learner.history.history['loss']),
            self.measured_time,
            self.transformation.__name__,
            self.combiner.__name__
        ))
        result = ExperimentResult()
        result.N = self.N
        result.n = self.n
        result.k = self.k
        result.transformation = self.transformation.__name__
        result.accuracy = max(accuracy, 1 - accuracy)
        result.time = self.measured_time
        result.batch_size = self.batch_size
        result.experiment = self.__class__.__name__
        return result
