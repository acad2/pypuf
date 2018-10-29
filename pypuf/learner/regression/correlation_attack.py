"""
This module provides an attack on XOR Arbiter PUFs that is based off known correlation in sub-challenge generation of
the input transformation.
"""
from copy import deepcopy
from itertools import permutations
from scipy.io import loadmat
from numpy.random import RandomState
from numpy import empty, roll
from pypuf.learner.base import Learner
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.tools import set_dist


class CorrelationAttack(Learner):

    OPTIMIZATION_ACCURACY_LOWER_BOUND = .65
    OPTIMIZATION_ACCURACY_UPPER_BOUND = .95
    OPTIMIZATION_ACCURACY_GOAL = .98

    def __init__(self, n, k,
                 training_set, validation_set,
                 weights_mu=0, weights_sigma=1, weights_prng=RandomState(),
                 lr_iteration_limit=1000,
                 logger=None, bias=False):
        self.n = n
        self.k = k

        self.training_set = training_set
        self.validation_set = validation_set

        self.weights_mu = weights_mu
        self.weights_sigma = weights_sigma
        self.weights_prng = weights_prng

        self.logger = logger
        self.bias = bias

        self.lr_learner = LogisticRegression(
            t_set=training_set,
            n=n,
            k=k,
            transformation=LTFArray.transform_lightweight_secure_original,
            combiner=LTFArray.combiner_xor,
            weights_mu=weights_mu,
            weights_sigma=weights_sigma,
            weights_prng=weights_prng,
            logger=logger,
            iteration_limit=lr_iteration_limit,
            bias=False
        )
        
        self.initial_accuracy = .5
        self.initial_iterations = 0
        self.initial_model = None
        self.best_iteration = 0
        self.rounds = ''
        self.permutations = None
        self.permuted_model = None
        self.best_permutation = None
        self.best_model = None
        self.best_accuracy = None

        assert n in (64, 128), 'Correlation attack for %i bit is currently not supported.' % n
        assert validation_set.N >= 1000, 'Validation set should contain at least 1000 challenges.'

        self.correlation_permutations = loadmat(
            'correlation_permutations_lightweight_secure_original_%i_10.mat' % n
        )['shiftOverviewData'][:, :, 0].astype('int64')

    def learn(self):
        # Find any model
        #self.lr_learner.convergence_decimals = 1.5
        self.initial_model = initial_model = self.lr_learner.learn()
        self.initial_accuracy = 1 - set_dist(initial_model, self.validation_set.block_subset(0, 2))
        self.initial_iterations = self.lr_learner.iteration_count
        initial_updater = self.lr_learner.updater

        self.best_model = initial_model
        self.best_accuracy = self.initial_accuracy

        if self.initial_accuracy < self.OPTIMIZATION_ACCURACY_LOWER_BOUND:
            self.logger.debug('initial learning below threshold, aborting')
            return initial_model

        if self.initial_accuracy > self.OPTIMIZATION_ACCURACY_GOAL:
            self.logger.debug('initial learning successful, aborting')
            return initial_model

        # Try all permutations with high initial accuracy and see if any of them lead to a good finial result
        adopted_weights = self.find_high_accuracy_weight_permutations(
            initial_model.weight_array,
            1.2 * self.best_accuracy - .2         # allow some accuracy loss by permuting
                                                  # the higher the initial accuracy, the higher the loss we allow
                                                  # result will never be below 0.925
        )
        self.logger.debug('Trying %i permuted weights.' % len(adopted_weights))
        for (iteration, weights) in enumerate(adopted_weights):
            original_permuted_weights = empty((self.k, self.n + 1))
            original_permuted_weights[::] = weights
            self.lr_learner.updater = deepcopy(initial_updater)
            self.lr_learner.updater.step_size *= 10
            model = self.lr_learner.learn(init_weight_array=weights, refresh_updater=False)
            accuracy = 1 - set_dist(model, self.validation_set.block_subset(1, 2))
            self.logger.debug('With a permutation, after restarting the learning we achieved accuracy %.2f!' % accuracy)
            if accuracy > 0.2 + 0.8 * self.initial_accuracy\
                    and accuracy > self.best_accuracy:  # demand some "substantial" improvement of accuracy
                                                           # what substantial means becomes weaker as we approach
                                                           # perfect accuracy
                self.permuted_model = LTFArray(weight_array=original_permuted_weights, transform=LTFArray.transform_id, combiner=LTFArray.combiner_xor)
                self.best_model = model
                self.best_accuracy = accuracy
                self.best_iteration = iteration
            else:
                self.logger.debug('Learning after permuting lead to accuracy %.2f, no improvement :-(' % accuracy)

            if accuracy > self.OPTIMIZATION_ACCURACY_GOAL:
                self.logger.debug('Found a model with accuracy better than %.2f, refine accuracy then abort.' %
                                  self.OPTIMIZATION_ACCURACY_GOAL)
                self.best_model = model
                return model

        self.logger.debug('After trying all permutations, we found a model with accuracy %.2f.' % self.best_accuracy)
        return self.best_model

    def find_high_accuracy_weight_permutations(self, weights, threshold):
        high_accuracy_permutations = []
        for permutation in list(permutations(range(self.k)))[1:]:
            adopted_weights = self.adopt_weights(weights, permutation)
            adopted_instance = LTFArray(
                weight_array=adopted_weights[:, :-1],
                transform=LTFArray.transform_lightweight_secure_original,
                combiner=LTFArray.combiner_xor,
                bias=adopted_weights[:,-1:]
            )
            accuracy = 1 - set_dist(adopted_instance, self.validation_set)
            self.logger.debug('For permutation %s, we have accuracy %.4f' % (permutation, accuracy))
            if accuracy >= threshold:
                high_accuracy_permutations.append(
                    {
                        'accuracy': accuracy,
                        'permutation': permutation,
                        'weights': adopted_instance.weight_array,
                    }
                )

        # return the 4k permutations with the highest initial accuracy
        high_accuracy_permutations.sort(key=lambda x: -x['accuracy'])
        self.permutations = [ item['permutation'] for item in high_accuracy_permutations ][:2*self.k]
        return [ item['weights'] for item in high_accuracy_permutations ][:2*self.k]

    def adopt_weights(self, weights, permutation):
        adopted_weights = empty(weights.shape)
        for l in range(self.k):
            adopted_weights[permutation[l], :] = \
                roll(weights[l, :], self.correlation_permutations[l, permutation[l]])
        return adopted_weights
