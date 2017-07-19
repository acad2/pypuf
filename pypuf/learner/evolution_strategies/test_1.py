import numpy as np
import scipy.stats as stats
import itertools
from pypuf import tools
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.learner.evolution_strategies.becker import Reliability_based_CMA_ES as CMA

class Reliability_based_CMA_ES():

    # recommended properties of parameters:
    #   pop_size=30
    #   parent_size=10 (>= 0.3*pop_size)
    #   priorities: linear low decreasing
    # repeat = 5
    def __init__(self, instance, pop_size, parent_size, priorities,
                 challenge_num, repeat, unreliability):
        self.transform = LTFArray.transform_id
        self.combiner = LTFArray.combiner_xor
        self.different_LTFs = np.zeros((instance.k, instance.n)) # learned LTFs
        self.unreliability = unreliability  # proportion of unreliable challenges
        self.epsilon = 0            # learned threshold for reliability
        self.num_of_LTFs = 0        # number of different learned LTFs
        self.new_LTF = np.empty(instance.n) # current learned LTF
        self.terminate = False      # termination variable for learning 1 LTF
        self.instance = instance    # simulation instance to be modelled
        self.challenge_num = challenge_num    # number of challenges used
        self.repeat = repeat        # frequency of same repeated challenge
        self.individuals = np.zeros((parent_size, instance.n))  # weight_arrays
        # mean, step_size,  pop_size,   parent_size,    priorities, cov_matrix,   path_cm,    path_ss
        # m,    sigma,      lambda,     mu,             w_i,        C,            p_c,        p_sigma
        self.mean = np.zeros(instance.n) # mean vector of distribution
        self.step_size = 1              # distance to next distribution
        self.pop_size = pop_size        # number of individuals per generation
        self.parent_size = parent_size  # number of considered individuals
        self.priorities = priorities    # array of consideration proportion
        self.cov_matrix = np.identity(instance.n)  # shape of distribution
        self.path_cm = np.matrix(np.zeros(instance.n)) # cumulated evolution path of covariance matrix
        self.path_ss = np.zeros(instance.n) # cumulated evolution path of step size
        # auxiliary constants
        self.mu_w = 1 / np.sum(np.square(priorities))
        self.c_mu = self.mu_w / instance.n**2
        self.d_sigma = 1 + np.sqrt(self.mu_w / instance.n)
        self.c_1 = 2 / instance.n**2
        self.c_c = 4 / instance.n
        self.c_sigma = 4 / instance.n
        assert len(priorities) == parent_size
        assert int(np.ndarray.sum(priorities)) == 1
        assert (self.c_1 + self.c_mu <= 1)

    def learn(self):
        # this is the main learning method
        # returns XOR-LTFArray with nearly same behavior as learned instance
        while self.num_of_LTFs < self.instance.k:
            self.new_LTF = self.learn_LTF()
            if self.is_different_LTF():
                self.different_LTFs[self.num_of_LTFs] = self.new_LTF
                self.num_of_LTFs += 1
        self.different_LTFs = self.set_pole_of_LTFs(self.instance, self.different_LTFs)
        return LTFArray(self.different_LTFs, self.transform, self.combiner, bias=False)

    def test_is_different_LTF(self):
        class object:
            num_of_LTFs = 1
            different_LTFs = np.array([[0.5, -1, -0.5, 1, 1.5, -1.5],
                                       [0, 1, 2, -0.5, 0.5, 3],
                                       [-0.5, -0.5, -1, 1, -2, 1]])
            new_LTF = np.array([0.5, 0.5, 0.5, -0.5, 0.5, 0.5])
            transform = LTFArray.transform
            combiner = LTFArray.combiner
            build_LTFArrays = CMA.build_LTFArrays
            class instance:
                n = 6
            instance = instance
            challenge_num = 100
        obj = object
        assert CMA.is_different_LTF()


    def is_different_LTF(self):
        # returns True iff new_LTF is different from previously learned LTFs
        if self.num_of_LTFs == 0 :
            return True
        weight_arrays = self.different_LTFs[:self.num_of_LTFs-1, :]
        new_LTFArray = LTFArray(self.new_LTF, LTFArray.transform_atf, LTFArray.combiner_xor)
        different_LTFs = self.build_LTFArrays(weight_arrays)
        challenges = tools.sample_inputs(self.instance.n, self.challenge_num)
        responses = np.empty((self.num_of_LTFs, self.challenge_num))
        responses[0, :] = new_LTFArray.eval(challenges)
        for i, current_LTF in enumerate(different_LTFs):
            responses[i+1, :] = current_LTF.eval(challenges)
        return not self.is_correlated(responses)

    def learn_LTF(self):
        # this is the main CMA-ES algorithm like that from Hansen
        terminate = False
        while not terminate:
            challenges = tools.sample_inputs(
                    self.instance.n, self.challenge_num)
            measured_rels = np.zeros(self.challenge_num)
            while np.sum(measured_rels) < (1-self.unreliability)*self.challenge_num:
                challenges = tools.sample_inputs(
                    self.instance.n, self.challenge_num)
                measured_rels = self.measure_rels(
                    self.instance, challenges, self.challenge_num, self.repeat)
            sorted_individuals = self.fitness(
                challenges, self.challenge_num, measured_rels,
                self.individuals, self.epsilon)
            if self.terminate:
                break
            parent = self.get_parent(
                sorted_individuals, self.parent_size, self.priorities)
            cm_mu = self.get_cm_mu(
                sorted_individuals, self.parent_size, self.priorities)
            self.individuals = self.reproduce(
                self.mean, self.cov_matrix, self.pop_size, self.step_size)
            self.mean = self.update_mean(
                self.mean, self.step_size, parent)
            self.path_cm = self.cumulation_for_cm(
                self.path_cm, self.c_c, self.path_ss, self.instance.n,
                self.mu_w, parent)
            self.path_ss = self.cumulation_for_ss(
                self.path_ss, self.c_sigma, self.mu_w, self.cov_matrix, parent)
            self.cov_matrix = self.update_cm(
                self.cov_matrix, self.c_1, self.c_mu, self.path_cm, cm_mu)
            self.step_size = self.update_ss(
                self.step_size, self.c_sigma, self.d_sigma, self.path_ss)
        return self.new_LTF


    # updating methods of evolution strategies
    @staticmethod
    def reproduce(mean, cov_matrix, pop_size, step_size):
        # returns a new generation of individuals as 2D array (pop_size, n)
        mutations = np.random.multivariate_normal(np.zeros(np.shape(mean)),
                                                  cov_matrix, pop_size)
        duplicated_mean = np.tile(np.matrix(mean), (pop_size, 1))
        return duplicated_mean + (step_size * mutations)

    @staticmethod
    def update_mean(mean, step_size, parent):
        # returns mean of a new population as array (n)
        return mean + step_size*parent

    @staticmethod
    def cumulation_for_cm(path_cm, c_c, path_ss, n, mu_w, parent):
        # returns cumulated evolution path of covariance matrix
        path_cm = path_cm * (1-c_c)
        if(np.linalg.norm(path_ss) < 1.5 * np.sqrt(n)):
            path_cm = path_cm + (np.sqrt(1 - (1-c_c)**2) *
                                 np.sqrt(mu_w) * parent)
        return path_cm

    @staticmethod
    def cumulation_for_ss(path_ss, c_sigma, mu_w, cov_matrix, parent):
        # returns cumulated evolution path of step-size
        cm_eigen_dec = Reliability_based_CMA_ES.modify_eigen_decomposition(cov_matrix)
        return (1-c_sigma) * path_ss + np.sqrt(1 - (1-c_sigma)**2) * \
                                       np.sqrt(mu_w) * cm_eigen_dec @ parent

    @staticmethod
    def update_cm(cov_matrix, c_1, c_mu, path_cm, cm_mu):
        # returns covariance matrix of a new population (pop_size, pop_size)
        return (1 - c_1 - c_mu) * cov_matrix + c_1 * path_cm * path_cm.T \
               + c_mu * cm_mu

    @staticmethod
    def update_ss(step_size, c_sigma, d_sigma, path_ss):
        # returns step-size of a new population
        factor = np.exp((c_sigma / d_sigma) *
                        ((np.linalg.norm(path_ss) / np.sqrt(np.shape(path_ss)[1])) - 1))
        return step_size * factor

    def fitness(self, challenges, challenge_num, measured_rels, individuals, epsilon, epsilon_theo):
        # returns individuals sorted by their fitness
        pop_size = np.shape(individuals)[0]
        built_LTFs = Reliability_based_CMA_ES.build_LTFArrays(individuals)
        delay_diffs = Reliability_based_CMA_ES.get_delay_differences(
            built_LTFs, pop_size, challenges, challenge_num)
        reliabilities = Reliability_based_CMA_ES.get_reliabilities(delay_diffs, epsilon_theo)
        correlations = Reliability_based_CMA_ES.get_correlations(reliabilities, measured_rels)
        for i in range(np.shape(correlations)):
            if correlations[i] > 0.9*epsilon:
                self.new_LTF = individuals[i, :]
                self.terminate = True
        return Reliability_based_CMA_ES.sort_individuals(individuals, correlations)

    # methods for calculating fitness
    @staticmethod
    def build_LTFArrays(individuals):
        # returns iterator over ltf_arrays created out of every individual
        transform = LTFArray.transform_id
        combiner = LTFArray.combiner_xor
        pop_size = np.shape(individuals)[0]
        for i in range(pop_size):
            yield LTFArray(np.matrix(individuals[i, :]), transform, combiner,
                           bias=False)

    @staticmethod
    def get_delay_differences(ltf_arrays, pop_size, challenges, challenge_num):
        # returns 2D array of delay differences for all challenges on every
        #   individual
        delay_diffs = np.empty((pop_size, challenge_num))
        for i, ltf_array in enumerate(ltf_arrays):
            for j, challenge in challenges:
                delay_diffs[i][j] = ltf_array.val(challenge)
        return delay_diffs

    @staticmethod
    def get_reliabilities(delay_diffs, epsilon_theo):
        # returns 2D array of reliabilities for all challenges on every individual
        reliabilities = np.zeros(np.shape(delay_diffs))
        indices_of_unreliable = delay_diffs > -epsilon_theo and delay_diffs < epsilon_theo
        reliabilities[indices_of_unreliable] = 1
        return reliabilities

    @staticmethod
    def get_correlations(reliabilities, measured_rels):
        # returns array of pearson correlation coefficients between reliability
        #   array of individual and instance for all individuals
        pop_size = np.shape(reliabilities)[0]
        correlations = np.empty(pop_size)
        for i in range(pop_size):
            correlations[i] = np.corrcoef(reliabilities[i,:], measured_rels)[0, 1]
        return correlations

    @staticmethod
    def sort_individuals(individuals, correlations):
        # returns 2D array of individuals as given from input, but sorted through
        #   correlation coefficients
        sorted_indices = np.argsort(correlations)
        return individuals[sorted_indices[::-1]]


    # helping methods
    @staticmethod
    def get_epsilon_theo(n, unreliability):
        return np.sqrt(n) * stats.norm.ppf(0.5 + unreliability/2)

    @staticmethod
    def set_pole_of_LTFs(instance, different_LTFs):
        # returns the correctly polarized XOR-LTFArray
        challenge_num = 10
        responses = np.empty((2, challenge_num))
        challenges = tools.sample_inputs(instance.n, challenge_num)
        cs1, cs2 = itertools.tee(challenges)
        xor_LTFArray = LTFArray(different_LTFs, LTFArray.transform_id,
                                LTFArray.combiner_xor)
        responses[0, :] = instance.eval(cs1)
        responses[1, :] = xor_LTFArray.eval(cs2)
        difference = np.sum(np.abs(responses[0, :] - responses[1, :]))
        print(responses, '\ndiff =', difference)
        if difference > challenge_num:
            different_LTFs[0, :] *= -1
        return different_LTFs

    @staticmethod
    def is_correlated(responses):
        # returns True iff 2 response arrays are more than 75% equal
        (num_of_LTFs, num) = np.shape(responses)
        for i in range(1, num_of_LTFs):
            differences = np.sum(np.abs(responses[0, :] - responses[i, :])) / 2
            if differences < 0.25*num or differences > 0.75*num:
                return True
        return False

    @staticmethod
    def measure_rels(instance, challenges, challenge_num, repeat):
        # returns array of measured reliabilities of instance
        measured_rels = np.zeros(challenge_num)
        responses = np.empty((challenge_num, repeat))
        for i, challenge in enumerate(challenges):
            for j in range(repeat):
                responses[i, j] = instance.eval(challenge)
            measured_rels[i] = np.abs(np.sum(responses[i, :])) / 2
        return measured_rels

    @staticmethod
    def get_parent(sorted_individuals, parent_size, priorities):
        # returns the weighted sum of the fittest individuals
        parent = np.empty(np.shape(sorted_individuals)[1])
        for i in range(parent_size):
            parent += priorities[i] * sorted_individuals[i, :]
        return parent

    @staticmethod
    def get_cm_mu(sorted_individuals, parent_size, priorities):
        # returns the weighted sum of the fittest individuals
        cm_mu = np.empty(np.shape(sorted_individuals)[1])
        for i in range(parent_size):
            cm_mu += priorities[i] * sorted_individuals[i,:] * \
                     sorted_individuals[i,:].T
        return cm_mu

    @staticmethod
    def modify_eigen_decomposition(matrix):
        # returns modified eigen-decomposition of matrix A = B * D^2 * B^T
        #   B * D^(-1) * B^T
        eigen_values, eigen_vectors = np.linalg.eigh(matrix)
        diagonal = np.sqrt((np.diag(eigen_values)))
        diagonal_inverse = np.linalg.inv(diagonal)
        return eigen_vectors @ diagonal_inverse @ eigen_vectors.T
