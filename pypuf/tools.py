"""
This module provides a set of different functions which can be used e.g. for challenges generation, statistical purpose
or polynomial division. The spectrum is rich and the functions are used in many different modules. Its a kind of a
helper module.
"""
import itertools
from numpy import count_nonzero, array, append, zeros, vstack, mean, prod, ones, dtype, full, shape, copy, int8
from numpy import sum as np_sum
from numpy import abs as np_abs
from numpy.random import RandomState
from random import sample

BIT_TYPE = int8


aes_s_box = array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])


def binary_to_int(binary):
    assert all((binary==0) | (binary==1)), 'The input must be a binary array!'
    number = 0
    n = len(binary)
    for i in range(n):
        number += binary[i] * 2**(n-i-1)
    return number


def array_1_1_to_int(array_1_1):
    assert all((array_1_1==-1) | (array_1_1==1)), 'The input must be a {1,-1}-array!'
    cp = copy(array_1_1)
    place(cp, cp==1, [0])
    place(cp, cp==-1, [1])
    return binary_to_int(cp)


def byte_to_array_1_1(byte):
    assert type(byte)==uint8, 'The input must be of type numpy.uint8!'
    binary = unpackbits(array([byte], dtype=uint8))
    res = array(binary, dtype=int8)
    place(res, res==1, [-1])
    place(res, res==0, [1])
    return res


def substitute_aes(challenge):
    assert all((challenge == -1) | (challenge == 1)), 'The input must be a {1,-1}-array!'
    n = len(challenge)
    num = int(n / 8)
    s = copy(challenge)
    for i in range(num):
        number = array_1_1_to_int(challenge[8*i:8*(i+1)])
        s[8*i:8*(i+1)] = byte_to_array_1_1(aes_s_box[number].astype(uint8))
    return s


def random_input(n, random_instance=RandomState()):
    """
    This method generates an array with random integer.
    By default a fresh `numpy.random.RandomState`instance is used.
    :param n: int
              Number of bits which should be generated
    :param random_instance: numpy.random.RandomState
                            The PRNG which is used to generate the output bits.
    :returns: array of int8
              A pseudo random array of -1 and 1
    """
    return random_inputs(n, 1, random_instance)[0]


def all_inputs(n):
    """
    This functions generates a iterator which produces all possible {-1,1}-vectors.
    :param int
           Length of a n bit vector
    :returns: array of int8
              An array with all possible different {-1,1}-vectors of length `n`.
    """
    return (array(list(itertools.product((-1, +1), repeat=n)))).astype(BIT_TYPE)


def random_inputs(n, num, random_instance=RandomState()):
    """
    This function generates an iterator for a random sample of {-1,1}-vectors of length `n` (with replacement).
    If no PRNG provided, a fresh `numpy.random.RandomState` instance is used.
    :param n: int
              Length of a n bit vector
    :param num: int
                Number of n bit vector
    :param random_instance: numpy.random.RandomState
                            The PRNG which is used to generate the arrays.
    :return: array of num {-1,1} int8 arrays
             An array with num random {-1,1} int arrays.
    """
    return 2 * random_instance.randint(0, 2, (num, n), dtype=BIT_TYPE) - 1


def sample_inputs(n, num, random_instance=RandomState()):
    """
    This function generates an iterator for either random samples of {-1,1}-vectors of length `n` if `num` < 2^n,
    and an iterator for all {-1,1}-vectors of length `n` otherwise.
    Note that we return only 2^n vectors even with `num` > 2^n.
    In other words, the output of this function is deterministic if and only if num >= 2^n.
    :param n: int
              Length of a n bit vector
    :param num: int
                Number of n bit vector
    :param random_instance: numpy.random.RandomState
                            The PRNG which is used to generate the arrays.
    :return: array of num {-1,1} int8 arrays
             An array with num random {-1,1} int arrays depending on num and n.
    """
    return random_inputs(n, num, random_instance) if num < 2 ** n else all_inputs(n)


def append_last(arr, item):
    """
    Returns an array for a given array arr and appends on the lowest level the element item.
    :param arr: n dimensional array of type
                       Matrix with initial values
    :param item: type
                 element to be appended
    :return: n dimensional array of type
             initial arr with appended element item
    """
    assert arr.dtype == dtype(type(item)), 'The elements of arr and item must be of the same type, but the array has ' \
                                           'type %s and the item has type %s.' % (arr.dtype, dtype(type(item)))
    dimension = list(shape(arr))
    assert len(dimension) >= 1, 'arr must have at least one dimension.'
    # the lowest level should contain one item
    dimension[-1] = 1
    # create an array white shape(array) where the lowest level contains only one item
    item_arr = full(dimension, item, dtype=BIT_TYPE)
    # the item should be appended at the lowest level
    axis = len(dimension) - 1
    return append(arr, item_arr, axis=axis)


def approx_dist(instance1, instance2, num, random_instance=RandomState()):
    """
    Approximate the distance of two functions instance1, instance2 by evaluating instance1 random set of inputs.
    instance1, instance2 needs to have eval() method and input_length member.
    :param instance1: pypuf.simulation.arbiter_based.base.Simulation
    :param instance2: pypuf.simulation.arbiter_based.base.Simulation
    :param num: int
                Number of n bit vector
    :param random_instance: numpy.random.RandomState
                            The PRNG which is used to generate the input arrays.
    :return: float
             Probability (randomly uniform x) for instance1.eval(x) != instance2.eval(x)
    """
    assert instance1.n == instance2.n
    inputs = random_inputs(instance1.n, num, random_instance=random_instance)
    return (num - count_nonzero(instance1.eval(inputs) == instance2.eval(inputs))) / num


def approx_fourier_coefficient(s, training_set):
    """
    Approximate the Fourier coefficient of a function on the subset `s`
    by evaluating the function on `training_set`
    :param s: list of int8
                  A {0,1}-array indicating the coefficient's index set
    :param training_set: pypuf.tools.TrainingSet
    :return: float
             The approximated value of the coefficient
    """
    assert_result_type(s)
    assert_result_type(training_set.challenges)
    return mean(training_set.responses * chi_vectorized(s, training_set.challenges))


def chi_vectorized(s, inputs):
    """
    Parity function of inputs on indices in s.
    :param s: list of int8
              A {0,1}-array indicating the index set
    :param inputs: array of int8 shape(N,n)
                   {-1,1}-valued inputs to be evaluated.
    :return: array of int8 shape(N)
             chi_s(x) = prod_(i in s) x_i for all x in inputs (`latex formula`)
    """
    assert_result_type(s)
    assert_result_type(inputs)
    assert len(s) == len(inputs[0])
    result = inputs[:, s > 0]
    if result.size == 0:
        return ones(len(inputs), dtype=BIT_TYPE)
    return prod(result, axis=1, dtype=BIT_TYPE)


def compare_functions(function1, function2):
    """
    compares two functions on bytecode layer
    :param function1: function object
    :param function2: function object
    :return: bool
    """
    function1_code = function1.__code__
    function2_code = function2.__code__
    functions_equal = function1_code.co_code == function2_code.co_code
    # The bytcode maybe differ from each other https://stackoverflow.com/a/20059029
    functions_equal &= function1_code.co_name == function2_code.co_name
    return functions_equal and function1_code.co_filename == function2_code.co_filename


def transform_challenge_01_to_11(challenge):
    """
    This function is used to transform a challenge from 0,1 notation to -1,1 notation.
    :param challenge: array of int8
                      Challenge vector in 0,1 notation
    :return: array of int8
             Same vector in -1,1 notation
    """
    assert_result_type(challenge)
    res = copy(challenge)
    res[res == 1] = -1
    res[res == 0] = 1
    return res


def transform_challenge_11_to_01(challenge):
    """
    This function is used to transform a challenge from -1,1 notation to 0,1 notation.
    :param challenge: array of int8
                      Challenge vector in -1,1 notation
    :return: array of int8
             Same vector in 0,1 notation
    """
    assert_result_type(challenge)
    res = copy(challenge)
    res[res == 1] = 0
    res[res == -1] = 1
    return res


def poly_mult_div(challenge, irreducible_polynomial, k):
    """
    Return the list of polynomials
        [challenge^2, challenge^3, ..., challenge^(k+1)] mod irreducible_polynomial
    based on the challenge challenge and the irreducible polynomial irreducible_polynomial.
    :param challenge: array of int8
                      Challenge vector in 0,1 notation
    :param irreducible_polynomial: array of int8
                                   Vector in 0,1 notation
    :param k: int
              Number of PUFs
    :return: array of int8
             Array of polynomials
    """
    import polymath as pm
    assert_result_type(challenge)
    assert_result_type(irreducible_polynomial)
    # TODO Change the type to int8 or uint8
    challenge = challenge.astype('uint8')
    irreducible_polynomial = irreducible_polynomial.astype('uint8')
    # TODO Change the type to int8 or uint8
    # challenge = challenge.astype('int64')
    # irreducible_polynomial = irreducible_polynomial.astype('int64')
    c_original = challenge
    res = None
    for i in range(k):
        challenge = pm.polymul(challenge, c_original)
        challenge = pm.polymodpad(challenge, irreducible_polynomial)
        if i == 0:
            res = array([challenge], dtype='int8')
        else:
            res = vstack((res, challenge))
    res = res.astype(BIT_TYPE)
    assert_result_type(res)
    return res


def approx_stabilities(instance, num, reps, random_instance=RandomState()):
    """
    This function approximates the stability of the given `instance` for
    `num` challenges evaluating it `reps` times per challenge. The stability
    is the probability that the instance gives the correct response when
    evaluated.
    :param instance: pypuf.simulation.base.Simulation
                     The instance for the stability approximation
    :param num: int
                Amount of challenges to be evaluated
    :param reps: int
                 Amount of repetitions per challenge
    :return: array of float
             Array of the stabilities for each challenge
    """

    challenges = sample_inputs(instance.n, num, random_instance)
    responses = zeros((reps, num))
    for i in range(reps):
        responses[i, :] = instance.eval(challenges)
    return 0.5 + 0.5 * np_abs(np_sum(responses, axis=0)) / reps


def assert_result_type(arr):
    """
    This function checks the type of the array to match the RESULT_TYPE
    :param arr: array of arbitrary type
    """
    assert arr.dtype == dtype(BIT_TYPE), 'Must be an array of {0}. Got array of {1}'.format(BIT_TYPE, arr.dtype)


class ChallengeResponseSet:
    """
    A set of challenges and corresponding responses.
    """

    def __init__(self, challenges, responses):
        """
        Create a set of challenges and corresponding responses. Note that the order of the
        challenges and responses parameter is relevant.
        :param challenges: List of challenges
        :param responses: List of responses, ordered accordingly
        """
        self.challenges = challenges
        self.responses = responses
        assert len(self.challenges) == len(self.responses)
        self.N = len(self.challenges)

    def random_subset(self, N):
        """
        Gives a random subset of this challenge response set.
        :param N: Either a relative (to the total number) or absolute number of challenges.
        :return: A random subset samples from this challenge response set.
        """
        if N < 1:
            N = int(self.N * N)
        return self.subset(sample(range(self.N), N))

    def block_subset(self, i, total):
        """
        Gives the i-th block of this challenge response set.
        :param i: Index of the block that is to be returned.
        :param total: Total number of blocks.
        :return: A challenge response set.
        """
        return self.subset(range(
            int(i / total * self.N),
            int((i + 1) / total * self.N)
        ))

    def subset(self, subset_slice):
        """
        Gives the subset of this challenge response set defined by the slice given.
        :param subset_slice: A python array slice
        :return: A challenge response set defined accordingly
        """
        return ChallengeResponseSet(
            challenges=self.challenges[subset_slice],
            responses=self.responses[subset_slice]
        )


class TrainingSet(ChallengeResponseSet):
    """
    Basic data structure to hold a collection of challenge response pairs.
    Note that this is, strictly speaking, not a set.
    """

    def __init__(self, instance, N, random_instance=RandomState()):
        """
        :param instance: pypuf.simulation.base.Simulation
                         Instance which is used to generate responses for random challenges.
        :param N: int
                  Number of desired challenges
        :param random_instance: numpy.random.RandomState
                                PRNG which is used to draft challenges.
        """
        self.instance = instance
        challenges = array(list(sample_inputs(instance.n, N, random_instance=random_instance)))
        super().__init__(
            challenges=challenges,
            responses=instance.eval(challenges)
        )
