from numpy import array
from numpy.random import RandomState
from pypuf.learner.regression.input_transform_attack import RandomPermutationAttack
from pypuf.tools import ChallengeResponseSet, sample_inputs, generate_random_permutations, approx_dist
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


prng = RandomState(0xaaaa)
n = 64
k = 4
N = 50000
weight_array = LTFArray.normal_weights(k=k, n=n)
combiner = LTFArray.combiner_xor
seeds = prng.randint(low=0, high=2**32, size=k)
permutations = generate_random_permutations(k=k, n=n, seed=prng.randint(2**32))
#transform = LTFArray.generate_permutation_transform(permutations=permutations, atf=False)
transform = LTFArray.transform_random
instance = LTFArray(
    weight_array=weight_array,
    combiner=combiner,
    transform=transform
)

challenges = (array(list(sample_inputs(n=n, num=N, random_instance=prng)))).astype('int8')
responses = instance.eval(challenges)
ratio = 0.8
num_training = int((1-ratio)*N)
num_validation = int(ratio*N)
training_set = ChallengeResponseSet(challenges[:-num_training], responses[:-num_training])
validation_set = ChallengeResponseSet(challenges[:-num_validation], responses[:-num_validation])
attack = RandomPermutationAttack(
    n=n,
    k=k,
    training_set=training_set,
    validation_set=validation_set,
    seed=prng.randint(2**32),
    transform=transform
)

model = attack.learn()

num = 10000
accuracy = 1 - approx_dist(
    instance1=instance,
    instance2=model,
    num=num,
    random_instance=RandomState(prng.randint(2**32))
)

print(accuracy)
