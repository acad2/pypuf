"""
Hyperparameter Optimization: Testing Dropout.
"""
from pypuf.experiments.experiment.mlp_keras import ExperimentMLP
from pypuf.experiments.experimenter import Experimenter
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from numpy.random import RandomState


SEED_RANGE = 2 ** 32
SAMPLES_PER_POINT = 10
BATCH_SIZE = 1000
ITERATION_LIMIT = 1000
LOG_NAME = 'keras_mlp'


definitions = [
    [4, 64, (400000, 800000), True, 1, 2, 3, 4],
]


for k, n, (min_CRPs, max_CRPs), preprocess, seed_i, seed_c, seed_m, seed_a in definitions:

    log_name = LOG_NAME + '_{0}_{1}'.format(k, n)
    experiments = list()
    parameters = list()

    for num in range(min_CRPs, max_CRPs + min_CRPs, min_CRPs):
        parameters.append([k, n, LTFArray.transform_id, LTFArray.combiner_xor, num])
        parameters.append([k, n, LTFArray.transform_atf, LTFArray.combiner_xor, num])
        parameters.append([k, n, LTFArray.transform_aes_substitution, LTFArray.combiner_xor, num])
        parameters.append([k, n, LTFArray.transform_lightweight_secure, LTFArray.combiner_xor, num])
        if n == 64:
            parameters.append([k, n, LTFArray.transform_fixed_permutation, LTFArray.combiner_xor, num])
        parameters.append([k, n, LTFArray.transform_random, LTFArray.combiner_xor, num])

    for s in range(SAMPLES_PER_POINT):
        for p in range(len(parameters)):
            experiment = ExperimentMLP(
                log_name=log_name + '_{0}'.format(p),
                N=parameters[p][4],
                n=parameters[p][1],
                k=parameters[p][0],
                transformation=parameters[p][2],
                combiner=parameters[p][3],
                preprocess=preprocess,
                batch_size=BATCH_SIZE,
                iteration_limit=ITERATION_LIMIT,
                seed_simulation=RandomState((seed_i + p * SAMPLES_PER_POINT + s) % SEED_RANGE).randint(SEED_RANGE),
                seed_challenges=RandomState((seed_c + p * SAMPLES_PER_POINT + s) % SEED_RANGE).randint(SEED_RANGE),
                seed_model=RandomState((seed_m + p * SAMPLES_PER_POINT + s) % SEED_RANGE).randint(SEED_RANGE),
                seed_accuracy=RandomState((seed_a + p * SAMPLES_PER_POINT + s) % SEED_RANGE).randint(SEED_RANGE),
            )
            experiments.append(experiment)

    experimenter = Experimenter(result_log_name=log_name, cpu_limit=1)
    for e in experiments:
        experimenter.queue(e)
    experimenter.run()
