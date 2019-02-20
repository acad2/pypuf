"""
Hyperparameter Optimization: Testing Dropout.
"""
from numpy.random import RandomState
from pypuf.experiments.experiment.mlp_keras import ExperimentMLP
from pypuf.experiments.experimenter import Experimenter
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


SEED_RANGE = 2 ** 32
SAMPLES_PER_POINT = 10
BATCH_SIZE = 1000
ITERATION_LIMIT = 1000
LOG_NAME = 'keras_mlp_DO'


definitions = [
    # k  n   CRP_min max
    [1, 64, (1000, 3000), False, None, None, None, None],
    #[2, 64, (400, 8000), False, None, None, None, None],
    #[3, 64, (1600, 32000), False, None, None, None, None],
    #[4, 64, (6400, 128000), False, None, None, None, None],
]


for d in definitions:

    k = d[0]
    n = d[1]
    min_CRPs = d[2][0]
    max_CRPs = d[2][1]
    preprocess = d[3]
    seed_i = RandomState(d[4]).randint(SEED_RANGE)
    seed_c = RandomState(d[5]).randint(SEED_RANGE)
    seed_m = RandomState(d[6]).randint(SEED_RANGE)
    seed_a = RandomState(d[7]).randint(SEED_RANGE)
    log_name = LOG_NAME + '_{0}_{1}'.format(k, n)

    experiments = list()
    parameters = list()

    for num in range(min_CRPs, max_CRPs + min_CRPs, min_CRPs):
        parameters.append([k, n, LTFArray.transform_id, LTFArray.combiner_xor, num])
        parameters.append([k, n, LTFArray.transform_atf, LTFArray.combiner_xor, num])
        parameters.append([k, n, LTFArray.transform_aes_substitution, LTFArray.combiner_xor, num])
        parameters.append([k, n, LTFArray.transform_lightweight_secure_original, LTFArray.combiner_xor, num])
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

    experimenter = Experimenter(experiments=experiments, log_name=log_name)
    experimenter.run()
