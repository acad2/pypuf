"""
Hyperparameter Optimization: Testing Dropout.
"""
from pypuf.experiments.experiment.mlp_keras import ExperimentMLP
from pypuf.experiments.experimenter import Experimenter
from pypuf.plots import SuccessRatePlot
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from numpy.random.mtrand import RandomState


CPU_LIMIT = 3
SEED_RANGE = 2 ** 32
SAMPLES_PER_POINT = 5
BATCH_SIZE = 1000
ITERATION_LIMIT = 10000
PRINT_KERAS = False
LOG_NAME = 'test_plot'   # ''keras_mlp'


plots = list()


def update_plots(_=None):
    for p in plots:
        p.plot()


definitions = [
    [2, 64, (5000, 20000), True, 1, 2, 3, 4],
    #[4, 64, (200000, 800000), True, None, None, None, None],
    #[4, 64, (200000, 800000), False, None, None, None, None],
    #[6, 64, (1000000, 4000000), True, None, None, None, None],
    #[6, 64, (1000000, 4000000), False, None, None, None, None]
]


for k, n, (min_CRPs, max_CRPs), preprocess, seed_s, seed_c, seed_m, seed_a in definitions:

    log_name = LOG_NAME + '_{0}_{1}'.format(k, n)
    experiments = list()
    parameters = list()
    seed_s = seed_s if seed_s is not None else RandomState().randint(SEED_RANGE)
    seed_c = seed_c if seed_c is not None else RandomState().randint(SEED_RANGE)
    seed_m = seed_m if seed_m is not None else RandomState().randint(SEED_RANGE)
    seed_a = seed_a if seed_a is not None else RandomState().randint(SEED_RANGE)

    experimenter = Experimenter(result_log_name=log_name, cpu_limit=CPU_LIMIT)
    """
    plot = SuccessRatePlot(
        filename='figures/' + LOG_NAME + '-{0}-{1}.pdf'.format(n, k),
        results=experimenter.results,
        group_by='transformation',
    )
    plots.append(plot)
    """

    for num in range(min_CRPs, max_CRPs + min_CRPs, min_CRPs):
        #parameters.append([k, n, LTFArray.transform_id, LTFArray.combiner_xor, num])
        parameters.append([k, n, LTFArray.transform_atf, LTFArray.combiner_xor, num])
        parameters.append([k, n, LTFArray.transform_aes_substitution, LTFArray.combiner_xor, num])
        parameters.append([k, n, LTFArray.transform_lightweight_secure, LTFArray.combiner_xor, num])
        #if n == 64:
            #parameters.append([k, n, LTFArray.transform_fixed_permutation, LTFArray.combiner_xor, num])
        parameters.append([k, n, LTFArray.transform_random, LTFArray.combiner_xor, num])

    for s in range(SAMPLES_PER_POINT):
        for p in range(len(parameters)):
            experiment = ExperimentMLP(
                log_name=None,
                N=parameters[p][4],
                n=parameters[p][1],
                k=parameters[p][0],
                transformation=parameters[p][2],
                combiner=parameters[p][3],
                preprocess=preprocess,
                batch_size=BATCH_SIZE,
                iteration_limit=ITERATION_LIMIT,
                print_keras=PRINT_KERAS,
                seed_simulation=(seed_s + p * SAMPLES_PER_POINT + s) % SEED_RANGE,
                seed_challenges=(seed_c + p * SAMPLES_PER_POINT + s) % SEED_RANGE,
                seed_model=(seed_m + p * SAMPLES_PER_POINT + s) % SEED_RANGE,
                seed_accuracy=(seed_a + p * SAMPLES_PER_POINT + s) % SEED_RANGE,
            )
            experiments.append(experiment)

    for e in experiments:
        experiment_id = experimenter.queue(e)
        #plot.experiment_ids.append(experiment_id)
    #experimenter.update_callback = update_plots
    experimenter.run()
    #update_plots()
