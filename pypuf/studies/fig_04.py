from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression, Parameters
from pypuf.studies.base import Study


class Fig04(Study):

    CRPS = 30000
    SAMPLE_SIZE = 100
    LR_TRANSFORMATIONS = ['id', 'random', 'lightweight_secure', 'fixed_permutation']
    SIZE = (64, 4)

    SHUFFLE = True

    def name(self):
        return 'fig_04'

    def experiments(self):
        e = []
        (n, k) = self.SIZE
        for transformation in self.LR_TRANSFORMATIONS:
            for i in range(self.SAMPLE_SIZE):
                e.append(
                    ExperimentLogisticRegression(
                        progress_log_prefix=None,
                        parameters=Parameters(
                            seed_instance=314159 + i,
                            seed_model=265358 + i,
                            seed_challenge=979323 + i,
                            seed_distance=846264 + i,
                            n=n,
                            k=k,
                            transformation=transformation,
                            combiner='xor',
                            N=self.CRPS,
                            mini_batch_size=0,
                            convergence_decimals=2,
                            shuffle=False,
                        )
                    )
                )
        return e
        # TODO add correlation attack

    def plot(self):
        print(self.experimenter.results[['transformation', 'accuracy']])
