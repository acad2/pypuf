from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression, Parameters
from pypuf.studies.base import Study
from matplotlib import pyplot
from seaborn import distplot
from numpy.ma import arange


class Fig04(Study):

    CRPS = 30000
    SAMPLE_SIZE = 100
    LR_TRANSFORMATIONS = ['id', 'atf', 'random', 'lightweight_secure', 'fixed_permutation']
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
        groups = self.experimenter.results.groupby(['transformation'])
        figure, axes = pyplot.subplots(nrows=len(groups), ncols=1)
        figure.subplots_adjust(hspace=3)
        figure.set_size_inches(w=6, h=2 * len(groups))
        for axis, (transformation, group_results) in zip(axes, groups):
            axis.set_title(transformation)
            axis.set_xlim([.48, 1])
            axis.set_ylabel('rel. frequency')
            distplot(
                group_results[['accuracy']],
                ax=axis,
                kde=False,
                bins=arange(.48, 1.0, .02),
                hist=True,
                norm_hist=True,
            )
        axes[-1].set_xlabel('accuracy')
        figure.tight_layout()
        figure.savefig('figures/' + self.name() + '.pdf')
        pyplot.close(figure)
