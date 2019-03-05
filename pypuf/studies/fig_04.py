from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression, Parameters
from pypuf.studies.base import Study
from matplotlib import pyplot
from seaborn import distplot
from numpy import arange, ones_like


class Fig04(Study):

    CRPS = 30000
    SAMPLE_SIZE = 110
    LR_TRANSFORMATIONS = ['atf', 'random', 'lightweight_secure', 'fixed_permutation']
    SIZE = (64, 4)

    SHUFFLE = True

    NICE_TRANSFORMATION_NAMES = {
        'atf': 'Classic',
        'fixed_permutation': 'Permutation-Based',
        'lightweight_secure': 'Lightweight Secure',
        'random': 'Pseudorandom',
    }

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
        results = self.experimenter.results
        groups = results[results['transformation'] != 'id'].groupby(['transformation'])
        figure, axes = pyplot.subplots(nrows=len(groups), ncols=1)
        figure.subplots_adjust(hspace=3)
        figure.set_size_inches(w=5, h=1.5 * len(groups))
        for axis, (transformation, group_results) in zip(axes, groups):
            axis.set_title('{} using {:,} CRPs'.format(self.NICE_TRANSFORMATION_NAMES[transformation], self.CRPS))
            axis.set_xlim([.48, 1])
            distplot(
                group_results[['accuracy']],
                ax=axis,
                kde=False,
                bins=arange(.48, 1.01, .01),
                hist=True,
                norm_hist=False,
                color='blue',
                hist_kws={
                    'alpha': 1,
                    # the following line turns the histogram of absolute frequencies into one with relative frequencies
                    'weights': ones_like(group_results[['accuracy']]) / float(len(group_results[['accuracy']]))
                }
            )
        axes[-1].set_xlabel('accuracy')
        axes[(len(groups) - 1) // 2].set_ylabel('rel. frequency')
        figure.tight_layout()
        figure.savefig('figures/' + self.name() + '.pdf')
        pyplot.close(figure)
