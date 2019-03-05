"""
Plots to visualize results by experiments or studies.
"""

from pypuf.experiments.result import ExperimentResult
from numpy import mean as mean_np, quantile as quantile_np, sum as sum_np, unique, zeros
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
from seaborn import lineplot, set_style
from itertools import cycle
from numpy import zeros


class SuccessRatePlot:
    """
    Show the ratio of experiment results with accuracy higher than a given threshold to the total number of results,
    dependent on the number of examples in the training set.
    """
    def __init__(self, filename, group_by, experiment_hashes=None, success_threshold=.7, group_labels=None):
        """
        Prepare a plot
        :param filename: destination file (PDF)
        :param results: an object with results keyed with experiment ids
        :param group_by: determines among which groups success rates are computed
        :param experiment_hashes: ids of results that shall be used
        :param success_threshold: defines what is considered a success
        :param group_labels: can be used to translate group_by values to human-readable names, e.g.
            { 'permutation_fixed': 'A fixed permutation', ... }
        """
        self.title_size = 6
        self.tick_size = 4
        self.x_label_size = 5
        self.legend_size = 4

        self.n = None
        self.k = None
        self.experiment = None
        self.title = None
        self.results = None

        self.filename = filename
        self.experiment_hashes = experiment_hashes or []
        self.success_threshold = success_threshold
        self.group_by = group_by
        self.group_labels = {} if group_labels is None else group_labels

        self.figure = plt.figure()
        self.figure.set_size_inches(w=3.34, h=1.7)
        self.axis = self.figure.add_subplot(1, 1, 1)

        self.plot_data = None

    def plot(self, results):
        """
        Draw the plot and save it to the file system.
        """
        if self.experiment_hashes:
            results = results.loc[results['experiment_hash'].isin(self.experiment_hashes)]

        if results.empty:
            return

        self.axis.clear()

        self.axis.set_xlabel('number of examples in the training set', size=self.x_label_size)
        # self.axis.set_ylabel('success rate (threshold: %f)' % success_threshold, size=ylabelsize)
        for w in ['major', 'minor']:
            self.axis.tick_params(width=0.5, which=w, labelsize=self.tick_size)
        for axis in ['top', 'bottom', 'left', 'right']:
            self.axis.spines[axis].set_linewidth(0.5)

        n_k_combinations = results.groupby(['n', 'k']).size().reset_index().values[:, :2]
        if not n_k_combinations.size:
            return
        assert len(n_k_combinations) == 1,\
            "For SuccesRatePlot, all experiments must be run with same n and k, but there were %s." % n_k_combinations
        self.n = results['n'].unique()[0]
        self.k = results['k'].unique()[0]
        self.title = 'Success Rate on %i-bit, %i-XOR Arbiter PUF' % (
            self.n,
            self.k,
        )
        self.axis.set_title(self.title, size=self.title_size)

        color = cycle([plt.cm.colors.rgb2hex(x) for x in [
            "#F44336",  # Red
            "#4CAF50",  # Green
            "#03A9F4",  # Indigo
            "#FFC107",  # Amber
            "#795548",  # Brown
        ]])

        marker = cycle(["8", "s", "^", "x", "D"])

        self.plot_data = {}

        self.axis.set_xscale("log")
        min_N = results['N'].min()
        max_N = results['N'].max()
        if min_N == max_N:
            max_N += 1
        self.axis.set_xlim([min_N, max_N])
        self.axis.set_ylim([-.02, 1.02])

        for name, group_results in results.groupby([self.group_by]):
            Ns = group_results['N'].unique()
            success_rate = zeros((len(Ns), 2))
            for idx, N in enumerate(Ns):
                group_N_results = group_results[group_results['N'] == N]
                success_rate[idx, 0] = N
                success_rate[idx, 1] = \
                    group_N_results[
                        group_N_results['accuracy'] > self.success_threshold
                    ].size / group_N_results.size
            success_rate.sort(axis=0)
            self.plot_data[name] = success_rate

            label = self.group_labels[name] if name in self.group_labels else name
            col = next(color)
            mar = next(marker)
            self.axis.scatter(success_rate[:, 0], success_rate[:, 1], 8, color=col, label=label, marker=mar, alpha=.7)
            self.axis.plot(success_rate[:, 0], success_rate[:, 1], '-', color=col, linewidth=0.8, alpha=.7)

        if self.axis.has_data():
            legend = self.axis.legend(loc='best', fontsize=self.legend_size)
            self.figure.savefig(self.filename, bbox_extra_artists=(legend,), bbox_inches='tight', pad_inches=0)


class AccuracyPlotter:

    def __init__(self, estimator=None, group_by='transformation', grid=False):
        assert estimator is None or isinstance(estimator, tuple)
        self.estimator = estimator
        self.group_by = group_by
        self.grid = grid
        self.df = None
        self.figure = plt.figure()
        self.axis = self.figure.add_subplot()

    @staticmethod
    def get_estimator(estimator):
        if estimator is None:
            return mean_np
        elif estimator[0] == 'quantile':
            def quantile(a):
                return quantile_np(a, estimator[1], interpolation='linear')
            return quantile
        elif estimator[0] == 'success':
            def success(a):
                return sum_np(a >= estimator[1]) / len(a)
            return success
        return

    def get_data_frame(self, source):
        """ names has to be removed after logs of DataFrames contains header """
        names = ['seed_s', 'seed_m', 'n', 'k', 'N', 'accuracy', 'acc_val', 'acc_train', 'epochs', 'time',
                 'transformation', 'combiner']
        self.df = read_csv(source, names=names, sep='\t', header=None) if isinstance(source, str) else source
        return

    def get_title(self):
        k = self.df['k'][0]
        n = self.df['n'][0]
        return 'Comparison of Accuracies \non ({0}, {1})-XOR-Arbiter PUFs \n'.format(k, n) + \
               'using {0} as estimator'.format('{0} with p={1}'.format(self.estimator[0], self.estimator[1])
                                               if self.estimator is not None else 'mean')

    def create_plot(self):
        assert isinstance(self.df, DataFrame)
        set_style('white')
        estimator = self.get_estimator(self.estimator)
        self.axis = lineplot(
            x='N',
            y='accuracy',
            hue=self.group_by,
            estimator=estimator,
            ci=None,
            data=self.df,
            alpha=0.6
        )
        legend_alpha = 1
        ticks = unique(self.df['N'])
        max_tick = max(ticks)
        self.axis.set_xlim([0, max_tick + (max_tick / 200)])
        self.axis.set_ylim([0.495, 1.005])
        self.axis.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        self.axis.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
        self.axis.xaxis.set_minor_locator(plt.MultipleLocator(ticks[1] - ticks[0]))
        self.axis.tick_params(which='major', width=1.0, length=5)
        self.axis.tick_params(which='minor', width=0.5, length=2)
        if self.grid:
            legend_alpha = 0.5
            plt.grid(True)
            self.axis.grid(b=True, which='major', color='lightgrey', linewidth=1)
            self.axis.grid(b=True, which='minor', color='lightgrey', linewidth=0.5)
        legend = self.axis.legend(loc='best', framealpha=legend_alpha)
        for l in legend.get_lines():
            l.set_alpha(0.7)
        self.axis.set_title(self.get_title())
        return self.figure

    def save_plot(self, path):
        self.figure.savefig(fname=path, dpi=500, quality=95, format='png')
