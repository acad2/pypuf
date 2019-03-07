"""
Index of the first successful permutation for the correlation attack.
"""
from pypuf.experiments.experiment.correlation_attack import ExperimentCorrelationAttack
from pypuf.experiments.experiment.correlation_attack import Parameters as CorrelationAttackParameters
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experiment.logistic_regression import Parameters as LogisticRegressionParameters

from pypuf.plots import PermutationIndexPlot
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.studies.base import Study

from math import ceil
from datetime import timedelta


def time_to_string(delta):
    if delta.total_seconds() > 4 * 24 * 60 ** 2:
        return '%i days' % round(delta.days)
    else:
        hours = delta.seconds // 3600
        if 24 * delta.days + hours == 0:
            minutes = (delta.seconds - (3600 * hours)) // 60
            seconds = (delta.seconds - (3600 * hours) - (60 * minutes)) // 1
            return '%02im %02is' % (minutes, seconds)
        else:
            minutes = round((delta.seconds - (3600 * hours)) / 60)
            return '%2ih %02im' % (24 * delta.days + hours, minutes)


def round_time(seconds):
    units = [
        4 * 24 * 60 ** 2,  # 4 days
        1,  # second
    ]
    for u in units:
        if seconds > u:
            return seconds // u * u
    return seconds


NICE_ATTACK_NAMES = {'lr': 'LR', 'corr': 'Correlation Attack'}
NICE_TRANSFORM_NAMES = {
    LTFArray.transform_atf.__name__: "Classic",
    LTFArray.transform_lightweight_secure.__name__: "Lightweight Secure",
    LTFArray.transform_fixed_permutation.__name__: "Permutation-Based"
}


class BreakingLightweightSecureRuntimeTableFig05(Study):
    SAMPLES_PER_ENTRY = 1000

    KS = [4, 4, 5, 6]
    TRAINING_SET_SIZES = [12000, 30000, 300000, 1000000]
    TRANSFORMATIONS = [LTFArray.transform_atf, LTFArray.transform_lightweight_secure,
                       LTFArray.transform_fixed_permutation]

    def __init__(self):
        super().__init__()
        self.result_plot = None
        self.success_threshold = .98
        self.optimization_threshold = .65
        self.corr_experiment_hashes = []
        self.study_experiment_hashes = []

    def get_success_rate(self, results):
        return

    def gen_table(self, results):
        """
        Generate runtime/success comparison from data
        """
        columns = [
            ('lr', LTFArray.transform_atf.__name__),
            ('lr', LTFArray.transform_lightweight_secure.__name__),
            ('corr', LTFArray.transform_lightweight_secure.__name__),
            ('lr', LTFArray.transform_fixed_permutation.__name__),
            # ('lr', 'random'),
        ]
        latex = r'\begin{tabular}{rrr' + len(columns) * 'rc' + '}\n  \\toprule\n'
        latex += r'  &&&'
        latex += '&\t\t'.join(
            [r'\multicolumn{2}{c}{%s on}' % NICE_ATTACK_NAMES[attack] for (attack, transform) in columns])
        latex += r'\\'
        latex += '\n'
        latex += r'  $n$ & $k$ & \# CRPs &' + '\t\t'
        latex += '&\t\t'.join(
            [r'\multicolumn{2}{c}{%s}' % NICE_TRANSFORM_NAMES[transform] for (attack, transform) in columns])
        latex += r'\\'
        latex += '\n  \\midrule\n'
        for (n, k, crps) in [
            (64, 4, 12000),
            (64, 4, 30000),
            (64, 5, 300000),
            (64, 6, 1000000),
            #    (128, 4, 500000),
            #    (128, 5, 1000000),
        ]:
            latex += '  {:d}&{:d}&{:,d}&\t\t'.format(n, k, crps)
            column_latex = []
            for (attack, transformation) in columns:
                sel = results.loc[(results['n'] == n) & (results['k'] == k) & (results['N'] == crps)]
                if attack == 'corr':
                    sel = sel.loc[sel['experiment_hash'].isin(self.corr_experiment_hashes)]
                else:
                    sel = sel.loc[sel['transformation'] == transformation]
                if sel.empty:
                    continue
                successful_runs = sel.loc[sel['accuracy'] > self.success_threshold]
                failed_runs = sel.loc[sel['accuracy'] <= self.success_threshold]
                success_rate = len(successful_runs) / len(sel)
                avg_wall_time = sel['measured_time'].mean()
                avg_no_success_wall_time = failed_runs['measured_time'].mean() if not failed_runs.empty else 0

                if success_rate > 0:
                    avg_success_wall_time = successful_runs['measured_time'].mean()
                    expected_tries_till_success = ceil(1 / success_rate)
                    expected_time_till_success = (expected_tries_till_success - 1) * avg_no_success_wall_time \
                                                 + avg_success_wall_time
                else:
                    expected_time_till_success = avg_wall_time * len(sel)

                marker = r'${}^{%d}$' % len(sel)

                if success_rate > 0:
                    column_latex.append(r'&{}'.format(
                        time_to_string(timedelta(seconds=round_time(expected_time_till_success)))) + marker)
                else:
                    column_latex.append(r'&longer than %s' % (
                        time_to_string(timedelta(seconds=round_time(expected_time_till_success)))) + marker)
            latex += '&\t\t'.join(column_latex)
            latex += r'\\' + '\n'

        latex += '  \\bottomrule\n'
        latex += r'\end{tabular}'

        runtime_table = open('figures/runtime-table.tex', 'w')
        runtime_table.write(latex)
        runtime_table.close()

    def name(self):
        return 'breaking_lightweight_secure_runtime_table+fig_05'

    def plot(self):
        if self.experimenter.results.empty:
            return

        results = self.experimenter.results
        results = results.loc[results['experiment_hash'].isin(self.study_experiment_hashes)]

        self.gen_table(results)
        self.result_plot.plot(results)

    def experiments(self):
        experiments = []

        for i in range(self.SAMPLES_PER_ENTRY):
            for idx, training_set_size in enumerate(self.TRAINING_SET_SIZES):
                ex = ExperimentCorrelationAttack(
                    progress_log_prefix=self.name(),
                    parameters=CorrelationAttackParameters(
                        n=64,
                        k=self.KS[idx],
                        N=training_set_size,
                        seed_instance=314159 + i,
                        seed_model=265358 + i,
                        seed_challenge=979323 + i,
                        seed_distance=846264 + i,
                        lr_iteration_limit=1000,
                        mini_batch_size=0,
                        convergence_decimals=2,
                        shuffle=False
                    )
                )
                experiments.append(ex)
                self.corr_experiment_hashes.append(ex.hash)
                self.study_experiment_hashes.append(ex.hash)
                for transform in self.TRANSFORMATIONS:
                    ex = ExperimentLogisticRegression(
                        progress_log_prefix=self.name(),
                        parameters=LogisticRegressionParameters(
                            n=64,
                            k=self.KS[idx],
                            N=training_set_size,
                            seed_instance=314159 + i,
                            seed_model=265358 + i,
                            seed_challenge=979323 + i,
                            seed_distance=846264 + i,
                            transformation=transform.__name__,
                            combiner=LTFArray.combiner_xor.__name__,
                            mini_batch_size=0,
                            convergence_decimals=2,
                            shuffle=False
                        )
                    )
                    experiments.append(ex)
                    self.study_experiment_hashes.append(ex.hash)

        self.result_plot = PermutationIndexPlot(
            filename='figures/breaking_lightweight_secure_fig_05.pdf',
            group_by='N',
            experiment_hashes=self.corr_experiment_hashes,
            group_labels={
                self.TRAINING_SET_SIZES[i]:
                    "Perm. Index Dist. (k={}, {:,} CRPs)".format(self.KS[i], self.TRAINING_SET_SIZES[i])
                for i in range(len(self.TRAINING_SET_SIZES))
            },
            group_subplot_layout={
                self.TRAINING_SET_SIZES[0]: (4, 2, 1),
                self.TRAINING_SET_SIZES[1]: (4, 2, 2),
                self.TRAINING_SET_SIZES[2]: (4, 1, 2),
                self.TRAINING_SET_SIZES[3]: (4, 1, 3),
            }, w=3.34, h=3.9
        )

        return experiments
