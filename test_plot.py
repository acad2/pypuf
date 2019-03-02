from pypuf.plots import AccuracyPlotter


estimators = [None, ('quantile', 0.2), ('quantile', 0.5), ('quantile', 0.8), ('success', 0.8), ('success', 0.95)]

for e in estimators:

    p = AccuracyPlotter(estimator=e, group_by='transformation', grid=True)
    p.get_data_frame(source='studies/logs/test_plot_2_64.log')
    p.create_plot()
    p.save_plot(path='studies/figures/plot_{0}.png'.format('{0}_{1}'.format(e[0], e[1]) if e is not None else 'mean'))
