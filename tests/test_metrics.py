import numpy as np
import pandas as pd

from silvimetric import shatter, Storage, grid_metrics, MetricGraph
from silvimetric import all_metrics as s
from silvimetric import l_moments

class TestMetrics():

    def test_metrics(self, metric_data):
        ms: pd.DataFrame = MetricGraph.run_metrics(metric_data, list(grid_metrics.values()))
        assert isinstance(ms, pd.DataFrame)
        adjusted = [k.split('_')[-1] for k in ms.keys()]

        assert all(a in grid_metrics.keys() for a in adjusted)


    def test_intermediate_metric(self, metric_data):
        ms = list(l_moments.values())
        computed = MetricGraph.run_metrics(metric_data, ms)

        assert computed.m_Z_l1.any()
        assert computed.m_Z_l2.any()
        assert computed.m_Z_l3.any()
        assert computed.m_Z_l4.any()
        assert computed.m_Z_lcv.any()
        assert computed.m_Z_lskewness.any()
        assert computed.m_Z_lkurtosis.any()

    def test_dependencies(self, metric_data):
        # should be able to create a dependency graph, use dask.get to retrieve
        # the necessary keys, and those values should be dataframes.
        import dask

        cv = s['cv']
        mean = s['mean']
        stddev = s['stddev']
        median = s['median']

        mean.dependencies = [ median ]
        stddev.dependencies = [ median ]
        cv.dependencies = [ mean, stddev ]

        # graph: MetricGraph = MetricGraph.make_graph(cv)
        a = MetricGraph.get_methods(metric_data, [cv, mean])
        b = dask.compute(*a)

        assert b[0].m_Z_cv.any()
        assert b[1].m_Z_mean.any()

    def test_filter(self, metric_shatter_config, test_point_count, maxy,
            resolution):

        m = metric_shatter_config.metrics[0]
        assert len(m.filters) == 1

        pc = shatter(metric_shatter_config)
        assert pc == test_point_count

        s = Storage.from_db(metric_shatter_config.tdb_dir)
        with s.open('r') as a:
            q = a.query(coords=False, use_arrow=False).df

            nor_mean = q[:]['m_NumberOfReturns_mean']
            nor = q[:]['NumberOfReturns']
            assert not nor_mean.isna().any()
            assert nor.notna().any()

            assert a[:,:]['Z'].shape[0] == 100
            xdom = a.schema.domain.dim('X').domain[1]
            ydom = a.schema.domain.dim('Y').domain[1]
            assert xdom == 10
            assert ydom == 10

            data = a.query(attrs=['Z'], coords=True, use_arrow=False).df[:]
            data = data.set_index(['X','Y'])

            for xi in range(xdom):
                for yi in range(ydom):
                    curr = data.loc[xi,yi]
                    curr.size == 1
                    curr.iloc[0].size == 900
                    # this should have all indices from 0 to 9 filled.
                    # if oob error, it's not this test's fault
                    assert bool(np.all( curr.iloc[0] == ((maxy/resolution) - (yi + 1)) ))

