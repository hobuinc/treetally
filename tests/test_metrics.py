import numpy as np
import pandas as pd
from math import ceil

from silvimetric import shatter, Storage, Metric
from silvimetric import all_metrics as s
from silvimetric import l_moments
from silvimetric.resources.attribute import Attribute
from silvimetric.resources.taskgraph import Graph

class TestMetrics():

    def test_dag(self, metric_data, metric_dag_results):
        metrics = list(l_moments.values())
        g = Graph(metrics).init()
        assert g.initialized
        assert all(n.initialized for n in g.nodes.values())
        assert list(g.nodes.keys()) == ['l1', 'l2', 'l3', 'l4', 'lcv', 'lskewness', 'lkurtosis', 'lmombase']
        g.run(metric_data)
        for node in g.nodes.values():
            assert isinstance(node.results, pd.DataFrame)
            assert bool(all(node.results.any()))


        assert all(g.results == metric_dag_results)


    def test_metrics(self, metric_data, metric_data_results):
        metrics = list(s.values())
        graph = Graph(metrics)
        metrics = graph.run(metric_data)
        assert isinstance(metrics, pd.DataFrame)
        adjusted = [k.split('_')[-1] for k in metrics.keys()]

        assert all(a in s.keys() for a in adjusted)
        assert all(metrics == metric_data_results)

    def test_dependencies(self, metric_data):
        # should be able to create a dependency graph
        cv = s['cv']
        mean = s['mean']
        stddev = s['stddev']
        median = s['median']

        mean.dependencies = [ median ]
        stddev.dependencies = [ median ]
        cv.dependencies = [ mean, stddev ]

        b = Graph([cv, mean]).run(metric_data)
        # cv/mean should be there
        assert b.m_Z_cv.any()
        assert b.m_Z_mean.any()

        #and median/stddev should not
        assert not any(x in b.dtypes for x in  ['m_Z_median', 'm_Z_stddev'])

    def test_filter(self, metric_shatter_config, test_point_count,
            resolution, alignment):

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
            xysize = 10 if alignment == 'pixelisarea' else 11
            maxy = s.config.root.maxy

            assert a[:,:]['Z'].shape[0] == xysize ** 2
            xdom = a.schema.domain.dim('X').domain[1]
            ydom = a.schema.domain.dim('Y').domain[1]
            assert xdom == xysize
            assert ydom == xysize

            data = a.query(attrs=['Z'], coords=True, use_arrow=False).df[:]
            data = data.set_index(['X','Y'])

            for xi in range(xdom):
                for yi in range(ydom):
                    curr = data.loc[xi,yi]
                    curr.size == 1
                    curr.iloc[0].size == 900
                    # this should have all indices from 0 to 9 filled.
                    # if oob error, it's probably not this test's fault
                    assert bool(np.all( curr.iloc[0] == (ceil(maxy/resolution) - yi - 1) ))

    def test_custom(self, metric_data: pd.DataFrame, attrs: list[Attribute]) -> None:
        def m_over500(data):
            return data[data >= 500].count()
        z_att = attrs[0]
        m_cust = Metric(name='over500', dtype=np.float32, method=m_over500,
            attributes=[z_att])

        b = Graph(m_cust).init().run(metric_data)

        assert b.m_Z_over500.any()
        assert b.m_Z_over500.values[0] == 3

    def test_dependency_passing(self, dep_crr, depless_crr, metric_data):
        nd1 = Graph(depless_crr).init().run(metric_data)
        nd2 = Graph(dep_crr).init().run(metric_data)
        assert all(nd2.m_Z_deps_crr == nd1.m_Z_depless_crr)