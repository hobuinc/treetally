import tiledb
import numpy as np
import pytest
import os
import copy

from silvimetric import Storage, all_metrics, Attribute, Attributes, StorageConfig, Log
from silvimetric import __version__ as svversion

class Test_Storage(object):

    def test_schema(self, storage: Storage, attrs: list[Attribute]):
        with storage.open('r') as st:
            s:tiledb.ArraySchema = st.schema
            assert s.has_attr('count')
            assert s.attr('count').dtype == np.int32

            for a in attrs:
                assert s.has_attr(a.name)
                # assert s.attr(a.name) == a.schema()

    def test_time_reserve(self, storage):
        for x in range(5):
            time_slot = storage.reserve_time_slot()
            assert time_slot == x + 1

    def test_local(self, storage: Storage, attrs: list[Attribute]):
        with storage.open('r') as st:
            sc = st.schema
            assert sc.has_attr('count')
            assert sc.attr('count').dtype == np.int32

            for a in attrs:
                assert sc.has_attr(a.name)
                # assert sc.attr(a.name) == a.schema()

    def test_config(self, storage: Storage):
        """Check that instantiation metadata is properly written"""

        storage.saveConfig()
        config = storage.getConfig()
        assert config.resolution == storage.config.resolution
        assert config.alignment == storage.config.alignment
        assert config.root == storage.config.root
        assert config.crs == storage.config.crs
        assert storage.config.version == svversion

    def test_metric_dependencies(self, tmp_path_factory, metrics, crs, resolution, alignment, attrs, bounds):
        ms = copy.deepcopy(metrics)

        path = tmp_path_factory.mktemp("test_tdb")
        p = os.path.abspath(path)
        log = Log('DEBUG')

        ms[0].dependencies = [Attributes['HeightAboveGround']]
        sc = StorageConfig(tdb_dir=p, crs=crs, resolution=resolution,
                alignment = alignment, attrs=attrs, metrics=ms, root=bounds)

        with pytest.raises(ValueError) as e:
            Storage.create(sc)
        assert str(e.value) == 'Missing required dependency, HeightAboveGround.'

        ms[0].dependencies = [Attributes['NumberOfReturns']]
        s = Storage.create(sc)
        assert isinstance(s, Storage)

        ms[0].dependencies = []

    def test_metrics(self, storage: Storage):
        m_list = storage.getMetrics()
        a_list = storage.getAttributes()

        with storage.open('r') as st:
            s: tiledb.ArraySchema = st.schema
            for m in m_list:
                assert m.name in all_metrics.keys()
                def e_name(att):
                    return s.attr(m.entry_name(att.name))
                def schema(att):
                    return all_metrics[m.name].schema(att)
                assert all([e_name(a) == schema(a) for a in a_list])
