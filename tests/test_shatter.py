import os
import pytest
import numpy as np
import dask
import json
import uuid

from silvimetric import Extents, Log, info, shatter, Bounds, Storage
from silvimetric import StorageConfig, ShatterConfig


@dask.delayed
def write(x, y, val, s:Storage, attrs, dims, metrics):
    m_list = [m.entry_name(a.name) for m in metrics for a in attrs]
    data = { a.name: np.array([np.array([val], dims[a.name]), None], object)[:-1]
                for a in attrs }

    for m in m_list:
        data[m] = [val]

    data['count'] = [val]
    with s.open('w') as w:
        w[x,y] = data

def confirm_storage(storage, timestamp, maxy):
    with storage.open('r', timestamp=timestamp) as a:
        assert a[:,:]['Z'].size == 100
        xdom = a.schema.domain.dim('X').domain[1]
        ydom = a.schema.domain.dim('Y').domain[1]
        assert xdom == 9
        assert ydom == 9

        for xi in range(xdom):
            for yi in range(ydom):
                a[xi, yi]['Z'].size == 1
                a[xi, yi]['Z'].item().size == 900
                # this should have all indices from 0 to 9 filled.
                # if oob error, it's not this test's fault
                assert bool(np.all( a[xi, yi]['Z'].item() == ((maxy/storage.config.resolution) - (yi + 1)) ))

class Test_Shatter(object):

    def test_command(self, shatter_config, storage: Storage, maxy):
        shatter(shatter_config)
        assert storage.config.next_time_slot == 1
        timestamp = (0,0)
        confirm_storage(storage, timestamp, maxy)

    def test_multiple(self, shatter_config, storage: Storage, maxy):
        shatter(shatter_config)
        with storage.open('r') as a:
            assert a[:,:]['Z'].size == 100
            xdom = a.schema.domain.dim('X').domain[1]
            ydom = a.schema.domain.dim('Y').domain[1]
            assert xdom == 9
            assert ydom == 9

            for xi in range(xdom):
                for yi in range(ydom):
                    a[xi, yi]['Z'].size == 1
                    a[xi, yi]['Z'].item().size == 900
                    # this should have all indices from 0 to 9 filled.
                    # if oob error, it's not this test's fault
                    assert bool(np.all( a[xi, yi]['Z'][0] == ((maxy/storage.config.resolution) - (yi + 1)) ))

        # change attributes to make it a new run
        shatter_config.name = uuid.uuid4()
        shatter_config.mbr = ()
        shatter_config.time_slot = 2

        shatter(shatter_config)
        with storage.open('r') as a:
            # querying flattens to 20, there will 10 pairs of values
            assert a[:,:]['Z'].shape[0] == 200
            assert a[:,0]['Z'].shape[0] == 20
            assert a[0,:]['Z'].shape[0] == 20
            # now test that the second set is the same as the first set
            # and test that this value is still the same as the original
            # which was set at ((maxy/resolution)-yindex)
            for xi in range(xdom):
                for yi in range(ydom):
                    a[xi, yi]['Z'].size == 2
                    a[xi, yi]['Z'][0].size == 900
                    a[xi, yi]['Z'][1].size == 900
                    assert bool(np.all(a[xi, yi]['Z'][1] == a[xi,yi]['Z'][0]))
                    assert bool(np.all(a[xi, yi]['Z'][1] == ((maxy/storage.config.resolution) - (yi + 1))))

        m = info(storage.config.tdb_dir)
        assert len(m['history']) == 2

    def test_parallel(self, storage, attrs, dims, threaded_dask, metrics):
        # test that writing in parallel doesn't affect ordering of values
        # constrained by NumberOfReturns being uint8

        count = 255
        tl = [write(0,0,val,storage,attrs,dims, metrics) for val in range(count)]

        dask.compute(tl)

        with storage.open('r') as r:
            d = r[0,0]
            for idx in range(count):
                assert bool(np.all(d['Z'][idx] == d['Intensity'][idx]))
                assert bool(np.all(d['Intensity'][idx] == d['NumberOfReturns'][idx]))
                assert bool(np.all(d['NumberOfReturns'][idx] == d['ReturnNumber'][idx]))

    def test_config(self, shatter_config, storage, test_point_count):
        shatter(shatter_config)
        try:
            meta = storage.getMetadata('shatter', shatter_config.time_slot)
        except BaseException as e:
            pytest.fail("Failed to retrieve 'shatter' metadata key." + e.args)
        meta_j = json.loads(meta)
        pc = meta_j['point_count']
        assert pc == test_point_count

    @pytest.mark.parametrize('sh_cfg', ['shatter_config', 'uneven_shatter_config'])
    def test_sub_bounds(self, sh_cfg, test_point_count, request, maxy, resolution):
        s = request.getfixturevalue(sh_cfg)
        storage = Storage.from_db(s.tdb_dir)
        e = Extents.from_storage(s.tdb_dir)

        pc = 0
        for b in e.split():
            log = Log(20)
            time_slot = storage.reserve_time_slot()
            sc = ShatterConfig(tdb_dir = s.tdb_dir,
                               log = log,
                               filename = s.filename,
                               tile_size = s.tile_size,
                               attrs = s.attrs,
                               metrics = s.metrics,
                               debug = s.debug,
                               bounds = b.bounds,
                               date=s.date,
                               time_slot=time_slot)
            pc = pc + shatter(sc)
        history = info(s.tdb_dir)['history']
        assert len(history) == 4
        assert isinstance(history, list)
        pcs = [ h['point_count'] for h in history ]
        assert sum(pcs) == test_point_count
        assert pc == test_point_count

        with storage.open('r') as a:
            xdom = a.schema.domain.dim('X').domain[1]
            ydom = a.schema.domain.dim('Y').domain[1]

            data = a.query(attrs=['Z'], coords=True, use_arrow=False).df[:]
            data = data.set_index(['X','Y'])

            for xi in range(xdom):
                for yi in range(ydom):
                    curr = data.loc[xi,yi]
                    # check that each cell only has one allocation
                    curr.size == 1

    def test_partial_overlap(self, partial_shatter_config, test_point_count):
        pc = shatter(partial_shatter_config)
        assert pc == test_point_count / 4

    @pytest.mark.skipif(
        os.environ.get('AWS_SECRET_ACCESS_KEY') is None or
        os.environ.get('AWS_ACCESS_KEY_ID') is None,
        reason='Missing necessary AWS environment variables'
    )
    def test_remote_creation(self, s3_shatter_config, s3_storage):
        # need processes scheduler to accurately test bug fix
        dask.config.set(scheduler="processes")
        resolution = s3_storage.config.resolution
        maxy = s3_storage.config.root.maxy
        shatter(s3_shatter_config)
        with s3_storage.open('r') as a:
            assert a[:,:]['Z'].shape[0] == 100
            xdom = a.schema.domain.dim('X').domain[1]
            ydom = a.schema.domain.dim('Y').domain[1]
            assert xdom == 10
            assert ydom == 10
            # get all data local so we're not hittin s3 all the time
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