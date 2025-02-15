import pytest
import os
import pdal
import copy

from datetime import datetime
from typing_extensions import Generator

from silvimetric import Extents, Bounds, Attribute, Storage
from silvimetric import all_metrics
from silvimetric import Log, Metric, ShatterConfig, StorageConfig
from silvimetric import ApplicationConfig, ExtractConfig
from silvimetric import __version__ as svversion

# pull together fixtures
pytest_plugins=[
    'fixtures.shatter_fixtures', 'fixtures.extract_fixtures',
    'fixtures.command_fixtures', 'fixtures.chunk_fixtures',
    'fixtures.western_fixtures', 'fixtures.data_fixtures',
    'fixtures.cli_fixtures', 'fixtures.fusion_fixtures',
    'fixtures.metric_fixtures', 'fixtures.dask_fixtures'
]

@pytest.fixture(scope='function')
def tdb_filepath(storage_config) -> Generator[str, None, None]:
    yield storage_config.tdb_dir

@pytest.fixture(scope='function')
def app_config(tdb_filepath) -> Generator[ApplicationConfig, None, None]:
    log = Log(20) # INFO
    app = ApplicationConfig(tdb_dir = tdb_filepath,
                            log = log)
    yield app

@pytest.fixture(scope='function')
def storage_config(tmp_path_factory, bounds, resolution, crs, attrs, metrics, alignment) -> Generator[StorageConfig, None, None]:
    path = tmp_path_factory.mktemp("test_tdb")
    p = os.path.abspath(path)
    log = Log('INFO')

    sc =  StorageConfig(tdb_dir = p,
                        log = log,
                        crs = crs,
                        root = bounds,
                        resolution = resolution,
                        attrs = attrs,
                        metrics = metrics,
                        alignment = alignment,
                        version = svversion)
    Storage.create(sc)
    yield sc

@pytest.fixture(scope='function')
def storage(storage_config):
    yield Storage.from_db(storage_config.tdb_dir)

@pytest.fixture(scope='function')
def shatter_config(copc_filepath, storage_config, bounds, date) -> Generator[ShatterConfig, None, None]:
    log = Log('INFO') # INFO
    s = ShatterConfig(tdb_dir = storage_config.tdb_dir,
                      log = log,
                      filename = copc_filepath,
                      attrs = storage_config.attrs,
                      metrics = storage_config.metrics,
                      bounds = bounds,
                      date = date, tile_size=10)

    yield s

@pytest.fixture(scope='function')
def extract_config(tif_filepath, metrics, shatter_config, extract_attrs):
    from silvimetric.commands import shatter
    tdb_dir = shatter_config.tdb_dir
    shatter.shatter(shatter_config)
    log = Log(20)
    c =  ExtractConfig(tdb_dir = tdb_dir,
                       log = log,
                       out_dir = tif_filepath,
                       attrs = extract_attrs,
                       metrics = metrics)
    yield c


@pytest.fixture(scope='function')
def metrics() -> Generator[list[Metric], None, None]:
    yield [copy.deepcopy(all_metrics['mean']), copy.deepcopy(all_metrics['median'])]

@pytest.fixture(scope='function')
def bounds(minx, maxx, miny, maxy) -> Generator[Bounds, None, None]:
    b =  Bounds(minx, miny, maxx, maxy)
    yield b

@pytest.fixture(scope='function')
def extents(resolution, alignment, bounds) -> Generator[Extents, None, None]:
    yield Extents(bounds,resolution,alignment,bounds)

@pytest.fixture(scope="function")
def attrs(dims) -> Generator[list[Attribute], None, None]:
    yield [Attribute(a, dims[a]) for a in
           ['Z', 'NumberOfReturns', 'ReturnNumber', 'Intensity']]

@pytest.fixture(scope="session")
def dims() -> Generator[dict, None, None]:
    yield { d['name']: d['dtype'] for d in pdal.dimensions }

@pytest.fixture(scope='session')
def resolution() -> Generator[int, None, None]:
    yield 30

@pytest.fixture(scope='session', params=['pixelisarea','pixelispoint'])
def alignment(request) -> Generator[int, None, None]:
    yield request.param

@pytest.fixture(scope='session')
def test_point_count(alignment) -> Generator[int, None, None]:
    if alignment == 'pixelisarea':
        yield 90000
    else: #pixelispoint
        yield 108900

@pytest.fixture(scope='session')
def minx() -> Generator[float, None, None]:
    yield 300

@pytest.fixture(scope='session')
def miny() -> Generator[float, None, None]:
    yield 300

@pytest.fixture(scope='session')
def maxx() -> Generator[float, None, None]:
    yield 600

@pytest.fixture(scope='session')
def maxy() -> Generator[float, None, None]:
    yield 600

@pytest.fixture(scope='session')
def crs() -> Generator[str, None, None]:
    yield "EPSG:5070"

@pytest.fixture(scope='session')
def date() -> Generator[datetime, None, None]:
    yield datetime(2011, 1, 1)