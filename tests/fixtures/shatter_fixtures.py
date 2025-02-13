import pytest
from typing_extensions import Generator
from uuid import uuid4
import os

from silvimetric import __version__ as svversion
from silvimetric import StorageConfig, ShatterConfig, Storage, Log, Bounds

@pytest.fixture(scope="function")
def s3_bucket() -> Generator[str, None, None]:
    yield "silvimetric"

@pytest.fixture(scope='function')
def s3_uri(s3_bucket) -> Generator[str, None, None]:
    uuid = uuid4()
    yield f"s3://{s3_bucket}/test_silvimetric/{uuid}"

@pytest.fixture(scope="function")
def s3_storage_config(s3_uri, bounds, resolution, alignment, crs, attrs, metrics) -> Generator[StorageConfig, None, None]:
    yield StorageConfig(bounds, crs, resolution, alignment, attrs, metrics,
                        svversion, tdb_dir=s3_uri)

@pytest.fixture(scope='function')
def s3_storage(s3_storage_config) -> Generator[Storage, None, None]:
    import subprocess
    yield Storage.create(s3_storage_config)
    subprocess.call(["aws", "s3", "rm", "--recursive", s3_storage_config.tdb_dir])

@pytest.fixture(scope="function")
def s3_shatter_config(s3_storage, copc_filepath, attrs, metrics, date) -> Generator[ShatterConfig, None, None]:
    config = s3_storage.config
    yield ShatterConfig(filename=copc_filepath, attrs=attrs, metrics=metrics,
                        debug=True, tdb_dir=config.tdb_dir, date=date)

@pytest.fixture(scope='function')
def uneven_storage_config(tmp_path_factory, bounds, crs, attrs, metrics) -> Generator[StorageConfig, None, None]:
    log = Log('INFO')
    path = tmp_path_factory.mktemp("test_tdb")
    p = os.path.abspath(path)

    log = Log('DEBUG')
    sc = StorageConfig(tdb_dir = p,
                        log = log,
                        crs = crs,
                        root = bounds,
                        resolution = 7,
                        attrs = attrs,
                        metrics = metrics,
                        version = svversion)
    Storage.create(sc)
    yield sc

@pytest.fixture(scope='function')
def uneven_shatter_config(copc_filepath, uneven_storage_config, date) -> Generator[ShatterConfig, None, None]:
    tdb_dir = uneven_storage_config.tdb_dir
    log = uneven_storage_config.log

    s = ShatterConfig(tdb_dir = tdb_dir,
                      log = log,
                      filename = copc_filepath,
                      attrs = uneven_storage_config.attrs,
                      metrics = uneven_storage_config.metrics,
                      debug = True,
                      date=date)
    yield s

@pytest.fixture(scope='function')
def partial_storage_config(tmp_path_factory, crs, attrs, metrics) -> Generator[StorageConfig, None, None]:
    path = tmp_path_factory.mktemp("test_tdb")
    p = os.path.abspath(path)
    log = Log('DEBUG')

    bounds = Bounds(300,300,450,450)
    sc = StorageConfig(tdb_dir = p,
                        log = log,
                        crs = crs,
                        root = bounds,
                        resolution = 30,
                        attrs = attrs,
                        metrics = metrics,
                        version = svversion)
    Storage.create(sc)
    yield sc

@pytest.fixture(scope='function')
def partial_shatter_config(copc_filepath, date, partial_storage_config) -> Generator[ShatterConfig, None, None]:
    tdb_dir = partial_storage_config.tdb_dir
    psc: StorageConfig = partial_storage_config
    log = Log('INFO') # INFO
    yield ShatterConfig(tdb_dir=tdb_dir,
                      log=log,
                      attrs=psc.attrs,
                      metrics=psc.metrics,
                      filename=copc_filepath,
                      debug=True,
                      date=date)