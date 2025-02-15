import os
import pytest

from typing_extensions import Generator, List

from silvimetric import Extents, Data

@pytest.fixture(scope='function', autouse=True)
def filtered(copc_data, extents: Extents) -> Generator[List[Extents], None, None]:
    yield list(extents.chunk(copc_data, 1))

@pytest.fixture(scope='function')
def unfiltered(extents: Extents) -> Generator[List[Extents], None, None]:
    yield list(extents.get_leaf_children(30))

@pytest.fixture(scope='session')
def copc_filepath(alignment) -> Generator[str, None, None]:
    if alignment == 'pixelispoint':
        path = os.path.join(os.path.dirname(__file__), "..", "data",
                "test_data_pixel_point.copc.laz")
        assert os.path.exists(path)
        yield os.path.abspath(path)
    elif alignment == 'pixelisarea':
        path = os.path.join(os.path.dirname(__file__), "..", "data",
                "test_data_pixel_area.copc.laz")
        assert os.path.exists(path)
        yield os.path.abspath(path)

@pytest.fixture(scope='function')
def copc_data(copc_filepath, storage_config) -> Generator[Data, None, None]:
    d = Data(copc_filepath, storage_config)
    yield d