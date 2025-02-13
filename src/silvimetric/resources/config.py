import pyproj

import json
import uuid
import numpy as np

from pathlib import Path
from abc import ABC, abstractmethod
from typing_extensions import Union, Tuple
from datetime import datetime

from dataclasses import dataclass, field

from .log import Log
from .extents import Bounds
from .metric import Metric
from .metrics import grid_metrics
from .attribute import Attribute, Attributes
from .. import __version__


class SilviMetricJSONEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__

@dataclass(kw_only=True)
class Config(ABC):
    """Base config"""
    tdb_dir: str
    """Path to TileDB directory to use."""
    log: Log = field(default_factory = lambda: Log("INFO"))
    """Log object."""
    debug: bool = field(default=False)
    """Debug flag."""

    def to_json(self):
        keys = self.__dataclass_fields__.keys()
        d = {}
        for k in keys:
            d[k] = self.__dict__[k]
        if not isinstance(d['log'], dict):
            d['log'] = d['log'].to_json()
        return d

    @classmethod
    def from_json(self, data: str):
        return self.from_string(json.dumps(data))

    @classmethod
    @abstractmethod
    def from_string(self, data: str):
        raise NotImplementedError

    def __repr__(self):
        return json.dumps(self.to_json())




@dataclass
class StorageConfig(Config):
    """ Config for constructing a Storage object """
    root: Bounds
    """Root project bounding box"""
    crs: pyproj.CRS
    """Coordinate reference system, same for all data in a project"""
    resolution: float = 30.0
    """Resolution of cells, same for all data in a project, defaults to 30.0"""
    alignment: str = 'PixelIsArea'
    """Alignment of pixels in database, same for all data in a project, options: 'PixelIsArea' or 'PixelIsPoint', defaults to 'PixelIsArea'"""

    attrs: list[Attribute] = field(default_factory=lambda: [
        Attribute(a, Attributes[a].dtype)
        for a in [ 'Z', 'NumberOfReturns', 'ReturnNumber', 'Intensity' ]])
    """List of :class:`silvimetric.resources.attribute.Attribute` attributes that
    represent point data, defaults to Z, NumberOfReturns, ReturnNumber, Intensity"""
    metrics: list[Metric] = field(default_factory=lambda: [ grid_metrics[m]
                                  for m in grid_metrics.keys() ])
    """List of :class:`silvimetric.resources.metrics.grid_metrics` grid_metrics that
    represent derived data, defaults to values in grid_metrics object"""
    version: str = __version__
    """Silvimetric version"""
    capacity: int = 1000000
    """TileDB Capacity, defaults to 1000000"""
    next_time_slot: int = 1
    """Next time slot to be allocated to a shatter process. Increment after
    use., defaults to 1"""

    def __post_init__(self) -> None:

        crs = self.crs
        if isinstance(crs, dict):
            crs = json.loads(crs)
        elif isinstance(crs, pyproj.CRS):
            self.crs = crs
        else:
            self.crs = pyproj.CRS.from_user_input(crs)
        if not len(self.attrs):
            self.attrs = [Attributes[a] for a in [ 'Z', 'NumberOfReturns',
                                            'ReturnNumber', 'Intensity' ]]
        if not len(self.metrics):
            self.metrics = [ grid_metrics[m] for m in grid_metrics.keys() ]

        if not self.crs.is_projected:
            raise Exception("Given coordinate system is not a rectilinear"
                    " projected coordinate system")

        self.metric_definitions = { m.name: str(m) for m in
                self.metrics}

    def __eq__(self, other):

        # We don't compare logs
        eq = True
        for k in other.__dict__.keys():
            if k != 'log':
                if self.__dict__[k] != other.__dict__[k]:
                    return False
        return True

    def to_json(self):
        d = super().to_json()

        d['attrs'] = [a.to_json() for a in self.attrs]
        d['metrics'] = [m.to_json() for m in self.metrics]
        d['crs'] = json.loads(self.crs.to_json())
        d['root'] = self.root.to_json()
        d['next_time_slot'] = self.next_time_slot

        return d

    @classmethod
    def from_string(cls, data: str):
        x = json.loads(data)
        root = Bounds(*x['root'])
        if 'metrics' in x:
            ms = [ Metric.from_dict(m) for m in x['metrics']]
        else:
            ms = [ ]
        if 'attrs' in x:
            attrs = [ Attribute.from_dict(a) for a in x['attrs']]
        else:
            attrs = [ ]
        if 'crs' in x:
            crs = pyproj.CRS.from_user_input(json.dumps(x['crs']))
        else:
            crs = None

        n = cls(tdb_dir = x['tdb_dir'],
                root = root,
                log = Log(**x['log']),
                resolution = x['resolution'],
                alignment = x['alignment'],
                attrs = attrs,
                crs = crs,
                metrics = ms,
                capacity = x['capacity'],
                version = x['version'],
                next_time_slot=x['next_time_slot'])

        return n

    def __repr__(self):
        j = self.to_json()
        return json.dumps(j)

@dataclass
class ApplicationConfig(Config):
    """ Base application config """

    debug: bool = False,
    """Debug mode, defaults to False"""
    progress: bool = False,
    """Should processes display progress bars, defaults to False"""

    # Dask configuration
    dasktype: str = 'processes'
    """Dask parallelization type. For information see
    https://docs.dask.org/en/stable/scheduling.html#local-threads """
    scheduler: str = 'distributed'
    """Dask scheduler, defaults to 'distributed'"""
    workers: int = 12
    """Number of dask workers"""
    threads: int = 4
    """Number of threads per dask worker"""
    watch: bool = False
    """Open dask diagnostic page in default web browser"""

    def to_json(self):
        d = super().to_json()
        return d

    @classmethod
    def from_string(cls, data: str):
        x = json.loads(data)
        n = cls(tdb_dir = x['tdb_dir'],
                debug = x['debug'],
                progress = x['progress'],
                dasktype = x['dasktype'],
                scheduler = x['scheduler'],
                workers = x['workers'],
                threads = x['threads'],
                watch = x['watch'])
        return n

    def __repr__(self):
        return json.dumps(self.to_json())

Mbr = tuple[tuple[tuple[int,int], tuple[int,int]], ...]
@dataclass
class ShatterConfig(Config):
    """Config for Shatter process"""
    filename: str
    """Input filename referencing a PDAL pipeline or point cloud file."""
    date: Union[datetime, Tuple[datetime, datetime]]
    """A date or date range representing data collection times."""
    attrs: list[Attribute] = field(default_factory=list)
    """List of attributes to use in shatter. If this is not set it will be
    filled by the attributes in the database instance."""
    metrics: list[Metric] = field(default_factory=list)
    """A list of metrics to use in shatter. If this is not set it will be filled
    by the metrics in the database instance."""
    bounds: Union[Bounds, None] = field(default=None)
    """The bounding box of the shatter process., defaults to None"""
    name: uuid.UUID = field(default=uuid.uuid4())
    """UUID representing this shatter process and will be generated if not
    provided., defaults to uuid.uuid()"""
    tile_size: Union[int, None] = field(default=None)
    """The number of cells to include in a tile., defaults to None"""
    start_time:float = 0
    """The process starting time in seconds since Jan 1 1970., defaults to 0"""
    end_time:float = 0
    """The process ending time in seconds since Jan 1 1970., defaults to 0"""
    point_count: int = 0
    """The number of points that has been processed so far., defaults to 0"""
    mbr: Mbr = field(default_factory=tuple)
    """The minimum bounding rectangle derived from TileDB array fragments.
    This will be used to for resuming shatter processes and making sure it
    doesn't repeat work., defaults to tuple()"""
    finished: bool = False
    """Finished flag for shatter process., defaults to False"""
    time_slot: int = 0
    """The time slot that has been reserved for this shatter process. Will be
    used as the timestamp in tiledb writes to better organize and manage
    processes., defaults to 0"""

    def __post_init__(self) -> None:
        from .storage import Storage
        s = Storage.from_db(self.tdb_dir)

        if isinstance(self.tile_size, float):
            self.tile_size = int(self.tile_size)
            self.log.warning(f'Truncating tile size to integer({self.tile_size})')
            pass
        if not self.attrs:
            self.attrs = s.getAttributes()
        if not self.metrics:
            self.metrics = s.getMetrics()

        del s

    def history_json(self):
        # removing a attrs and metrics, since they'll be in the storage log
        # removing mbr because it's too big to do json pretty printing with
        # could add a custom json logger to handle mbr logging in the future
        if isinstance(self.date, tuple):
            date = [ dt.strftime('%Y-%m-%dT%H:%M:%SZ') for dt in self.date]
        else:
            date = self.date.strftime('%Y-%m-%dT%H:%M:%SZ')
        d = dict(filename=self.filename, name=str(self.name), time_slot=self.time_slot, bounds=self.bounds.to_json(),
                date=date)

        return d

    def to_json(self):
        d = super().to_json()

        d['name'] = str(self.name)
        d['time_slot'] = self.time_slot
        d['bounds'] = self.bounds.to_json() if self.bounds is not None else None
        d['attrs'] = [a.to_json() for a in self.attrs]
        d['metrics'] = [m.to_json() for m in self.metrics]
        d['mbr'] = list(self.mbr)

        if isinstance(self.date, tuple):
            d['date'] = [ dt.strftime('%Y-%m-%dT%H:%M:%SZ') for dt in self.date]
        else:
            d['date'] = self.date.strftime('%Y-%m-%dT%H:%M:%SZ')
        return d

    @classmethod
    def from_string(cls, data: str):
        x = json.loads(data)
        return cls.from_dict(x)

    @classmethod
    def from_dict(cls, data: dict):
        x = data

        ms = list([ Metric.from_dict(m) for m in x['metrics']])
        attrs = list([ Attribute.from_dict(a) for a in x['attrs']])
        if isinstance(x['date'], list):
            date = tuple(( datetime.strptime(d, '%Y-%m-%dT%H:%M:%SZ') for d in x['date']))
        else:
            date = datetime.strptime(x['date'], '%Y-%m-%dT%H:%M:%SZ')
        mbr = tuple(tuple(tuple(mb) for mb in m) for m in x['mbr'])
        # TODO key error if these aren't there. If we're calling from_string
        # then these keys need to exist.

        n = cls(tdb_dir=x['tdb_dir'],
                filename=x['filename'],
                attrs=attrs,
                metrics=ms,
                debug=x['debug'],
                name=uuid.UUID(x['name']),
                bounds=Bounds(*x['bounds']),
                tile_size=x['tile_size'],
                start_time=x['start_time'],
                end_time=x['end_time'],
                point_count=x['point_count'],
                mbr=mbr,
                date=date,
                time_slot=x['time_slot'],
                finished=x['finished'])

        return n

    def __repr__(self):
        return json.dumps(self.to_json())


@dataclass
class ExtractConfig(Config):
    """Config for the Extract process."""
    out_dir: str
    """The directory where derived rasters should be written."""
    attrs: list[Attribute] = field(default_factory=list)
    """List of attributes to use in shatter. If this is not set it
    will be filled by the attributes in the database instance."""
    metrics: list[Metric] = field(default_factory=list)
    """A list of metrics to use in shatter. If this is not set it
    will be filled by the metrics in the database instance."""
    bounds: Bounds = field(default=None)
    """The bounding box of the shatter process., defaults to None"""

    def __post_init__(self) -> None:
        from .storage import Storage
        config = Storage.from_db(self.tdb_dir).config
        if not len(self.attrs):
            self.attrs = config.attrs
        if not len(self.metrics):
            self.metrics = config.metrics
        if self.bounds is None:
            self.bounds: Bounds = config.root

        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)

        self.resolution: float = config.resolution
        self.crs: pyproj.CRS = config.crs

    def to_json(self):
        d = super().to_json()

        d['attrs'] = [a.to_json() for a in self.attrs]
        d['metrics'] = [m.to_json() for m in self.metrics]
        d['crs'] = json.loads(self.crs.to_json())
        d['bounds'] = self.bounds.to_json()
        return d

    @classmethod
    def from_string(cls, data: str):
        x = json.loads(data)
        if 'metrics' in x:
            ms = [ Metric.from_dict(m) for m in x['metrics']]
        if 'attrs' in x:
            attrs = [ Attribute.from_dict(a) for a in x['attrs']]
        if 'bounds' in x:
            bounds = Bounds(*x['bounds'])
        if 'log' in x:
            l = x['log']
            log = Log(l['log_level'], l['logdir'], l['logtype'], l['logfilename'])
        else:
            log = Log("INFO")
        n = cls(tdb_dir=x['tdb_dir'], out_dir=x['out_dir'], attrs=attrs,
                metrics=ms, debug=x['debug'], bounds=bounds, log=log)

        return n

    def __repr__(self):
        return json.dumps(self.to_json())