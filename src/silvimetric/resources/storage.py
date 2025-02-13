import os
import tiledb
import numpy as np
from datetime import datetime
import pathlib
import contextlib
import json
import urllib
from typing_extensions import Generator

from math import floor

from .config import StorageConfig, ShatterConfig
from .metric import Metric, Attribute
from .bounds import Bounds

class Storage:
    """ Handles storage of shattered data in a TileDB Database. """

    def __init__(self, config: StorageConfig, ctx:tiledb.Ctx=None):

        if not tiledb.object_type(config.tdb_dir) == "array":
            raise Exception(f"Given database directory '{config.tdb_dir}' does not exist")

        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return


    @staticmethod
    def create( config:StorageConfig, ctx:tiledb.Ctx=None):
        """
        Creates TileDB storage.

        Parameters
        ----------
        config : StorageConfig
            Storage StorageConfig
        ctx : tiledb.Ctx, optional
            TileDB Context, by default is None

        Returns
        -------
        Storage
            Returns newly created Storage class

        Raises
        ------
        Exception
            Raises bounding box errors if not of lengths 4 or 6
        """

        # TODO make any changes to tiledb setup here.

        if not ctx:
            ctx = tiledb.default_ctx()

        # adjust cell bounds if necessary
        config.root.adjust_alignment(config.resolution, config.alignment)

        # dims = { d['name']: d['dtype'] for d in pdal.dimensions if d['name'] in config.attrs }
        xi = floor((config.root.maxx - config.root.minx) / float(config.resolution))
        yi = floor((config.root.maxy - config.root.miny) / float(config.resolution))

        dim_row = tiledb.Dim(name="X", domain=(0,xi), dtype=np.int32)
        dim_col = tiledb.Dim(name="Y", domain=(0,yi), dtype=np.int32)
        domain = tiledb.Domain(dim_row, dim_col)

        count_att = tiledb.Attr(name="count", dtype=np.int32)
        dim_atts = [attr.schema() for attr in config.attrs]

        metric_atts = [m.schema(a) for m in config.metrics for a in config.attrs if a in m.attributes or not m.attributes]

        # Check that all attributes required for metric usage are available
        att_list = [a.name for a in config.attrs]
        required_atts = [d.name for m in config.metrics for d in m.dependencies
                if isinstance(d, Attribute)]
        for ra in required_atts:
            if ra not in att_list:
                raise ValueError(f'Missing required dependency, {ra}.')

        # allows_duplicates lets us insert multiple values into each cell,
        # with each value representing a set of values from a shatter process
        # https://docs.tiledb.com/main/how-to/performance/performance-tips/summary-of-factors#allows-duplicates
        schema = tiledb.ArraySchema(domain=domain, sparse=True,
            attrs=[count_att, *dim_atts, *metric_atts], allows_duplicates=True,
            capacity=1000)
        schema.check()

        tiledb.SparseArray.create(config.tdb_dir, schema)
        with tiledb.SparseArray(config.tdb_dir, "w") as a:
            meta = str(config)
            a.meta['config'] = meta

        s = Storage(config, ctx)
        s.saveConfig()

        return s

    @staticmethod
    def from_db(tdb_dir: str):
        """
        Create Storage object from information stored in a database.

        :param tdb_dir: TileDB database directory.
        :return: Returns the derived storage.
        """
        with tiledb.open(tdb_dir, 'r') as a:
            s = a.meta['config']
            config = StorageConfig.from_string(s)

        return Storage(config)

    def saveConfig(self) -> None:
        """
        Save StorageConfig to the Database
        """
        with self.open('w') as w:
            w.meta['config'] = str(self.config)

    def getConfig(self) -> StorageConfig:
        """
        Get the StorageConfig currently in use by the Storage.

        :return: StorageConfig representing this object.
        """
        with self.open('r') as a:
            s = a.meta['config']
            config = StorageConfig.from_string(s)

        return config

    def getMetadata(self, key: str, timestamp: int) -> str:
        """
        Return metadata at given key.

        :param key: Key to look for in metadata.
        :param timestamp: Time stamp for querying database.
        :return: Metadata value found in storage.
        """

        with self.open('r', (timestamp, timestamp)) as r:
            val = r.meta[f'{key}']

        return val


    def saveMetadata(self, key: str, data: str, timestamp: int) -> None:
        """
        Save metadata to storage.

        :param key: Metadata key to save to.
        :param data: Data to save to metadata.
        """
        with self.open('w', (timestamp, timestamp)) as w:
            w.meta[f'{key}'] = data
        return

    def getAttributes(self) -> list[Attribute]:
        """
        Find list of attribute names from storage config.

        :return: List of attribute names.
        """
        return self.getConfig().attrs

    def getMetrics(self) -> list[Metric]:
        """
        Find List of metric names from storage config

        :return: List of metric names.
        """
        return self.getConfig().metrics

    def getDerivedNames(self) -> list[str]:
        # if no attributes are set in the metric, use all
        return [m.entry_name(a.name) for m in self.config.metrics
                for a in self.config.attrs if not m.attributes
                or a.name in [ma.name for ma in m.attributes]]

    @contextlib.contextmanager
    def open(self, mode:str='r', timestamp=None) -> Generator[tiledb.SparseArray, None, None]:
        """
        Open stream for TileDB database in given mode and at given timestamp.

        :param mode: Mode to open TileDB stream in. Valid options are
            'w', 'r', 'm', 'd'., defaults to 'r'.
        :param timestamp: Timestamp to open database at., defaults to None.
        :raises Exception: Incorrect Mode was given, only valid modes are 'w' and 'r'.
        :raises Exception: Path exists and is not a TileDB array.
        :raises Exception: Path does not exist.
        :yield: TileDB array context manager.
        """

        # tiledb and dask have bad interaction with opening an array if
        # other threads present

        if tiledb.object_type(self.config.tdb_dir) == "array":
            if mode in ['w', 'r', 'd', 'm']:
                tdb = tiledb.open(self.config.tdb_dir, mode, timestamp=timestamp)
            else:
                raise Exception(f"Given open mode '{mode}' is not valid")
        elif pathlib.Path(self.config.tdb_dir).exists():
            raise Exception(f"Path {self.config.tdb_dir} already exists and is not" +
                    " initialized for TileDB access.")
        else:
            raise Exception(f"Path {self.config.tdb_dir} does not exist")

        try:
            yield tdb
        finally:
            tdb.close()

    def reserve_time_slot(self) -> int:
        """
        Increment time slot in database and reserve that spot for a new
        shatter process.

        :param config: Shatter config will be written as metadata to reserve
            time slot.

        :return: Time slot.
        """
        with tiledb.open(self.config.tdb_dir, 'r') as r:
            latest = json.loads(r.meta['config'])

        time = latest['next_time_slot']
        latest['next_time_slot'] = time + 1

        with tiledb.open(self.config.tdb_dir, 'w') as w:
            w.meta['config'] = json.dumps(latest)

        return time

    def get_history(self, start_time: datetime, end_time: datetime,
                    bounds: Bounds, name:str=None, concise:bool=False):
        """
        Retrieve history of the database at current point in time.

        :param start_time: Query parameter, starting datetime of process.
        :param end_time: Query parameter, ending datetime of process.
        :param bounds: Query parameter, bounds to query by.
        :param name: Query paramter, shatter process uuid., by default None
        :param concise: Whether or not to give shortened version of history.
        :return: Returns list of array fragments that meet query parameters.
        """

        # Get metadata configs from each timestamp that contains array fragments
        # In future may want to collect all configs regardless, since they're
        # still being stored there.
        af = tiledb.array_fragments(self.config.tdb_dir, True)
        m = [ ]
        for idx in range(len(af)):

            time_range =  af[idx].timestamp_range
            # all shatter processes should be input as a point in time, eg (1,1)
            if isinstance(time_range, tuple):
                time_range = time_range[0]

            try:
                s_str = self.getMetadata('shatter', time_range)
            except KeyError:
                continue
            s = ShatterConfig.from_string(s_str)
            if s.bounds.disjoint(bounds):
                continue

            # filter name
            if name is not None and name != s.name:
                continue

            # filter dates
            if isinstance(s.date, tuple) and len(s.date) == 2:
                if s.date[1] < start_time or s.date[0] > end_time:
                    continue
            elif isinstance(s.date, tuple) and len(s.date) == 1:
                if s.date[0] < start_time or s.date[0] > end_time:
                    continue
            else:
                if s.date < start_time or s.date > end_time:
                    continue

            if concise:
                h = s.history_json()
            else:
                h = s.to_json()
            m.append(h)

        return m

    def mbrs(self, proc_num: int):
        """
        Get minimum bounding rectangle of a given shatter process. If this process
        has been finished and consolidated the mbr will be much less granulated
        than if the fragments are still intact. Mbrs are represented as tuples
        in the form of ((minx, maxx), (miny, maxy))

        :param proc_num: Process number or time slot of the shatter process.
        :return: Returns mbrs that match the given process number.
        """
        af_all = self.get_fragments_by_time(proc_num)
        mbrs_list = tuple(mbrs for af in af_all for mbrs in af.mbrs)
        mbrs = tuple(tuple(tuple(a.item() for a in mb) for mb in m) for m in mbrs_list)
        return mbrs

    def get_fragments_by_time(self, proc_num: int) -> list[tiledb.FragmentInfo]:
        """
        Get TileDB array fragments from the time slot specified.

        :param proc_num: Requested time slot.
        :return: Array fragments from time slot.
        """
        af = tiledb.array_fragments(self.config.tdb_dir, include_mbrs=True)
        return [a for a in af if a.timestamp_range == (proc_num, proc_num)]

    def delete(self, proc_num: int) -> ShatterConfig:
        """
        Delete Shatter process and all associated data from database.

        :param proc_num: Shatter process time slot
        :return: Config of deleted Shatter process
        """

        self.config.log.debug(f'Deleting time slot {proc_num}...')
        with self.open('r', (proc_num, proc_num)) as r:
            sh_cfg: ShatterConfig = ShatterConfig.from_string(r.meta['shatter'])
            sh_cfg.mbr = ()
            sh_cfg.finished = False

        self.config.log.debug('Deleting fragments...')
        tiledb.Array.delete_fragments(self.config.tdb_dir,
            timestamp_start=proc_num, timestamp_end=proc_num)
        self.config.log.debug('Rewriting config.')
        with self.open('w', (proc_num, proc_num)) as w:
            w.meta['shatter'] = json.dumps(sh_cfg.to_json())
        return sh_cfg

    def consolidate_shatter(self, proc_num: int, retries=0) -> None:
        """
        Consolidate the fragments from a shatter process into one fragment.
        This makes the database perform better, but reduces the granularity of
        time traveling.

        :param proc_num: Time slot associated with shatter process.
        """
        try:
            afs = self.get_fragments_by_time(proc_num)
            uris = [ os.path.split(urllib.parse.urlparse(f.uri).path)[-1] for f in afs ]
            tiledb.consolidate(self.config.tdb_dir, fragment_uris=uris)
            c = tiledb.Config({"sm.vacuum.mode": "fragments"})
            tiledb.vacuum(self.config.tdb_dir, c)
            self.config.log.info(f"Consolidated time slot {proc_num}.")
        except Exception as e:
            if retries >= 3:
                self.config.log.warning("Failed to consolidate time slot "
                        f"{proc_num} {retries} time(s). Stopping.")
                raise e
            self.config.log.warning("Failed to consolidate time slot "
                    f"{proc_num} {retries+1} time. Retrying...")
            self.consolidate_shatter(proc_num, retries+1)

