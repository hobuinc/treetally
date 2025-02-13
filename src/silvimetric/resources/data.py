from .bounds import Bounds
from .config import StorageConfig
import numpy as np
import copy

import pdal

import pathlib
import json

class Data:
    """Represents a point cloud or PDAL pipeline, and performs essential operations
    necessary to understand and execute a Shatter process."""

    def __init__(self,
                 filename: str,
                 storageconfig: StorageConfig,
                 bounds: Bounds = None):
        self.filename = filename
        """Path to either PDAL pipeline or point cloud file"""

        self.bounds = bounds
        """Bounds of this section of data"""

        self.reader_thread_count = 2
        """Thread count for PDAL reader. Keep to 2 so we don't hog threads"""

        self.reader = self.get_reader()
        """PDAL reader"""

        if self.bounds is None:
            self.bounds = Data.get_bounds(self.reader)

        # adjust bounds if necessary
        self.bounds.adjust_alignment(storageconfig.resolution, storageconfig.alignment)
        self.bounds = Bounds.shared_bounds(self.bounds, storageconfig.root)

        self.storageconfig = storageconfig
        """:class:`silvimetric.resources.StorageConfig`"""
        self.pipeline = self.get_pipeline()
        """PDAL pipeline"""

        self.log = storageconfig.log

    def to_json(self):
        j = dict(filename=self.filename, bounds=self.bounds.get(),
                pipeline=json.loads(self.pipeline.pipeline), is_pipeline=self.is_pipeline())
        return j

    def __repr__(self):
        return json.dumps(self.to_json(), indent=2)

    def is_pipeline(self) -> bool:
        """Does this instance represent a pdal.Pipeline or a simple filename

        :return: Return true if input is a pipeline
        """

        p = pathlib.Path(self.filename)
        if p.suffix == '.json' and p.name != 'ept.json':
            return True
        return False


    def make_pipeline(self) -> pdal.Pipeline:
        """Take a COPC or EPT endpoint and generate a PDAL pipeline for it

        :return: Return PDAL pipeline
        """

        reader = pdal.Reader(self.filename, tag='reader')
        reader._options['threads'] = self.reader_thread_count
        if self.bounds:
            reader._options['bounds'] = str(self.bounds)


        return reader.pipeline()
        ## TODO
        ## Remove these filters.assign stages for now
        ## Waiting for a pdal/pdal-python fix to this
        ## https://github.com/PDAL/python/issues/174

        # class_zero = pdal.Filter.assign(value="Classification = 0")
        # rn = pdal.Filter.assign(value="ReturnNumber = 1 WHERE ReturnNumber < 1")
        # nor = pdal.Filter.assign(value="NumberOfReturns = 1 WHERE NumberOfReturns < 1")
        # ferry = pdal.Filter.ferry(dimensions="X=>xi, Y=>yi")

        # return reader | class_zero | rn | nor

    def get_pipeline(self) -> pdal.Pipeline:
        """Fetch the pipeline for the instance

        :raises Exception: File type isn't COPC or EPT
        :raises Exception: More than one reader detected
        :return: Return PDAL pipline
        """

        # If we are a pipeline, read and parse it. If we
        # aren't, go make_pipeline using some options that
        # process the data
        if self.is_pipeline():
            p = pathlib.Path(self.filename)
            j = p.read_bytes().decode('utf-8')
            stages = pdal.pipeline._parse_stages(j)
            pipeline = pdal.Pipeline(stages)
        else:
            pipeline = self.make_pipeline()

        # only support COPC or EPT if someone gave us a pipeline
        # because we need to use bounds-accelerated reads to
        # process data quickly
        allowed_readers = ['copc', 'ept']
        readers = []
        stages = []

        for stage in pipeline.stages:
            stage_type, stage_kind = stage.type.split('.')
            if stage_type == 'readers':
                if not stage_kind in allowed_readers:
                    raise Exception(f"Readers for SilviMetric must be of type 'copc' or 'ept', not '{stage_kind}'")
                readers.append(stage)

            # we only answer to copc or ept readers
            if stage_kind in allowed_readers:
                if self.bounds:
                    res = self.storageconfig.resolution
                    collar = Bounds(
                        self.bounds.minx - res,
                        self.bounds.miny - res,
                        self.bounds.maxx + res,
                        self.bounds.maxy + res
                    )
                    stage._options['bounds'] = str(collar)
                    # stage._options['bounds'] = str(self.bounds)

            # We strip off any writers from the pipeline that were
            # given to us and drop them  on the floor
            if stage_type != 'writers':
                stages.append(stage)

        # we don't support weird pipelines of shapes
        # that aren't simply a line.
        if len(readers) != 1:
            raise Exception(f"Pipelines can only have one reader of type {allowed_readers}")

        resolution = self.storageconfig.resolution
        # Add xi and yi – only need this for PDAL < 2.6
        ferry = pdal.Filter.ferry(dimensions="X=>xi, Y=>yi")
        assign_x = pdal.Filter.assign(value=f"xi = (X - {self.storageconfig.root.minx}) / {resolution}")
        assign_y = pdal.Filter.assign(value=f"yi = (({self.storageconfig.root.maxy} - Y) / {resolution}) - 1")
        # hag = pdal.Filter.hag_nn()

        stages.append(ferry)
        stages.append(assign_x)
        stages.append(assign_y)
        # stages.append(hag)

        # return our pipeline
        return pdal.Pipeline(stages)

    def execute(self):
        """Execute PDAL pipeline

        :raises Exception: PDAL error message passed from execution
        """
        try:
            self.pipeline.execute()
            if self.pipeline.log and self.pipeline.log is not None:
                self.log.debug(f"PDAL log: {self.pipeline.log}")
        except Exception as e:
            if self.pipeline.log and self.pipeline.log is not None:
                self.log.debug(f"PDAL log: {self.pipeline.log}")
            print(self.pipeline.pipeline, e)
            raise e

    def get_array(self) -> np.ndarray:
        """Fetch the array from the execute()'d pipeline

        :return: get data as a numpy ndarray
        """
        return self.pipeline.arrays[0]
    array = property(get_array)

    def get_reader(self) -> pdal.Reader:
        """Grab or make the reader for this instance so we can use it to do things
        like get the count()

        :return: get PDAL reader for input
        """
        if self.is_pipeline():
            p = pathlib.Path(self.filename)
            j = p.read_bytes().decode('utf-8')
            stages = pdal.pipeline._parse_stages(j)
            pipeline = pdal.Pipeline(stages)
            for stage in pipeline.stages:
                stage_type, stage_kind = stage.type.split('.')
                if stage_type == 'readers':
                    return stage
        else:
            reader = pdal.Reader(self.filename)
            reader._options['threads'] = self.reader_thread_count
            return reader

    @staticmethod
    def get_bounds(reader: pdal.Reader) -> Bounds:
        """Get the bounding box of a point cloud from PDAL.

        :param reader: PDAL Reader representing input data
        :return: bounding box of point cloud
        """
        p = reader.pipeline()
        qi = p.quickinfo[reader.type]
        return Bounds.from_string(json.dumps(qi['bounds']))

    def estimate_count(self, bounds: Bounds) -> int:
        """For the provided bounds, estimate the maximum number of points that
        could be inside them for this instance.

        :param bounds: query bounding box
        :return: estimated point count
        """

        # TODO: if bounds is different from root bounds, find way to estimate
        # point count. PDAL quickinfo grabs info from header or ept.json
        # and this reflects point count of entire file
        reader = self.get_reader()
        if bounds:
            reader._options['bounds'] = str(bounds)

        pipeline = reader.pipeline()
        qi = pipeline.quickinfo[reader.type]
        pc = qi['num_points']

        return pc

    def count(self, bounds: Bounds) -> int:
        """For the provided bounds, read and count the number of points that are
        inside them for this instance.

        :param bounds: query bounding box
        :return: point count
        """

        reader = copy.deepcopy(self.get_reader())
        if bounds:
            reader._options['bounds'] = str(bounds)

        pipeline = reader.pipeline()
        pipeline.execute()
        return len(pipeline.arrays[0])
