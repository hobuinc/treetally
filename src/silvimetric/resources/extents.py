import math
import numpy as np

import logging
import dask
import dask.bag as db

from math import ceil

from .log import Log
from .bounds import Bounds
from .storage import Storage
from .data import Data

IndexDomain = tuple[float, float]
IndexDomainList = tuple[IndexDomain, IndexDomain]

class Extents(object):
    """Handles bounds operations for point cloud data."""

    def __init__(self,
                 bounds: Bounds,
                 resolution: float,
                 alignment: str,
                 root: Bounds):

        self.bounds = bounds
        """Bounding box of this section of data."""
        self.root = root
        """Root bounding box of the database."""

        # adjust bounds so they're matching up with cell lines or cell centers
        self.bounds.adjust_alignment(resolution, alignment)
        minx, miny, maxx, maxy = self.bounds.get()


        self.rangex = maxx - minx
        """Range of X Indices"""
        self.rangey = maxy - miny
        """Range of Y indices"""
        self.resolution = resolution
        """Resolution of database."""
        self.alignment = alignment
        """Alignment of pixels in database."""
        self.cell_count = int((self.rangex * self.rangey) / self.resolution ** 2)
        """Number of cells in this Extents"""

        self.x1 = math.floor((minx - self.root.minx) / resolution)
        """Minimum X index"""
        self.x2 = math.ceil((maxx - self.root.minx) / resolution)
        """Maximum X index"""

        self.y1 = math.ceil((self.root.maxy - maxy) / resolution)
        """Minimum Y index, or maximum Y value in point cloud"""
        self.y2 = math.ceil((self.root.maxy - miny) / resolution)
        """Maximum Y index, or minimum Y value in point cloud"""
        self.domain: IndexDomainList = ((self.x1, self.x2), (self.y1, self.y2))
        """Minimum bounding rectangle of this Extents"""

    def get_indices(self):
        """
        Create indices for this section of the database relative to the root
        bounds.

        :return: Indices of this bounding box
        """
        return np.array(
            [(i,j) for i in range(self.x1, self.x2)
            for j in range(self.y1, self.y2)],
            dtype=[('x', np.int32), ('y', np.int32)]
        )

    def disjoint_by_mbr(self, mbr):
        """
        Determine if this Extents shares any points with a minimum bounding
        rectangle.

        :param mbr: Minimum bounding rectangle as defined by TileDB.
        :return: True if no shared points, false otherwise.
        """
        xs, ys = mbr
        x1, x2 = xs
        y1, y2 = ys
        if x1 > self.x2 or x2 < self.x1:
            return True
        if y1 > self.y2 or y2 < self.y1:
            return True
        return False

    def disjoint(self, other):
        """
        Determined if this Extents shares any points with another Extents object.

        :param other: Extents object to compare against.
        :return: True if no shared points, false otherwise.
        """
        return self.bounds.disjoint(other.bounds)

    def chunk(self, data: Data, res_threshold=100, pc_threshold=600000,
            depth_threshold=6):
        """
        Split up a dataset into tiles based on the given thresholds. Unlike Scan
        this will filter out any tiles that contain no points.

        :param data: Incoming Data object to oeprate on.
        :param res_threshold: Resolution threshold., defaults to 100
        :param pc_threshold: Point count threshold., defaults to 600000
        :param depth_threshold: Tree depth threshold., defaults to 6
        :return: Return list of Extents that fit the criteria
        """
        if self.root is not None:
            bminx, bminy, bmaxx, bmaxy = self.root.get()
            r = self.root
        else:
            bminx, bminy, bmaxx, bmaxy = self.bounds.get()
            r = self.bounds

        # make bounds in scale with the desired resolution
        minx = bminx + (self.x1 * self.resolution)
        maxx = bminx + (self.x2 * self.resolution)

        miny = bmaxy - (self.y2 * self.resolution)
        maxy = bmaxy - (self.y1 * self.resolution)

        chunk = Extents(Bounds(minx, miny, maxx, maxy), self.resolution, self.alignment, r)

        if self.bounds == self.root:
            self.root = chunk.bounds

        filtered = []
        curr = db.from_delayed([dask.delayed(ch.filter)(data, res_threshold, pc_threshold,
            depth_threshold, 1) for ch in chunk.split()])
        curr_depth = 1

        logger = data.storageconfig.log
        while curr.npartitions > 0:

            logger.debug(f'Filtering {curr.npartitions} tiles at depth {curr_depth}')
            n = curr.compute()
            to_add = [ne for ne in n if isinstance(ne, Extents)]
            if to_add:
                filtered = filtered + to_add

            curr = db.from_delayed([ne for ne in n if not isinstance(ne, Extents)])
            curr_depth += 1

        return filtered


    def split(self):
        """
        Split this extent into 4 children along the cell lines

        :return: Returns 4 child extents
        """
        minx, miny, maxx, maxy = self.bounds.get()

        x_adjusted = math.floor((maxx - minx) / 2 / self.resolution)
        y_adjusted = math.ceil((maxy - miny) / 2 / self.resolution)

        midx = minx + (x_adjusted * self.resolution)
        midy = maxy - (y_adjusted * self.resolution)

        exts =  [
            Extents(Bounds(minx, miny, midx, midy), self.resolution, self.alignment, self.root), #lower left
            Extents(Bounds(midx, miny, maxx, midy), self.resolution, self.alignment, self.root), #lower right
            Extents(Bounds(minx, midy, midx, maxy), self.resolution, self.alignment, self.root), #top left
            Extents(Bounds(midx, midy, maxx, maxy), self.resolution, self.alignment, self.root)  #top right
        ]
        return exts

    def filter(self, data: Data, res_threshold=100, pc_threshold=600000, depth_threshold=6, depth=0):
        """
        Creates quad tree of chunks for this bounds, runs pdal quickinfo over
        this to determine if there are any points available. Uses a bottom resolution
        of 1km.

        :param data: Data object containing point cloud details.
        :param res_threshold: Resolution threshold., defaults to 100
        :param pc_threshold: Point count threshold., defaults to 600000
        :param depth_threshold: Tree depth threshold., defaults to 6
        :param depth: Current tree depth., defaults to 0
        :return: Returns a list of Extents.
        """

        pc = data.estimate_count(self.bounds)
        target_pc = pc_threshold
        minx, miny, maxx, maxy = self.bounds.get()

        # is it empty?
        if not pc:
            return [ ]
        else:
            # has it hit the threshold yet?
            area = (maxx - minx) * (maxy - miny)
            next_split_x = (maxx-minx) / 2
            next_split_y = (maxy-miny) / 2

            # if the next split would put our area below the resolution, or if
            # the point count is less than the threshold (600k) then use this
            # tile as the work unit.
            if next_split_x < self.resolution or next_split_y < self.resolution:
                return [ self ]
            elif pc < target_pc:
                return [ self ]
            elif area < res_threshold**2 or depth >= depth_threshold:
                pc_per_cell = pc / (area / self.resolution**2)
                cell_estimate = ceil(target_pc / pc_per_cell)

                return self.get_leaf_children(cell_estimate)
            else:
                return [ dask.delayed(ch.filter)(data, res_threshold, pc_threshold, depth_threshold, depth=depth+1) for ch in self.split() ]


    def _find_dims(self, tile_size):
        """
        Find most square-like Extents given the number of cells per tile.

        :param tile_size: Number of cells per tile.
        :return: Returns x and y coordinates in a list.
        """
        s = math.sqrt(tile_size)
        if int(s) == s:
            return [s, s]
        rng = np.arange(1, tile_size+1, dtype=np.int32)
        factors = rng[np.where(tile_size % rng == 0)]
        idx = int((factors.size/2)-1)
        x = factors[idx]
        y = int(tile_size/ x)
        return [x, y]

    def get_leaf_children(self, tile_size):
        """
        Get children Extents with given number of cells per tile.

        :param tile_size: Cells per tile.
        :yield: Yield from list of child extents.
        """
        res = self.resolution
        xnum, ynum = self._find_dims(tile_size)

        local_xs = np.array([
                [x, min(x+xnum, self.x2)]
                for x in range(self.x1, self.x2, int(xnum))
            ], dtype=np.float64)
        dx = (res * local_xs) + self.root.minx

        local_ys = np.array([
                [min(y+ynum, self.y2), y]
                for y in range(self.y1, self.y2, int(ynum))
            ], dtype=np.float64)
        dy = self.root.maxy - (res * local_ys)

        coords_list = np.array([[*x,*y] for x in dx for y in dy],dtype=np.float64)
        yield from [
            Extents(Bounds(minx, miny, maxx, maxy), self.resolution, self.alignment, self.root)
            for minx,maxx,miny,maxy in coords_list
        ]

    @staticmethod
    def from_storage(tdb_dir: str):
        """
        Create Extents from information stored in database.

        :param tdb_dir: TileDB database directory.
        :return: Returns resulting Extents.
        """
        storage = Storage.from_db(tdb_dir)
        meta = storage.getConfig()
        return Extents(meta.root, meta.resolution, meta.alignment, meta.root)

    @staticmethod
    def from_sub(tdb_dir: str, sub: Bounds):
        """
        Create an Extents that is less than the overall extents of the database.

        :param tdb_dir: TileDB database directory.
        :param sub: Desired bounding box.
        :return: Returns resulting Extents.
        """
        storage = Storage.from_db(tdb_dir)

        meta = storage.getConfig()
        res = meta.resolution
        align = meta.alignment
        base_extents = Extents(meta.root, res, align, meta.root)
        base = base_extents.bounds

        return Extents(sub, res, align, base)


    def __repr__(self):
        minx, miny, maxx, maxy = self.bounds.get()
        return f"([{minx:.2f},{maxx:.2f}],[{miny:.2f},{maxy:.2f}])"
