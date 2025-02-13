from pathlib import Path
from osgeo import gdal
from pyproj import CRS

from silvimetric import grid_metrics, ExtractConfig, Extents, Log, extract, Storage

def tif_test(extract_config):
    minx, miny, maxx, maxy = extract_config.bounds.get()
    resolution = extract_config.resolution
    filenames = [grid_metrics[m.name].entry_name(a.name)
                    for m in extract_config.metrics
                    for a in extract_config.attrs]
    storage = Storage.from_db(extract_config.tdb_dir)
    e = Extents(extract_config.bounds, extract_config.resolution, storage.config.alignment, storage.config.root)
    root_maxy = storage.config.root.maxy

    for f in filenames:
        if 'Return' in f:
            continue
        path = Path(extract_config.out_dir) / f'{f}.tif'
        assert path.exists()

        raster: gdal.Dataset = gdal.Open(str(path))
        derived = CRS.from_user_input(raster.GetProjection())
        rminx, xres, xskew, rmaxy, yskew, yres  = raster.GetGeoTransform()
        assert rminx == minx
        assert rmaxy == maxy
        assert xres == resolution
        assert -yres == resolution


        assert derived == extract_config.crs

        xsize = e.x2
        ysize = e.y2
        assert raster.RasterXSize == xsize
        assert raster.RasterYSize == ysize

        r = raster.ReadAsArray()
        assert all([ r[y,x] == ((root_maxy/resolution)-y-1)  for y in range(e.y1, e.y2) for x in range(e.x1, e.x2)])

class Test_Extract(object):

    def test_config(self, extract_config, tdb_filepath, tif_filepath, extract_attrs):
        assert all([a in [*extract_attrs, 'count'] for a in extract_config.attrs])
        assert all([a in extract_config.attrs for a in extract_attrs])
        assert extract_config.tdb_dir == tdb_filepath
        assert extract_config.out_dir == tif_filepath

    def test_extract(self, extract_config):
        extract(extract_config)
        tif_test(extract_config)

    def test_sub_bounds_extract(self, extract_config):
        s = extract_config
        e = Extents.from_storage(s.tdb_dir)
        log = Log(20)

        for b in e.split():
            ec = ExtractConfig(tdb_dir = s.tdb_dir,
                               log = log,
                               out_dir = s.out_dir,
                               attrs = s.attrs,
                               metrics = s.metrics,
                               bounds = b.bounds)
            extract(ec)
            tif_test(ec)

    def test_multi_value(self, multivalue_config):
        extract(multivalue_config)
        tif_test(multivalue_config)