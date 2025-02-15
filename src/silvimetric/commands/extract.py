from pathlib import Path
from itertools import chain


from typing_extensions import Union
from osgeo import gdal, osr
import dask
import numpy as np
import pandas as pd


from .. import Storage, Extents, ExtractConfig

np_to_gdal_types = {
    np.dtype(np.byte).str: gdal.GDT_Byte,
    np.dtype(np.int8).str: gdal.GDT_Int16,
    np.dtype(np.uint16).str: gdal.GDT_UInt16,
    np.dtype(np.int16).str: gdal.GDT_Int16,
    np.dtype(np.uint32).str: gdal.GDT_UInt32,
    np.dtype(np.int32).str: gdal.GDT_Int32,
    np.dtype(np.uint64).str: gdal.GDT_UInt64,
    np.dtype(np.int64).str: gdal.GDT_Int64,
    np.dtype(np.float32).str: gdal.GDT_Float32,
    np.dtype(np.float64).str: gdal.GDT_Float64
}

def write_tif(xsize: int, ysize: int, data:np.ndarray, name: str,
        config: ExtractConfig) -> None:
    """
    Write out a raster with GDAL

    :param xsize: Length of X plane.
    :param ysize: Length of Y plane.
    :param data: Data to write to raster.
    :param name: Name of raster to write.
    :param config: ExtractConfig.
    """
    osr.UseExceptions()
    path = Path(config.out_dir) / f'{name}.tif'
    crs = config.crs
    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs.to_wkt())
    b = config.bounds

    transform = [b.minx, config.resolution, 0,
                 b.maxy, 0, -1*config.resolution]

    driver = gdal.GetDriverByName("GTiff")
    gdal_type = np_to_gdal_types[np.dtype(data.dtype).str]
    tif = driver.Create(str(path), int(xsize), int(ysize), 1, gdal_type, )
    tif.SetGeoTransform(transform)
    tif.SetProjection(srs.ExportToWkt())
    tif.GetRasterBand(1).WriteArray(data)
    tif.GetRasterBand(1).SetNoDataValue(np.nan)
    tif.FlushCache()
    tif = None

def get_metrics(data_in: pd.DataFrame, storage: Storage) -> Union[None, pd.DataFrame]:
    """
    Reruns a metric over this cell. Only called if there is overlapping data.

    :param data_in: Dataframe to be rerun.
    :param storage: Base storage object.
    :return: Combined dict of attribute and newly derived metric data.
    """

    #TODO should just use the metric calculation methods from shatter
    if data_in is None:
        return None

    def expl(x):
        return x.explode()

    attrs = [a.name for a in storage.config.attrs if a.name not in ['X','Y']]

    # set index so we can apply to the whole dataset without needing to skip X and Y
    # then reset in the index because that's what metric.do expects
    exploded = data_in.set_index(['X','Y']).apply(expl)[attrs].reset_index()
    metric_data = dask.persist(*[ m.do(exploded) for m in storage.config.metrics ])

    data_out = data_in.set_index(['X','Y']).join([m for m in metric_data])
    return data_out

def handle_overlaps(config: ExtractConfig, storage: Storage, indices: np.ndarray) -> pd.DataFrame:
    """
    Handle cells that have overlapping data. We have to re-run metrics over these
    cells as there's no other accurate way to determined metric values. If there
    are no overlaps, this will do nothing.

    :param config: ExtractConfig.
    :param storage: Database storage object.
    :param indices: Indices with overlap.
    :return: Dataframe of rerun data.
    """

    ma_list = storage.getDerivedNames()
    att_list = [ a.name for a in config.attrs ]

    minx = indices['x'].min()
    maxx = indices['x'].max()
    miny = indices['y'].min()
    maxy = indices['y'].max()

    att_meta = {}
    att_meta['X'] = np.int32
    att_meta['Y'] = np.int32
    att_meta['count'] = np.int32
    for a in config.attrs:
        att_meta[a.name] =  a.dtype

    with storage.open("r") as tdb:
        # TODO this can be more efficient. Use count to find indices, then work
        # with that smaller set from there. Working as is for now, but slow.
        dit = tdb.query(attrs=[*att_list, *ma_list], order='F', coords=True,
                return_incomplete=True, use_arrow=False).df[minx:maxx, miny:maxy]
        data = pd.DataFrame()

        storage.config.log.info('Collecting database information...')
        for d in dit:
            if data.empty:
                data = d
            else:
                data = pd.concat([data, d])


        # should find values that are not unique, meaning they have multiple entries
        data = data.set_index(['X','Y'])
        redo_indices = data.index[data.index.duplicated(keep='first')]
        if redo_indices.empty:
            return data.reset_index()

        # data with overlaps
        redo_data = data.loc[redo_indices][att_list].groupby(['X','Y']).agg(lambda x: list(chain(*x)))
        # data that has no overlaps
        clean_data = data.loc[data.index[~data.index.duplicated(False)]]

        storage.config.log.warning('Overlapping data detected. Rerunning metrics over these cells...')
        new_metrics = get_metrics(redo_data.reset_index(), storage)
    return pd.concat([clean_data, new_metrics]).reset_index()

def extract(config: ExtractConfig) -> None:
    """
    Pull data from database for each desired metric and output them to rasters

    :param config: ExtractConfig.
    """

    dask.config.set({"dataframe.convert-string": False})

    storage = Storage.from_db(config.tdb_dir)
    ma_list = storage.getDerivedNames()
    config.log.debug(f'Extracting metrics {[m for m in ma_list]}')
    root_bounds=storage.config.root

    e = Extents(config.bounds, config.resolution, storage.config.alignment, root=root_bounds)
    i = e.get_indices()
    xsize = e.x2
    ysize = e.y2

    # figure out if there are any overlaps and handle them
    final = handle_overlaps(config, storage, i).sort_values(['Y', 'X'])

    # output metric data to tifs
    config.log.info(f"Writing rasters to {config.out_dir}")
    for ma in ma_list:
        # TODO should output in sections so we don't run into memory problems
        m_data = np.full(shape=(ysize,xsize), fill_value=np.nan, dtype=final[ma].dtype)
        a = final[['X','Y',ma]].to_numpy()
        for x,y,md in a[:]:
            m_data[int(y)][int(x)] = md

        write_tif(xsize, ysize, m_data, ma, config)