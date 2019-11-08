import numpy as np
from shapely.geometry import Point
import geopandas as gpd


def check_polygon(p):
    if not isinstance(p, gpd.GeoSeries):
        raise TypeError('Must be gpd.GeoSeries, not {}'.format(type(p)))
    if p.hasnans:
        raise ValueError('GeoSeries contains invalid values.')
    if not p.size > 0:
        raise ValueError('GeoSeries is is_empty')
    return True


def check_pre_proj(p):
    # tests area expected in regular projection
    check_polygon(p)
    if not (np.isclose(p.area, 1712.995228)):
        raise ValueError('Incorrect area for Canada for expected projection. Check polygon.')
    return True


def check_post_proj(p):
    # tests area expected in rotated projection
    check_polygon(p)
    if not (np.isclose(p.area, 837.229487)):
        raise ValueError('Incorrect area for Canada for expected projection. Check polygon.')
    return True


def check_coords(x, y):
    if (not isinstance(x, np.ndarray)) or (not isinstance(y, np.ndarray)):
        raise TypeError("Please provide an object of type {}".format(np.ndarray))
    if x.max() >= 35 or x.min() <= -35:
        raise ValueError("Unexpected range of values in x dimension")
    if y.max() >= 35 or y.min() <= -35:
        raise ValueError("Unexpected range of values in y dimension")
    if x.size != 155 or y.size != 130:
        raise ValueError("Longitude and latitude expected size 155 and 130 respectively.")
    return True


def rotate_shapefile(
    p,
    crs={
        'proj': 'ob_tran',
        'o_proj': 'longlat',
        'lon_0': -97,
        'o_lat_p': 42.5,
        'a': 6378137,
        'to_meter': 0.0174532925199,
        'no_defs': True}
):
    '''Rotates a shapefile to a new crs defined by a proj4 dictionary.
    Args:
        p (geopandas.GeoSeries object): polygons of Canada
        crs (dict): proj4 dictionary

    Returns:
        target (geopandas.GeoSeries object): geographic polygons
            in new projection
    '''
    # this checks the polyon input
    check_polygon(p)
    # this checks polygon can be rotated
    check_pre_proj(p)
    target = p.to_crs(crs)
    # this checks the rotation
    check_post_proj(target)

    return target


def gen_raster_mask_from_vector(x, y, p):
    '''Determines if points are contained within polygons of Canada
    Args:
        x, y (np.ndarray): Arrays containing the rlon and rlat of CanRCM4
            grids
        p (geopandas.GeoSeries object): rotated polygons of Canada

    Returns:
        grid (np.ndarray): boolean 2D grid mask of CanRCM4 raster clipped
            based on polygon boundaries
    '''
    # this checks the coordinate inputes
    check_coords(x, y)
    # this checks the polygon input
    check_polygon(p)
    # this checks that the polygon is in rotated form
    check_post_proj(p)
    grid = np.meshgrid(x, y)[0]
    for i, rlon in enumerate(x):
        for j, rlat in enumerate(y):
            pt = Point(rlon, rlat)
            grid[j, i] = p.contains(pt)
    return grid == 1.
