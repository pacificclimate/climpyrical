import os
import pytest
import geopandas as gpd
import numpy as np
from climpyrical.mask import (
        check_polygon_validity,
        check_polygon_before_projection,
        check_polygon_after_projection,
        check_input_grid_coords,
        rotate_shapefile,
        gen_raster_mask_from_vector
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

canada = gpd.read_file(os.path.join(THIS_DIR, 'data/canada.geojson')).geometry

# rotate the vectors using the to_crs method. GeoPandas does
# not preserve the crs that's defined, otherwise a standalone
# pre-projected in rotated pole polygon would be used
rotated_crs = {
        'proj': 'ob_tran',
        'o_proj': 'longlat',
        'lon_0': -97,
        'o_lat_p': 42.5,
        'a': 6378137,
        'to_meter': 0.0174532925199,
        'no_defs': True
    }

rotated_canada = canada.to_crs(rotated_crs)

transformed_world = gpd.read_file(
                        os.path.join(
                            THIS_DIR,
                            './data/transformed_world.geojson')
                    ).geometry

bad_polygon = gpd.read_file(
                    os.path.join(
                        THIS_DIR,
                        './data/bad_polygon.geojson'
                    )
                ).geometry

good_polygon = gpd.read_file(
                    os.path.join(
                        THIS_DIR,
                        './data/good_polygon.geojson'
                    )
                ).geometry


@pytest.mark.parametrize('p,passed', [
    ({'string'}, False),
    ({5}, False),
    (canada, True),
    (rotated_canada, True),
    (transformed_world, True),
    (good_polygon, True),
    (bad_polygon, False),
    (gpd.GeoSeries(), False)])
def test_check_polygon_validity(p, passed):
    if passed:
        assert check_polygon_validity(p)
    else:
        with pytest.raises((TypeError, ValueError)):
            check_polygon_validity(p)


@pytest.mark.parametrize('p,passed', [
    (canada, True),
    (rotated_canada, False),
    (transformed_world, False)])
def test_check_polygon_before_projection(p, passed):
    if passed:
        assert check_polygon_before_projection(p)
    else:
        with pytest.raises((TypeError, ValueError)):
            check_polygon_before_projection(p)


@pytest.mark.parametrize('p,passed', [
    ({'string'}, False),
    ({5}, False),
    (canada, False),
    (rotated_canada, True),
    (transformed_world, False),
    (gpd.GeoSeries(), False)])
def test_check_polygon_after_projection(p, passed):
    if passed:
        assert check_polygon_after_projection(p)
    else:
        with pytest.raises((TypeError, ValueError)):
            check_polygon_after_projection(p)


@pytest.mark.parametrize(
    'x,y,passed',
    [
        ('x', np.linspace(-24, 24, 155), False),
        (np.linspace(-24, 24, 155), 'y', False),
        (
            np.linspace(0, 10, 30),
            np.linspace(-28.59999656677246, 28.15999984741211, 130),
            False
        ),
        (
            np.linspace(-33.8800048828125, 33.8800048828125, 155),
            np.linspace(0, 10, 130),
            False
        ),
        (
            np.linspace(-33.8800048828125, 33.8800048828125, 155),
            np.linspace(-28.59999656677246, 28.15999984741211, 130),
            True
        ),
        (
            np.linspace(-33.8800048828125, 33.8800048828125, 15),
            np.linspace(-28.59999656677246, 28.15999984741211, 10),
            False
        )
    ]
)
def test_check_input_grid_coords(x, y, passed):
    if passed:
        assert check_input_grid_coords(x, y)
    else:
        with pytest.raises((ValueError, TypeError)):
            check_input_grid_coords(x, y)


@pytest.mark.parametrize('p,crs,expected', [
    (canada, rotated_crs, rotated_canada)])
def test_rotate_shapefile(p, crs, expected):
    assert rotate_shapefile(p, crs).geom_almost_equals(expected).values[0]


maskarray = np.load(os.path.join(THIS_DIR, './data/maskarray.npy'))


@pytest.mark.parametrize(
    'x,y,p,expected',
    [
        (
            np.linspace(-33.8800048828125, 33.8800048828125, 155),
            np.linspace(-28.59999656677246, 28.15999984741211, 130),
            rotated_canada,
            maskarray
        )
    ]
)
def test_gen_raster_mask_from_vector(x, y, p, expected):
    assert np.array_equal(gen_raster_mask_from_vector(x, y, p), expected)
