import pytest
import geopandas as gpd
import numpy as np
from climpyrical.mask import (
    check_polygon_validity,
    check_polygon_before_projection,
    check_polygon_after_projection,
    check_input_grid_coords,
    rotate_shapefile,
    gen_raster_mask_from_vector,
)
from pkg_resources import resource_filename


canada = gpd.read_file(
    resource_filename("climpyrical", "tests/data/canada.geojson")
).geometry

# rotate the vectors using the to_crs method. GeoPandas does
# not preserve the crs that's defined, otherwise a standalone
# pre-projected in rotated pole polygon would be used
rotated_crs = {
    "proj": "ob_tran",
    "o_proj": "longlat",
    "lon_0": -97,
    "o_lat_p": 42.5,
    "a": 6378137,
    "to_meter": 0.0174532925199,
    "no_defs": True,
}

rotated_canada = canada.to_crs(rotated_crs)

transformed_world = gpd.read_file(
    resource_filename("climpyrical", "tests/data/transformed_world.geojson")
).geometry

bad_polygon = gpd.read_file(
    resource_filename("climpyrical", "tests/data/bad_polygon.geojson")
).geometry

good_polygon = gpd.read_file(
    resource_filename("climpyrical", "tests/data/good_polygon.geojson")
).geometry


@pytest.mark.parametrize(
    "p,error",
    [
        ({"string"}, TypeError),
        ({5}, TypeError),
        (canada, None),
        (rotated_canada, None),
        (transformed_world, None),
        (good_polygon, None),
        (bad_polygon, ValueError),
        (gpd.GeoSeries(), ValueError),
    ],
)
def test_check_polygon_validity(p, error):
    if error is None:
        check_polygon_validity(p)
    else:
        with pytest.raises(error):
            check_polygon_validity(p)


@pytest.mark.parametrize(
    "p,warning",
    [
        (canada, None),
        (rotated_canada, UserWarning),
        (transformed_world, UserWarning),
    ],
)
def test_check_polygon_before_projection(p, warning):
    if warning is None:
        check_polygon_before_projection(p)
    else:
        with pytest.warns(warning):
            check_polygon_before_projection(p)


@pytest.mark.parametrize(
    "p,error",
    [
        ({"string"}, TypeError),
        ({5}, TypeError),
        (canada, ValueError),
        (rotated_canada, None),
        (transformed_world, ValueError),
        (gpd.GeoSeries(), ValueError),
    ],
)
def test_check_polygon_after_projection(p, error):
    if error is None:
        check_polygon_after_projection(p)
    else:
        with pytest.raises(error):
            check_polygon_after_projection(p)


@pytest.mark.parametrize(
    "x,y,error",
    [
        ("x", np.linspace(-24, 24, 155), TypeError),
        (np.linspace(-24, 24, 155), "y", TypeError),
        (np.ones((2, 2)), np.linspace(-24, 24, 155), ValueError),
        (np.linspace(-24, 24, 155), np.ones((2, 2)), ValueError),
        (np.linspace(-24, 24, 155), np.linspace(-24, 24, 155), None),
    ],
)
def test_check_input_grid_coords(x, y, error):
    if error is None:
        check_input_grid_coords(x, y)
    else:
        with pytest.raises(error):
            check_input_grid_coords(x, y)


@pytest.mark.parametrize(
    "p,crs,expected", [(canada, rotated_crs, rotated_canada)]
)
def test_rotate_shapefile(p, crs, expected):
    assert rotate_shapefile(p, crs).geom_almost_equals(expected).values[0]


maskarray = np.load(
    resource_filename("climpyrical", "tests/data/maskarray.npy")
)


@pytest.mark.parametrize(
    "x,y,p,expected",
    [
        (
            np.linspace(-33.8800048828125, 33.8800048828125, 155),
            np.linspace(-28.59999656677246, 28.15999984741211, 130),
            rotated_canada,
            maskarray,
        )
    ],
)
def test_gen_raster_mask_from_vector(x, y, p, expected):
    assert np.array_equal(gen_raster_mask_from_vector(x, y, p), expected)
