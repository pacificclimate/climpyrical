import pytest
import geopandas as gpd
from climpyrical.data import read_data
from climpyrical.gridding import extend_north
from climpyrical.mask import (
    check_polygon_validity,
    check_polygon_before_projection,
    rotate_shapefile,
    gen_raster_mask_from_vector,
    make_box,
    gen_upper_archipelago_mask,
    stratify_coords,
)
from pkg_resources import resource_filename
import numpy as np

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
        (canada, None),
        (rotated_canada, None),
        (transformed_world, None),
        (good_polygon, None),
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
        (transformed_world.to_crs("+init=epsg:4236"), UserWarning),
    ],
)
def test_check_polygon_before_projection(p, warning):
    if warning is None:
        check_polygon_before_projection(p)
    else:
        with pytest.warns(warning):
            check_polygon_before_projection(p)


@pytest.mark.parametrize(
    "p,crs,expected", [(canada, rotated_crs, rotated_canada)]
)
def test_rotate_shapefile(p, crs, expected):
    assert rotate_shapefile(p, crs).geom_almost_equals(expected).values[0]


mask_ds = read_data(
    resource_filename("climpyrical", "nrc_data/processed/canada_mask_rp.nc")
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "x,y,p,progress_bar,error",
    [
        (
            mask_ds.rlon.values[::10],
            mask_ds.rlat.values[::10],
            rotated_canada,
            True,
            None,
        ),
        (
            mask_ds.rlon.values[:10],
            mask_ds.rlat.values[:10],
            rotated_canada,
            True,
            ValueError,
        ),
        (
            mask_ds.rlon.values[::10],
            mask_ds.rlat.values[::10],
            rotated_canada,
            False,
            None,
        ),
    ],
)
def test_gen_raster_mask_from_vector(x, y, p, progress_bar, error):
    if error is None:
        result = gen_raster_mask_from_vector(x, y, p, progress_bar)
        assert result.shape == (y.shape[0], x.shape[0])
    else:
        with pytest.raises(ValueError):
            gen_raster_mask_from_vector(x, y, p, progress_bar)


@pytest.mark.parametrize(
    "x,y,dx,dy,error",
    [
        (0, 0, 0.5, "0.5", TypeError),
        (0, 0, 0.5, 0.5, None),
        (0, 0, 0.5, 0.5, None),
    ],
)
def test_make_box(x, y, dx, dy, error):
    if error is None:
        p = make_box(x, y, dx, dy)
        assert p.area == (2.0 * dx) * (2.0 * dy)
        assert p.area != 0.0
    else:
        with pytest.raises(error):
            make_box(x, y, dx, dy)


mask_ds_ext = extend_north(mask_ds, "mask", 210, np.nan)


@pytest.mark.slow
@pytest.mark.parametrize(
    "x, y, p, north_ext, upper_limit, error",
    [
        (
            mask_ds_ext.rlon.values[::10],
            mask_ds_ext.rlat.values[::10],
            rotated_canada,
            210,
            30,
            None,

        ),
        (
            mask_ds_ext.rlon.values[::10],
            mask_ds_ext.rlat.values[::10],
            bad_polygon,
            210,
            30,
            ValueError,
        ),
    ],
)
def test_gen_upper_archipelago_mask(x, y, p, north_ext, upper_limit, error):
    if error is None:
        result = gen_upper_archipelago_mask(p, x, y, north_ext, upper_limit)
        assert result.shape == (y.shape[0], x.shape[0])
    else:
        with pytest.raises(ValueError):
            gen_upper_archipelago_mask(p, x, y, north_ext, upper_limit)


@pytest.mark.slow
@pytest.mark.parametrize(
    "p, error",
    [
        (rotated_canada, None),
        (gpd.GeoSeries(), ValueError),
        (rotated_canada.geometry, None),
    ],
)
def test_stratify_coords(p, error):
    if error is None:
        X, Y = stratify_coords(p)
        assert X.shape == Y.shape, "Stratified coords different sizes."
    else:
        with pytest.raises(error):
            stratify_coords(p)
