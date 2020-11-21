from climpyrical.data import (
    check_valid_keys,
    check_valid_data,
    read_data,
    interpolate_dataset,
    gen_dataset
)
import pytest
from pkg_resources import resource_filename
import xarray as xr
import numpy as np


@pytest.mark.parametrize(
    "actual_keys,required_keys,passed",
    [
        (
            ["rlat", "rlon", "dv", 4, "lat", "lon"],
            ["rlat", "rlon", "dv", 4],
            True,
        ),
        (["rlat", "rlon", True, 99999], ["rlat", "rlon", True, 99999], True),
        (
            ["rlat", "rlon", 4.0, False],
            ["hi", "nic", "was", "here", "lon"],
            False,
        ),
    ],
)
def test_check_valid_keys(actual_keys, required_keys, passed):
    if passed:
        assert check_valid_keys(actual_keys, required_keys)
    else:
        with pytest.raises(KeyError):
            check_valid_keys(actual_keys, required_keys)


# simulate an empty design value input
empty_ds = xr.Dataset({"empty": []}, coords={"rlon": 0, "rlat": 0})

# simulate an Null valued input
nan_ds = xr.Dataset(
    {"NaN": (["rlat", "rlon"], np.ones((10, 10)) * np.nan)},
    coords={"rlon": np.linspace(0, 10, 10), "rlat": np.linspace(0, 10, 10)},
)

# simulate unexpected coordinate value inputs
bad_coords = xr.Dataset(
    {"empty": [-10, 10, -10]},
    coords={
        "rlon": np.array([-10, np.nan, 10]),
        "rlat": np.array([-10, np.nan, 10]),
    },
)

non_mono_bad_coords = xr.Dataset(
    {"empty": [-10, 10, -10]},
    coords={"rlon": np.linspace(10, 0, 10), "rlat": np.linspace(10, 0, 10)},
)


@pytest.mark.parametrize(
    "ds,error",
    [
        (empty_ds, ValueError),
        (bad_coords, ValueError),
        (non_mono_bad_coords, ValueError),
        (nan_ds, ValueError),
        (
            xr.open_dataset(
                resource_filename("climpyrical", "tests/data/snw.nc")
            ),
            None,
        ),
        (
            xr.open_dataset(
                resource_filename("climpyrical", "tests/data/hdd.nc")
            ),
            None,
        ),
    ],
)
def test_valid_data(ds, error):
    # tests that xr.open_dataset does the same
    # thing as read_data for a few examples

    if error is None:
        assert check_valid_data(ds)
    else:
        with pytest.raises(error):
            check_valid_data(ds)


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_path,design_value_name,keys,expected",
    [
        (
            resource_filename("climpyrical", "tests/data/snw.nc"),
            "snw",
            ["rlat", "rlon", "lat", "lon"],
            KeyError
        ),
        (
            resource_filename("climpyrical", "tests/data/hdd.nc"),
            "heating_degree_days_per_time_period",
            ["rlat", "rlon", "lat", "lon", "extra"],
            KeyError
        ),
        (
            resource_filename("climpyrical", "tests/data/example2.nc"),
            "snw",
            ["rlat", "rlon", "lat", "lon"],
            (130, 155)
        ),
        (
            resource_filename("climpyrical", "tests/data/example3.nc"),
            "tas",
            ["lat", "lon"],
            KeyError
        ),
        (
            resource_filename("climpyrical", "tests/data/example4.nc"),
            "tos",
            ["lat", "lon"],
            KeyError
        )
    ],
)
def test_shape(data_path, design_value_name, keys, expected):
    # tests that the function loads a variety of test data
    # properly
    print("EXPECTED", expected)
    if isinstance(expected, tuple): 
        ds = read_data(data_path, keys)
        assert ds[design_value_name].shape == expected
    else:
        with pytest.raises(expected):
            read_data(data_path)


@pytest.mark.parametrize(
    "data_path,error",
    [
        (
            resource_filename("climpyrical", "tests/data/world.geojson"),
            TypeError,
        ),
        (resource_filename("climpyrical", "tests/data/example2.nc"), None),
    ],
)
def test_path(data_path, error):
    # tests path checker

    if error is None:
        read_data(data_path)
    else:
        with pytest.raises(error):
            read_data(data_path)


xx, yy = np.meshgrid(np.linspace(0, 50, 50), np.linspace(-25, 25, 50))
points = np.stack([xx.flatten(), yy.flatten()]).T
ones = np.ones(xx.shape).flatten()

xxn, yyn = np.meshgrid(np.linspace(0, 50, 100), np.linspace(-25, 25, 100))
target_points = np.stack([xxn.flatten(), yyn.flatten()]).T


@pytest.mark.parametrize(
    "points, values, target_points, method, error",
    [
        (points, ones, target_points, "linear", None),
        (points, ones, target_points, "nearest", None),
        (points, ones, target_points, "spaghetti", ValueError),
    ],
)
def test_interpolate_dataset(points, values, target_points, method, error):
    if error is None:
        np.testing.assert_allclose(
            np.ones(xxn.shape).flatten(),
            interpolate_dataset(points, values, target_points, method),
        )
    else:
        with pytest.raises(error):
            interpolate_dataset(points, values, target_points, method)


test_field = np.ones((2, 2))
@pytest.mark.parametrize(
    "dv, field, rlat, rlon, lat, lon",
    [
        ('test', test_field, [0, 1], [0, 1], test_field, test_field),
    ],
)
def test_gen_dataset(dv, field, rlat, rlon, lat, lon):
     ds = gen_dataset(dv, field, rlat, rlon, lat, lon)
     assert isinstance(ds, xr.Dataset)
     assert ds[dv].values.shape == test_field.shape
     assert lat.shape == test_field.shape
     assert lon.shape == test_field.shape
     assert len(rlat) == test_field.shape[0]
     assert len(rlon) == test_field.shape[1]
