from climpyrical.datacube import (
    check_valid_keys,
    check_valid_data,
    read_data,
    check_valid_data_path,
)
import pytest
from pkg_resources import resource_filename
import xarray as xr
import numpy as np


@pytest.mark.parametrize(
    "data_path,passed",
    [
        (resource_filename("climpyrical", "tests/data/snw.nc"), True),
        (4, False),
        (resource_filename("climpyrical", "tests/data/maskarray.npy"), False),
        # ("tests/data/bananas.nc", False)
    ],
)
def test_check_valid_data_path(data_path, passed):
    if passed:
        assert check_valid_data_path(data_path)
    else:
        with pytest.raises((KeyError, TypeError, EnvironmentError)):
            check_valid_data_path(data_path)


@pytest.mark.parametrize(
    "actual_keys,required_keys,passed",
    [
        (
            {"rlat", "rlon", "dv", 4, "lat", "lon"},
            {"rlat", "rlon", "dv", 4},
            True,
        ),
        ({"rlat", "rlon", True, 99999}, {"rlat", "rlon", True, 99999}, True),
        (
            {"rlat", "rlon", 4.0, False},
            {"hi", "nic", "was", "here", "lon"},
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
    "ds,design_value_name,passed",
    [
        (empty_ds, "empty", False),
        (bad_coords, "empty", False),
        (non_mono_bad_coords, "empty", False),
        (
            xr.open_dataset(
                resource_filename("climpyrical", "tests/data/snw.nc")
            ),
            "snw",
            True,
        ),
        (
            xr.open_dataset(
                resource_filename("climpyrical", "tests/data/hdd.nc")
            ),
            "heating_degree_days_per_time_period",
            True,
        ),
    ],
)
def test_valid_data(ds, design_value_name, passed):
    # tests that xr.open_dataset does the same
    # thing as read_data for a few examples

    if passed:
        assert check_valid_data(ds, design_value_name)
    else:
        with pytest.raises(ValueError):
            check_valid_data(ds, design_value_name)


@pytest.mark.parametrize(
    "data_path,design_value_name,keys,shape",
    [
        (
            resource_filename("climpyrical", "tests/data/snw.nc"),
            "snw",
            {"rlat", "rlon", "lat", "lon"},
            (66, 130, 155),
        ),
        (
            resource_filename("climpyrical", "tests/data/hdd.nc"),
            "heating_degree_days_per_time_period",
            {"rlat", "rlon", "lat", "lon", "level"},
            (35, 130, 155),
        ),
        (
            resource_filename("climpyrical", "tests/data/example1.nc"),
            "hyai",
            {"lon", "lat"},
            (27,),
        ),
        (
            resource_filename("climpyrical", "tests/data/example2.nc"),
            "tas",
            {"lat", "lon"},
            (1, 130, 155),
        ),
        (
            resource_filename("climpyrical", "tests/data/example3.nc"),
            "tas",
            {"lat", "lon"},
            (1, 128, 256),
        ),
        (
            resource_filename("climpyrical", "tests/data/example4.nc"),
            "tos",
            {"lat", "lon"},
            (24, 170, 180),
        ),
    ],
)
def test_shape(data_path, design_value_name, keys, shape):
    # tests that the function loads a variety of test data
    # properly
    ds = read_data(data_path, design_value_name, keys)
    assert ds[design_value_name].shape == shape


@pytest.mark.parametrize(
    "a,b",
    [
        (
            read_data(
                resource_filename("climpyrical", "tests/data/snw.nc"),
                "snw",
                {"rlat", "rlon", "lat", "lon"},
            ),
            xr.open_dataset(
                resource_filename("climpyrical", "tests/data/snw.nc")
            ),
        ),
        (
            read_data(
                resource_filename("climpyrical", "tests/data/hdd.nc"),
                "heating_degree_days_per_time_period",
                {"rlat", "rlon", "lat", "lon", "level"},
            ),
            xr.open_dataset(
                resource_filename("climpyrical", "tests/data/hdd.nc")
            ),
        ),
    ],
)
def test_xarray_open_dataset(a, b):
    # tests that xr.open_dataset does the same
    # thing as read_data for a few examples
    xr.testing.assert_identical(a, b)
