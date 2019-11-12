import os
import pytest
from climpyrical.datacube import read_data
from climpyrical.datacube import check_keys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    'actual_keys,required_keys,passed',
    [
        (
            {'rlat', 'rlon', 'dv', 4, 'lat', 'lon'},
            {'rlat', 'rlon', 'dv', 4},
            True
        ),
        (
            {'rlat', 'rlon', True, 99999},
            {'rlat', 'rlon', True, 99999},
            True
        ),
        (
            {'rlat', 'rlon', 4.0, False},
            {'hi', 'nic', 'was', 'here', 'lon'},
            False
        )
    ]
)
def test_check_keys(actual_keys, required_keys, passed):
    if passed:
        assert check_keys(actual_keys, required_keys)
    else:
        with pytest.raises(KeyError):
            check_keys(actual_keys, required_keys)


@pytest.mark.parametrize(
    'data_path,design_value_name,keys,shape',
    [
        (
            os.path.join(THIS_DIR, './data/snw.nc'),
            'snw',
            {'rlat', 'rlon', 'lat', 'lon'},
            (66, 130, 155)
        ),
        (
            os.path.join(THIS_DIR, './data/hdd.nc'),
            'heating_degree_days_per_time_period',
            {'rlat', 'rlon', 'lat', 'lon', 'level'},
            (35, 130, 155)
        ),
        (
            os.path.join(THIS_DIR, './data/example1.nc'),
            'hyai',
            {'lon', 'lat'},
            (27,)
        ),
        (
            os.path.join(THIS_DIR, './data/example2.nc'),
            'tas',
            {'lat', 'lon'},
            (1, 130, 155)
        ),
        (
            os.path.join(THIS_DIR, './data/example3.nc'),
            'tas',
            {'lat', 'lon'},
            (1, 128, 256)
        ),
        (
            os.path.join(THIS_DIR, './data/example4.nc'),
            'tos',
            {'lat', 'lon'},
            (24, 170, 180)
        )
    ]
)
def test_shape(data_path, design_value_name, keys, shape):
    ds = read_data(data_path, design_value_name, keys)
    assert(ds[design_value_name].shape == shape)
