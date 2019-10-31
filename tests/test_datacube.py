import sys
import pytest

sys.path.append('../modules/')
from datacube import read_data, check_keys


@pytest.mark.parametrize('actual_keys, required_keys,passed', [
    ({'rlat', 'rlon', 'dv', 4, 'lat', 'lon'}, {'rlat', 'rlon', 'dv', 4}, True),
    ({'rlat', 'rlon', True, 99999}, {'rlat', 'rlon', True, 99999}, True)])
def test_check_keys(actual_keys, required_keys, passed):
    checker = check_keys(actual_keys, required_keys)
    assert(checker == passed)

@pytest.mark.parametrize('actual_keys,required_keys,passed', [
    ({'rlat', 'rlon', 4.0, False}, {'hi', 'nic', 'was', 'here', 'lon'}, False)])
def test_bad_keys(actual_keys, required_keys, passed):
    with pytest.raises(KeyError):
        checker = check_keys(actual_keys, required_keys)

@pytest.mark.parametrize('data_path, design_value_name, keys, shape', [
    ('./data/good/snw.nc', 'snw', {'rlat', 'rlon', 'lat', 'lon'}, (66, 130, 155)),
    ('./data/good/hdd.nc', 'heating_degree_days_per_time_period', {'rlat', 'rlon', 'lat', 'lon', 'level'}, (35, 130, 155)),
    ('./data/bad/example1.nc', 'hyai', {'lon', 'lat'}, (27,)),
    ('./data/bad/example2.nc', 'tas', {'lat', 'lon'}, (1, 130, 155)),
    ('./data/bad/example3.nc', 'tas', {'lat', 'lon'}, (1, 128, 256)),
    ('./data/bad/example4.nc', 'tos', {'lat', 'lon'}, (24, 170, 180))])
@pytest.mark.xfail(raises=ValueError)
def test_shape(data_path, design_value_name, keys, shape):
    ds = read_data(data_path, design_value_name, keys)
    assert(ds[design_value_name].shape == shape)
