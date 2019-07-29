import pytest
import sys
sys.path.append('../scripts/')
from datacube import read_data, check_keys, check_path,check_time

@pytest.mark.parametrize('actual_keys,required_keys,passed', [
    ({'rlat', 'rlon', 'dv', 4}, {'rlat', 'rlon', 'dv', 4, 'lat', 'lon'}, True),
    ({'rlat', 'rlon', True, 99999}, {'rlat', 'rlon', True, 99999}, True),
    ({'rlat', 'rlon', 4.0, False}, {'hi', 'nic', 'was', 'here', 'lon'}, False)])
@pytest.mark.xfail(raises=KeyError)
def test_check_keys(actual_keys, required_keys, passed):
	checker = check_keys(actual_keys, required_keys)
	assert(checker == passed)

@pytest.mark.parametrize('data_path,passed', [
    (45000, False),
    ('./data/good/pr/*.nc', True),
    ('./data/good/hdd/*.nc', True),
    ('./data/bad/empty/*.nc', False),
    ('./data/bad/example2/*.nc', False),
    ('./data/bad/example3/*.nc', False)])
@pytest.mark.xfail(raises=ValueError)
def test_check_path(data_path, passed):
	checker = check_path(data_path)
	assert(checker == passed)

@pytest.mark.parametrize('len_time,passed', [
    (1, True),
    (0.1, False),
    ('s', False),
    (str, True)])
@pytest.mark.xfail(raises=ValueError)
def test_check_time(len_time, passed):
	checker = check_time(len_time)
	assert(checker==passed)

@pytest.mark.parametrize('data_path,expected_mv', [
    ('./data/good/wind/*.nc', False),
    ('./data/good/pr/*.nc', False),
    ('./data/good/hdd/*.nc', False),
    ('./data/bad/example1/*.nc', True),
    ('./data/bad/example2/*.nc', True),
    ('./data/bad/example3/*.nc', True)])
@pytest.mark.xfail(raises=KeyError)
@pytest.mark.xfail(raises=ValueError)
def test_path(data_path, expected_mv):
    ds = read_data(data_path)
    for var in ['dv', 'rlat', 'rlon', 'lat', 'lon']:
    	assert ((var not in ds.variables) == expected_mv)

@pytest.mark.parametrize('data_path,time_dim', [
    ('./data/bad/tas/*.nc', 1)])
@pytest.mark.xfail(raises=ValueError)
def test_time_dim(data_path, time_dim):
	ds = read_data(data_path)
	assert(len(ds['time'])==time_dim)

@pytest.mark.parametrize('data_path,nfiles', [
    ('./data/bad/empty/*.nc', 0),
    ('./data/good/wind/*.nc', 8),
    ('./data/good/pr/*.nc', 1),
    ('./data/good/hdd/*.nc', 11),
    ('./data/bad/example2/*.nc', 1),
    ('./data/bad/example1/*.nc', 1),
    ('./data/bad/example3/*.nc', 1),
    ('./data/bad/tas/*.nc', 1)])
@pytest.mark.xfail(raises=ValueError)
def test_shape(data_path, nfiles):
	ds = read_data(data_path)
	assert(ds['dv'].shape[0] == nfiles)
