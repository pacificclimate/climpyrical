import pytest
import sys
sys.path.append('../scripts/')
from datacube import *

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
    missing_value = False

    ds = read_data(data_path)

    for var in ['dv', 'rlat', 'rlon', 'lat', 'lon']:
        if var not in ds.variables:
            missing_value = True
    assert(missing_value == expected_mv), 'A key was not properly inserted into the Dataset'

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

'''@pytest.mark.parametrize('data_path, dim', [
    ('./data/good/wind/*.nc', 3),
    ,
    ('./data/good/pr/*.nc', 3),
    ('./data/bad/example1/*.nc', True),
    ('./data/bad/example2/*.nc', True),
    ('./data/bad/example3/*.nc', True)])

def test_shape(ds, dim):
	ds = read_data(data_path)
    assert(len(ds['dv'].shape) == 3), 'Incorrect dimension size after loading'
'''