import pytest
import sys
sys.path.append('../scripts/')
from datacube import *

@pytest.mark.parametrize('data_path,expected_mv', [
    ('./data/good/wind/*.nc', False),
    ('./data/good/pr/*.nc', False),
    ('./data/bad/example1/*.nc', True),
    ('./data/bad/example2/*.nc', True),
    ('./data/bad/example3/*.nc', True)])
@pytest.mark.xfail(raises=ValueError)
def test_path(data_path, expected_mv):
    missing_value = False

    ds = read_data(data_path)

    for var in ['dv', 'rlat', 'rlon', 'lat', 'lon']:
        if var not in ds.variables:
            missing_value = True
    assert(missing_value == expected_mv), 'A key was not properly inserted into the Dataset'
    assert(len(ds['dv'].shape) == 3), 'Incorrect dimension size after loading'

@pytest.mark.parametrize('data_path', [
    ('./data/bad/tas/*.nc')])
@pytest.mark.xfail(raises=ValueError)
def test_time_dim(data_path):
	ds = read_data(data_path)