import pytest
import sys
sys.path.append('../scripts/')
from datacube import *

def test_read_data(data_path):
    ds = read_data(data_path)

    for var in ['dv', 'rlat', 'rlon', 'lat', 'lon']: 
        assert var in ds.variables, 'Key \'{}\' not properly inserted into Dataset'.format(var)

    assert len(ds['dv'].shape) == 3, 'Incorrect dimension size after loading'
