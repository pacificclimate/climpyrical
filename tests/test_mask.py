import sys
import pytest
import geopandas as gpd
import numpy as np

sys.path.append('../modules/')
from mask import check_polygon, check_pre_proj, check_post_proj, check_coords

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
can_index = world[world.name == "Canada"].index
can_geom = world.loc[can_index, 'geometry']

@pytest.mark.parametrize('p,passed', [
    ({'string'}, False),
    ({5}, False),
    (can_geom, True),
    (world, False),
    (gpd.GeoSeries(), False)])
def test_check_polygon(p, passed):
    if passed:
        assert check_polygon(p)
    else:
        with pytest.raises((TypeError, ValueError)):
            check_polygon(p)

@pytest.mark.parametrize('p,passed', [
    ({'string'}, False),
    ({5}, False),
    (can_geom, True),
    (world, False),
    (gpd.GeoSeries(), False)])
def test_check_pre_proj(p, passed):
    if passed:
        assert check_pre_proj(p)
    else:
        with pytest.raises((TypeError, ValueError)):
            check_pre_proj(p)

@pytest.mark.parametrize('p,passed', [
    ({'string'}, False),
    ({5}, False),
    (can_geom, False),
    (world, False),
    (gpd.GeoSeries(), False)])
def test_check_post_proj(p, passed):
    if passed:
        assert check_post_proj(p)
    else:
        with pytest.raises((TypeError, ValueError)):
            check_post_proj(p)

@pytest.mark.parametrize('x,y,passed', [
    ('x', np.linspace(-24, 24, 155), False),
    (np.linspace(-24, 24, 155), 'y', False),
    (np.linspace(0, 10, 30), np.linspace(0, 10, 30), False),
    (np.linspace(-33.8800048828125, 33.8800048828125, 155), np.linspace(-28.59999656677246, 28.15999984741211, 130), True)])
def test_check_coords(x, y, passed):
    if passed:
        assert check_coords(x,y)
    else:
        with pytest.raises((ValueError, TypeError)):
            check_coords(x,y)

