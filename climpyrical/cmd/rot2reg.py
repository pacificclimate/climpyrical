import sys
from climpyrical.data import read_data
from climpyrical.gridding import rot2reg
import warnings

warnings.filterwarnings("ignore")

"""
quick usage of climpyrical.rot2reg
usage:
python rot2reg input.nc output.nc
"""

IN_PATH = sys.argv[1]
OUT_PATH = sys.argv[2]

ds = read_data(IN_PATH)

lonlat_proj = {
    "proj": "longlat",
    "ellps": "WGS84",
    "datum": "WGS84",
    "no_defs": True,
}

rotated_proj = {
    "proj": "ob_tran",
    "o_proj": "longlat",
    "lon_0": -97,
    "o_lat_p": 42.5,
    "a": 6378137,
    "to_meter": 0.0174532925199,
    "no_defs": True,
}

reg_ds = rot2reg(ds)

reg_ds.to_netcdf(OUT_PATH)
