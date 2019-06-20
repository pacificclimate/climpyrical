import numpy as np
from scipy import interpolate
import xarray as xr

def interp_can(ds, factor=10):

	rlon = ds['rlon'].values
	rlat = ds['rlat'].values
	dv_field = ds['dv'].values

	rlon_diff = np.diff(rlon).mean()/factor
	rlat_diff = np.diff(rlat).mean()/factor

	nrlon = np.arange(rlon.min(), rlon.max(), rlon_diff)
	nrlat = np.arange(rlat.min(), rlat.max(), rlat_diff)

	f = interpolate.interp2d(rlon, rlat, dv_field[0, ...], kind='linear')

	ds_interp = ((nrlon.shape[0], nrlat.shape[0]), f(nrlon, nrlat))

	return ds_interp