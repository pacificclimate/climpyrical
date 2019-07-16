import numpy as np
from scipy import interpolate
import xarray as xr

def regrid_coords(ds, factor):

	rlat, rlon = ds['rlat'].values, ds['rlon'].values

	nrlonsz = rlon.shape[0]*factor
	nrlatsz = rlat.shape[0]*factor

	nrlon = np.linspace(rlon.min(), rlon.max(), nrlonsz)
	nrlat = np.linspace(rlat.min(), rlat.max(), nrlatsz)

	coord = {
			'rlat': rlat,
			'rlon': rlon,
			'nrlat': nrlat,
			'nrlon': nrlon,
			'nrlatsz': nrlatsz,
			'nrlonsz': nrlonsz
	}

	return coord


def interp_can(ds, header='dv', method='linear', factor=10):

	coords = regrid_coords(ds, factor)

	dv_field = ds[header].values

	rlat, rlon = coords['rlat'], coords['rlon']

	nrlonsz = coords['nrlonsz']
	nrlatsz = coords['nrlatsz']

	nrlon = coords['nrlat']
	nrlat = coords['nrlon']

	rlon_ens = np.tile(rlon, rlat.shape[0])
	rlat_ens = np.repeat(rlat, rlon.shape[0])

	nrlon_ens = np.tile(nrlon, nrlat.shape[0])
	nrlat_ens = np.repeat(nrlat, nrlon.shape[0])

	rlat_rlon = np.array(list(zip(rlat_ens, rlon_ens)))
	nrlat_nrlon = np.array(list(zip(nrlat_ens, nrlon_ens)))

	ds_interp = np.zeros((dv_field.shape[0], nrlonsz, nrlatsz))

	flat = interpolate.interp2d(rlon, rlat, ds['lat'], kind='linear')
	flon = interpolate.interp2d(rlon, rlat, ds['lon'], kind='linear')

	if method != 'nearest':
		for i in range(dv_field.shape[0]):
			f = interpolate.interp2d(
								rlon,
								rlat,
								ds['dv'].values[i, ...],
								kind='linear'
				)

			ds_interp[i, ...] = f(nrlon, nrlat)

		new_ds = xr.Dataset(
						{'dv': (['run', 'rlat', 'rlon'], ds_interp)},
						coords={'lon': (['rlat', 'rlon'], flon(nrlon, nrlat)),
								'lat': (['rlat', 'rlon'], flat(nrlon, nrlat)),
								'rlat': nrlat,
								'rlon': nrlon,
								'run': range(ds_interp.shape[0])}
		)

		return new_ds


	elif method == 'nearest' and header=='sftlf':
		f = interpolate.NearestNDInterpolator(rlat_rlon, dv_field.reshape(-1, 1))
		ds_interp = f(nrlat_nrlon).reshape(nrlonsz, nrlatsz)
		ds_interp = np.expand_dims(ds_interp, 0)

		print(ds_interp.shape)

		new_ds = xr.Dataset(
						{'dv': (['rlat', 'rlon'], ds_interp[0, ...])},
						coords={'lon': (['rlat', 'rlon'], flon(nrlon, nrlat)),
								'lat': (['rlat', 'rlon'], flat(nrlon, nrlat)),
								'rlat': nrlat,
								'rlon': nrlon}
		)

	return ds_interp
