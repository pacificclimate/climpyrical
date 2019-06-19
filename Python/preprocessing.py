import numpy as np
import numpy.ma as ma
import xarray as xr

def load_land_mask(data_path):
    ds_mask = xr.open_dataset(data_path)
    mask = ma.masked_greater(ds_mask['sftlf'].values, 0.0).mask
    return mask


def mask_land_and_nan(dv_field, mask_land):
    nan_mask = np.apply_over_axes(
                            np.logical_or.reduce,
                            np.isnan(dv_field),
                            (0)
            )
    mask_master = ~np.logical_or(~mask_land, nan_mask)

    return mask_master


def mask_land_and_nan_ens_index(mask_master):
	mask_ens_master = (mask_master.reshape(mask_master.shape[0], mask_master.shape[1]*mask_master.shape[2]
                        )[0])

	idx = np.where(mask_ens_master==True)[0]
	return idx


def ens_flat(dv_field):
    """Flattens data cube into ensemble
    size x number of cells.
    --------------------------------
    Args:
        dv_field (xarray.DataArray): datacube
            containing design values of from
            an ensemble of CanRCM4 models
    Returns:
        dv_field (numpy.ndarray): reshaped 2-d array
            into number of cells x ensemble size
    """

    ens_sz = dv_field.shape[0]
    n_grid_cells = dv_field.shape[1]*dv_field.shape[2]

    dv_field = dv_field.reshape((ens_sz, n_grid_cells))

    return dv_field

def generate_pseudo_obs(ens_arr, frac):
    """Randomly sample a range of integers.
    --------------------------------
    Args:
        y_obs (numpy.ndarray): array to be sampled
        frac (float): fractional size of y_obs
            to sample
    Returns:
        (numpy.ndarray): array containing indices of
            sampled values
    """

    ens_sz = ens_arr.shape[0]

    # randomly select an ensemble member
    # to sample
    i = np.random.randint(0, ens_sz-1)
    y_obs = ens_arr[i, :]

    n_grid_cells = y_obs.shape[0]
    index = np.random.choice(np.arange(n_grid_cells),
                             int(frac*n_grid_cells))
    return y_obs[index]

def correct_extend_rlat_and_rlon_to_ens(rlat, rlon, lat_corr=42.5, lon_corr=-97.):

	rlon_ens = np.tile(rlon, rlat.shape[0])+lon_corr
	rlat_ens = np.repeat(rlat, rlon.shape[0])+lat_corr

	lat_lon_ens = list(zip(rlat_ens, rlon_ens))
	return np.asarray(lat_lon_ens)
