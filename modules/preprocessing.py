import numpy as np
from datacube import read_data
from mask import mask, interpolated_mask
from interpolator import gen_new_coords, interpolate_ensemble

def flatten_ensemble(dv_field):
    """Flattens data cube into ensemble
    size x number of cells.
    --------------------------------
    Args:
        dv_field (xarray.DataArray): datacube
            containing design values of from
            an ensemble of CanRCM4 models
    Returns:
        ens_dv_field (numpy.ndarray): reshaped datacube into
            (number of grid cells) x (number of ensemble members)
    """

    ens_sz = dv_field.shape[0]
    n_grid_cells = dv_field.shape[1]*dv_field.shape[2]

    ens_dv_field = dv_field.reshape((ens_sz, n_grid_cells))

    return ens_dv_field

def generate_pseudo_obs(ens_arr, frac):
    """Randomly sample a range of integers.
    --------------------------------
    Args:
        y_obs (numpy.ndarray): array to be sampled
        frac (float): fractional size of y_obs
            to sample
    Returns:
        (numpy.ndarray): array containing indices
            (in the ensemble shape) of randomly
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

    return index

def get_interpolation(mask_path, data_path, dv, factor=10):
    """Interpolates mask and ensemble data
    to the desired factor
    -------------------------------------
    Args:
        mask_path (str): path to mask file
        data_path (str): path to ensemble file
        dv (str): name of design value as found
            in ensemble file
        factor (int): factor to increase spatial resolution
    Returns:
        dict: dictionary containing new and old
            coordinates as well as interpolated values
    """
    ds = read_data(data_path, dv)
    dv_field = ds[dv].values
    ens = flatten_ensemble(dv_field)
    coordict = gen_new_coords(ds['rlat'].values, ds['rlon'].values, factor)
    mask_dict = mask(mask_path, dv_field)
    imask_dict = interpolated_mask(mask_path, dv_field, coordict, factor)
    points = coordict['icoordens'][imask_dict['index']]
    interp_dict = interpolate_ensemble(dv_field, coordict, imask_dict, mask_dict, ens)

    return {**interp_dict, **coordict}