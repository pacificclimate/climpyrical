import numpy as np
from scipy import interpolate
import xarray as xr

def gen_new_coords(rlat, rlon, factor):
    """Generates coordinate arrays from original dataset
    Args:
        rlat, rlon (np.ndarrays): rotated pole coordinates from dataset
            (should have differing lengths)
        factor (int): factor to increase spatial resolution
    Returns:
        coordict (dict): dictionary containing various
            arrays and shapes of interpolated and original
            coordinates
    """
    # scales the rlon and rlat arrays by factor
    irlon = np.linspace(np.min(rlon), np.max(rlon), rlon.shape[0]*factor)
    irlat = np.linspace(np.min(rlat), np.max(rlat), rlat.shape[0]*factor)

    # convert to the flattened ensemble shape
    rlon_ens = np.tile(rlon, rlat.shape[0])
    rlat_ens = np.repeat(rlat, rlon.shape[0])
    coordens = np.array(list(zip(rlon_ens, rlat_ens)))

    # converts the interpolated coords to the flattened
    # ensemble shape
    irlon_ens = np.tile(irlon, irlat.shape[0])
    irlat_ens = np.repeat(irlat, irlon.shape[0])
    icoordens = np.array(list(zip(irlon_ens, irlat_ens)))

    coordict = {
            'irlat': irlat,
            'irlon': irlon,
            'rlon_ens': rlon_ens,
            'rlat_ens': rlat_ens,
            'irlon_ens': irlon_ens,
            'irlat_ens': irlat_ens,
            'icoordens': icoordens,
            'coordens': coordens
    }

    return coordict

def interpolate_ensemble(dv_field, coordict, imask_dict, mask_dict, ens):
    """Interpolates the data onto the new scaled coordinates
    Args:
        dv_field (numpy.ndarray): design value field in cubic shape
        coordict (dict): coordinates of interpolated and original data
        imask_dict (dict): interpolated mask dictionary
        mask_dict (dict): mask dictionary (non interpolated)
        ens (np.ndarray): design value field in flattened ensemble shape
    Returns:
        interpolated_dict (dict): dictionary containing all relevant
            and useful results of the interpolation including the
            indices of valid grid cells, interpolated design value field
            in flattened shape, and interpolated coordinates also in
            flattened shape.
    """
    idx = mask_dict['index']
    iidx = imask_dict['index']

    # new size of interpolated flattened shape
    isize = coordict['irlon'].shape[0]*coordict['irlat'].shape[0]

    # creates interpolated ensemble from interpolated mask dict
    iens = np.broadcast_to(
                np.reshape(imask_dict['master_interp'], (isize)),
                (dv_field.shape[0], isize)
    ).astype(float)

    # locations of areas to interpolate
    points = coordict['icoordens'][imask_dict['index']]

    # interpolate each ensemble member
    for i in range(dv_field.shape[0]):
        iens[i, iidx] = interpolate.griddata(coordict['coordens'][idx], ens[i, idx], points, method='linear')

    # apply nan mask to any poorly interpolated areas
    nan_mask = np.apply_over_axes(
                np.logical_or.reduce,
                np.isnan(iens),
                (0)
    )
    master_idx = ~np.logical_or(~imask_dict['master'], nan_mask).flatten()

    interpolate_dict = {
        'idx': master_idx,
        'iens': iens,
        'irlat_ens': coordict['irlat_ens'],
        'irlon_ens': coordict['irlon_ens']
    }

    return interpolate_dict
