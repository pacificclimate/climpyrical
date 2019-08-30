import numpy as np
from scipy import interpolate
import xarray as xr

def gen_new_coords(rlat, rlon, factor):

    irlon = np.linspace(np.min(rlon), np.max(rlon), rlon.shape[0]*factor)
    irlat = np.linspace(np.min(rlat), np.max(rlat), rlat.shape[0]*factor)

    rlon_ens = np.tile(rlon, rlat.shape[0])
    rlat_ens = np.repeat(rlat, rlon.shape[0])

    irlon_ens = np.tile(irlon, irlat.shape[0])
    irlat_ens = np.repeat(irlat, irlon.shape[0])
    icoordens = np.array(list(zip(irlon_ens, irlat_ens)))

    coordens = np.array(list(zip(rlon_ens, rlat_ens)))

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

    idx = mask_dict['index']
    iidx = imask_dict['index']

    i, j = coordict['irlon'].shape[0], coordict['irlat'].shape[0]
    iens = np.broadcast_to(
                np.reshape(imask_dict['master_interp'], (i*j)),
                (dv_field.shape[0], i*j)
    ).astype(float)

    points = coordict['icoordens'][imask_dict['index']]

    for i in range(dv_field.shape[0]):
        iens[i, iidx] = interpolate.griddata(coordict['coordens'][idx], ens[i, idx], points, method='linear')

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
