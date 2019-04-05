import numpy as np
import numpy.ma as ma
from sklearn.decomposition import pca
from sklearn import linear_model

from operators import check_keys, ens_means, frac_grid_area


def mask_nan(ds):
    """Gets mask for NaN values in
    first of CanRCM4 ensemble.
    --------------------------------
    Args:
        ds (xarray.Dataset): datacube
            containing an ensemble of CanRCM4 models
    Returns:
        mask (numpy.ma): masked array
    """
    mask = ma.masked_invalid(ds['dv'][0, :, :].values)
    return mask


def mask_flat(ds, mask):
    """Flattens mask into ensemble shape
    for compatibility.
    --------------------------------
    Args:
        ds (xarray.Dataset): datacube
            containing an ensemble of CanRCM4 models
        mask (numpy.ma): masked array for run in ds
    Returns:
        mask (nump.ma): masked array in ensemble shape,
        ensemble_size x number of grid cells
    """
    dv_field = ds['dv'].values
    n_grid_cells = dv_field.shape[1]*dv_field.shape[2]
    mask = np.reshape(mask, n_grid_cells)

    return mask


def ens_flat(ds):
    """Flattens data cube into ensemble
    size x number of cells.
    --------------------------------
    Args:
        ds (xarray Dataset): datacube
            containing an ensemble of CanRCM4 models
    Returns:
        dv_field (numpy.ndarray): reshaped 2-d array
            into number of cells x ensemble size
    """
    check_keys(ds)

    dv_field = ds['dv'].values

    ens_sz = dv_field.shape[0]
    n_grid_cells = dv_field.shape[1]*dv_field.shape[2]

    dv_field = np.reshape(dv_field, (ens_sz, n_grid_cells))

    return dv_field


def rand_sample_index(y_obs, frac):
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
    n_grid_cells = y_obs.shape[0]
    index = np.random.choice(np.arange(n_grid_cells),
                             int(frac*n_grid_cells))
    return index


def get_obs(ens_arr):
    """Randomly sample a data cube to generate
    pseudo observations from ensemble.
    --------------------------------
    Args:
        ds (xarray Dataset): datacube
            containing an ensemble of
            CanRCM4 models
    Returns:
        y_sample (numpy.ndarray): fraction
            of random spatially sampled design
            value grid cells
        index (numpy.ndarray): index locations
            of y_sample from original ensemble size
    """
    ens_sz = ens_arr.shape[0]

    # randomly select an ensemble member
    # to sample
    i = np.random.randint(0, ens_sz-1)
    y_obs = ens_arr[i, :]

    return y_obs


def ens_to_eof(ens_arr):
    """Perform EOF/PCA dimensionality reduction
    on ensemble array
    --------------------------------
    Args:
        ens_arr (numpy.ndarray): reshaped 2-d array
            into number of cells x ensemble size
    Returns:
        eofs (numpy.ndarray): transformed EOFs of ens_arr
    """

    skpca = pca.PCA(0.95)
    eofs = skpca.fit_transform(ens_arr)

    return eofs


def regress_eof(eofs, obs):
    """Perform a linear regression between
    EOFs and observations
    --------------------------------
    Args:
        eofs (numpy.ndarray): EOF transformed
            ens_arr containing same grid cells
            sample as observations
        obs (numpy.ndarray): Gridded observations
            either from pseudo or real observations
    Returns:
        model (scikit-learn model obj): fitted
            model object to eofs and obs
    """
    lm = linear_model.LinearRegression()
    eofs = eofs.reshape(-1, 1)
    model = lm.fit(eofs, obs)
    print("Model score:", model.score(eofs, obs))
    return model


def predict_dv(model, eofs_of_model):
    """Reconstruct design value field from
    full EOF sample
    --------------------------------
    Args:
        model (scikit-learn model): model of regressed obs
            and models
        eofs_of_model (numpy.ndarray): EOF transformed
            ens_arr
    Returns:
        (numpy.ndarray): predicted design
            value field
    """
    eofs_of_model = eofs_of_model.reshape(-1, 1)
    return model.predict(eofs_of_model)


def pred_to_grid(ds, pred, mask):
    """Reshape from flattened ensemble size x
    number of cells to rlat x rlon grid space
    --------------------------------
    Args:
        ds (xarray.Dataset): datacube
            containing an ensemble of
            CanRCM4 models
        pred (numpy.ndarray): reconstructed
            predictions from EOF regression
        mask (numpy.ma): mask used to mask
            values not considered
    Returns:
        dv_field (xarray.DataArray): reconstructed design
            value field
    """

    dv_field = ds['dv'][0, :, :]
    dv_field.values[~mask.mask] = pred

    return dv_field


def eof_pseudo_full(ds, mask=None):
    """Perform all steps to reconstruct a
    design value field from pseudo observations
    ---------------
    Args:
        ds (xarray.Dataset): datacube
            containing an ens of CanRCM4 models
    Returns:
        ds (xarray.DataArray): datacube
            containing an ens of CanRCM4 models
            with added eofs variable
    """

    mean = ens_means(ds)
    area = frac_grid_area(ds)
    if mask is None:
        mask = mask_flat(ds, mask_nan(ds))

    # area weighted ens to get obs
    ens_obs = ens_flat(ds*area)

    # mask
    ens_obs = ens_obs[:, ~mask.mask]

    # get random ens to generate
    # the pseudo obs
    obs = get_obs(ens_obs)
    obs_idx = rand_sample_index(obs, 0.02)
    obs_sample = obs[obs_idx]

    ens = ens_flat(ds*area - mean)
    ens = ens[:, ~mask.mask]
    ens = ens_to_eof(ens.T)[:, 0]

    model = regress_eof(ens[obs_idx], obs_sample)
    pred = predict_dv(model, ens)

    pred = pred_to_grid(ds, pred, mask_nan(ds))
    pred = (pred/area + mean)

    return pred
