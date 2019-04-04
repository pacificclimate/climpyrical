import numpy as np
import numpy.ma as ma
from sklearn.decomposition import pca 
from sklearn import linear_model

from operators import check_keys, cell_count, center_data, weight_by_area

def masked_nan(ds):
    """Gets indices of nan values to 
    identify where land or ocean is. 
    --------------------------------
    Args: 
        ensemble_arr (numpy.ndarray): reshaped 2-d array
            into number of cells x ensemble size
    Returns:
        (numpy.ndarray): 2-d array containing
            the indices of nan values
    """
    #mask = ma.masked_values(ds['dv'][0,:,:], np.nan)
    mask = ma.masked_values(ds['dv'], np.nan)
    return mask

def ensemble_reshape(ds, mask=None):
    """Reshapes data cube into array shape 
    for eof calculations. 
    --------------------------------
    Args: 
        ds (xarray Dataset): datacube 
            containing an ensemble of 
            CanRCM4 models
        mask (numpy.ma.masked_invalid): masked array
            of a single ensemble member grid. This mask
            will usually be either an ocean or land mask. 
    Returns:
        dv_field (numpy.ndarray): reshaped 2-d array
            into number of cells x ensemble size
    """
    check_keys(ds)
    if mask is None:
        mask = ma.masked_invalid(ds['dv'][0,:,:].values)

    if not isinstance(mask, np.ma.core.MaskedArray):
        raise ValueError("Please provide a mask of type {}".format(np.ma.core.MaskedArray))

    if mask.shape != ds['dv'][0,:,:].shape:
        raise ValueError("Data and mask have incompatible shapes with {} and {}"
                         .format(mask.shape, ds['dv'][0,:,:].shape))

    dv_field = ds['dv'].values

    ensemble_sz = dv_field.shape[0]
    n_grid_cells = dv_field.shape[1]*dv_field.shape[2]

    mask = np.reshape(mask, n_grid_cells)
    dv_field = np.reshape(dv_field, (ensemble_sz, n_grid_cells))

    dv_field = dv_field[:, ~mask.mask]

    return dv_field


def rand_sample_index(y_obs, frac):
    """Randomly sample a range of integers. 
    --------------------------------
    Args: 
        p (int): maximum integer to sample
        frac (float): fractional size of p 
            to sample
    Returns:
        (numpy.ndarray): array containing indices from 
            a fraction of 0 to p
    """
    n_grid_cells = y_obs.shape[0]
    index = np.random.choice(np.arange(n_grid_cells),
                             int(frac*n_grid_cells))
    return index

def get_obs(ds):
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
    Y_ens = ensemble_reshape(ds)
    
    ensemble_sz = Y_ens.shape[0]
    n_grid_cells = Y_ens.shape[1]

    # randomly select an ensemble member
    # to spatially sample
    i = np.random.randint(0, n_grid_cells-1)
    y_obs = Y_ens[:, i]

    return y_obs

def ensemble_to_eof(ensemble_arr):
    """Perform EOF/PCA dimensionality reduction
    on ensemble array 
    --------------------------------
    Args:  
        ensemble_arr (numpy.ndarray): reshaped 2-d array
            into number of cells x ensemble size
    Returns:
        eofs (numpy.ndarray): EOF transformed
            ensemble_arr sorted by max variance 
            explained
    """
    
    skpca = pca.PCA()
    
    # transform to get proper shape for EOF dim reduction
    skpca.fit_transform(ensemble_arr.T)
    
    # transform back to familiar shape n x p
    eofs = skpca.components_.T
    
    return eofs

def regress_eof(eofs, obs):
    """Perform a linear regression between
    EOFs and observations 
    --------------------------------
    Args:  
        eofs (numpy.ndarray): EOF transformed
            ensemble_arr containing same grid cells 
            sample as observations 
        obs (numpy.ndarray): Gridded observations
            either from pseudo or real observations
    Returns:
        model (scikit-learn model obj): fitted 
            model object to eofs and obs
    """
    lm = linear_model.LinearRegression()
    model = lm.fit(eofs, obs)
    print("Model score:", model.score(eofs, obs))
    return model

def predict_dv(model, eofs_of_model):
    """Reconstruct design value field from
    full EOF sample 
    --------------------------------
    Args:  
        model (scikit-learn model obj): fitted 
            model object to eofs and obs
        eofs_of_model (numpy.ndarray): EOF transformed
            ensemble_arr containing all grid
            cells in design value field
    Returns:
        (numpy.ndarray): predicted design
            value field
    """
    return model.predict(eofs_of_model)

def reconstruct_eof_full(predictions, ds):
    """Add EOF reconstructed predictions
    to all grid cells of CanRCM4 model and 
    add to data cube 
    --------------------------------
    Args:  
        predictions (numpy.ndarray): predicted design
            value field
        ds (xarray Dataset): datacube 
            containing an ensemble of CanRCM4 models
    Returns:
        ds (xarray Dataset): datacube 
            containing an ensemble of CanRCM4 models
            with added eofs variable
    """
    full_arr = ensemble_reshape(ds)
    land_idx = get_land(full_arr)
    # replace land values with predictions
    full_arr[~land_idx.any(axis=1)][:, 0] = predictions
    # add predictions to data cube
    ds['eofs'] = full_arr[:, 0]

    return ds

def fit_transform(ds):
    """Perform all steps to reconstruct a 
    design value field from pseudo observations
    ---------------
    Args: 
        ds (xarray Dataset): datacube 
            containing an ensemble of CanRCM4 models 
    Returns:
        ds (xarray Dataset): datacube 
            containing an ensemble of CanRCM4 models
            with added eofs variable
    """
    y = get_obs(ds)

    # select random indices
    index = rand_sample_index(y, frac)
    y = y[index]

    ds = center_data(ds)
    ds = weight_by_area(ds)

    ensemble = ensemble_reshape(ds)
    ensemble_eof = ensemble_to_eof(ensemble)

    ens_cross_eof = ensemble_eof[index, :] 

    model = regress_eof(ens_cross_eof, y)
    predictions = predict_dv(model, ensemble_eof)

    return reconstruct_eof_full(predictions, ds)
