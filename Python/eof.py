import numpy as np
from sklearn.decomposition import pca 
from sklearn import linear_model

from operators import *

def ensemble_reshape(data_cube):
    check_keys(data_cube)
    x_field = data_cube['dv']
    n, p = cell_count(data_cube)
    field_ens = np.reshape(x_field.values, (p, n))
    return field_ens

def get_land(ensemble_arr):
    return np.isnan(ensemble_arr)

def rand_sample_index(p, frac):
    index = np.random.choice(np.arange(p), int(frac*p))
    return index

def pseudo_obs(data_cube, frac=0.02):
    
    Y_ens = ensemble_reshape(data_cube)
    land_idx = get_land(Y_ens)
    Y_land = Y_ens[~land_idx.any(axis=1)]

    p, n = Y_land.shape
    i = np.random.randint(0, n-1)
    y_land = Y_land[:, i]

    index = rand_sample_index(p, frac)
    y_sample = y_land[index]

    return y_sample, index

def ensemble_to_eof(ensemble_arr):
    land_idx = get_land(ensemble_arr)
    ensemble_arr = ensemble_arr[~land_idx.any(axis=1)]
    skpca = pca.PCA()
    # transform to get proper shape for EOF dim reduction
    
    skpca.fit_transform(ensemble_arr.T)
    # transform back to familiar shape n x p
    eofs = skpca.components_.T
    return eofs

def regress_eof(eofs, obs):
    lm = linear_model.LinearRegression()
    model = lm.fit(eofs, obs)
    print(model.score(eofs, obs))
    return model

def predict_dv(model, eofs_of_model):
    return model.predict(eofs_of_model)

def reconstruct_eof_full(predictions, data_cube):

    full_arr = ensemble_reshape(data_cube)
    land_idx = get_land(full_arr)
    full_arr[~land_idx.any(axis=1)][:, 0] = predictions
    
    data_cube['eofs'] = full_arr[:, 0]

    return data_cube

def fit_transform(data_cube):

    y, index = pseudo_obs(data_cube)

    ensemble = ensemble_reshape(data_cube)
    ensemble_eof = ensemble_to_eof(ensemble)

    ens_cross_eof = ensemble_eof[index, :] 

    model = regress_eof(ens_cross_eof, y)
    predictions = predict_dv(model, ensemble_eof)

    return reconstruct_eof_full(predictions, data_cube)