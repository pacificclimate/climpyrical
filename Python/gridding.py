import numpy as np
from scipy.spatial import distance

def euclidean_dist_index(lat_lon_obs, lat_lon_ens):
    dist_list = []
    for i, coord in enumerate(lat_lon_obs):
        dist_list.append(
                        distance.cdist(lat_lon_ens,
                                       [coord],
                                       'euclidean').argmin()
        )
    return np.asarray(dist_list)
