import numpy as np
from sklearn.decomposition import pca 

from operators import check_keys, cell_count,
					  center_data, weight_by_area

def eof_ensemble(data_cube):
	check_keys(data_cube)
	Y = data_cube['dv']
	return

def pseudo_obs():
	return

def land_mask(data_cube):
	check_keys(data_cube)
	is_land = 
