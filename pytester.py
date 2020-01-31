from climpyrical.mask import *
from climpyrical.gridding import *
from climpyrical.datacube import *

PATH = './climpyrical/tests/data/snw_test_ensemble.nc'
dv = 'Rain-RL50'
ds = read_data(PATH, dv)

# interpolate to 3x3 grids
dx, dy = np.diff(ds.rlon.values).mean()/3, np.diff(ds.rlat.values).mean()/3
inrlon = np.linspace(ds.rlon.min()-dx, ds.rlon.max()+dx, ds.rlon.shape[0]*3)
inrlat = np.linspace(ds.rlat.min()-dy, ds.rlat.max()+dy, ds.rlat.shape[0]*3)
new_ens = np.ones((inrlat.size, inrlon.size))

new_ds = np.repeat(np.repeat(ds[dv].values, 3, axis=1), 3, axis=2)

ids = xr.Dataset({dv: (['level', 'y', 'x'], new_ds)}, 
                 coords={'rlon':  ('x', inrlon), 
                         'rlat': ('y', inrlat), 
                         'level': ('level', range(35))})
ds = ids

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
can_index = world[world.name == "Canada"].index
can_geom = world.loc[can_index, 'geometry']

canada = rotate_shapefile(can_geom)

mask = gen_raster_mask_from_vector(ds.rlon.values, ds.rlat.values, canada)

ens_mean = np.mean(ds[dv].values, axis=0)

# Create a mask of extra NaN values that is flattened
flat_mask_with_extra_nan = np.logical_and(~np.isnan(ens_mean.flatten()), mask.flatten())

# Put back into the spatial shape
final_mask = flat_mask_with_extra_nan.reshape(ens_mean.shape)

# create two grids of rlon and rlat to use new mask with
rlon, rlat = np.meshgrid(ds.rlon, ds.rlat)

# check that the shapes of all of the arrays after masking are consistent
assert rlat[final_mask].shape == rlon[final_mask].shape and ens_mean[final_mask].shape == rlat[final_mask].shape

df = pd.read_csv('./climpyrical/tests/data/stations.csv', index_col=None)
station_dv = 'RL50 (kPa)'

rlon_st, rlat_st = transform_coords(df['lon'].values, df['lat'].values)
df = df.assign(
        rlon=rlon_st, 
        rlat=rlat_st
)

df = df[df['RL50 (kPa)'] != 0.0]

# Add nearest rlon and rlat locations in grid to stations
rlon_nearest_index, rlat_nearest_index = find_element_wise_nearest_pos(ds.rlon.values, ds.rlat.values, df.rlon.values, df.rlat.values)
df = df.assign(
    label='stations',
    rlon_nearest_index = rlon_nearest_index, 
    rlat_nearest_index = rlat_nearest_index,
    nearest_grid = list(zip(rlon_nearest_index, rlat_nearest_index))
)

# Group together stations falling in same grid cell and take mean
ndf = df.groupby(['nearest_grid'], as_index=False).agg({
                                station_dv: 'mean',
                                'lat':'min',
                                'lon':'min',
                                'rlat': 'min',
                                'rlon': 'min',
                                'rlon_nearest_index':'mean',
                                'rlat_nearest_index':'mean',
                                'label': 'min',
                            })

# Get the indices for each grid axis that are matched and grouped
rlon_i_agg, rlat_i_agg = ndf.rlon_nearest_index.values, ndf.rlat_nearest_index.values

# Get the corresponding ensemble mean value at that particular location
# ndf = ndf.assign(
#     mean_values_at_real_stations=find_nearest_index_value(
#                                         ds.rlon.values, 
#                                         ds.rlat.values, 
#                                         rlon_i_agg, 
#                                         rlat_i_agg, 
#                                         ens_mean, 
#                                         final_mask, 
#                                         ds
#     )
# )

bases = [find_nearest_index_value(ds.rlon.values, ds.rlat.values, rlon_i_agg, rlat_i_agg, ds[dv].values[i, :, :], final_mask, ds) for i in range(35)]
print(np.array(bases).shape)
values = np.stack([ds.rlon.values[ndf.rlon_nearest_index], ds.rlat.values[ndf.rlat_nearest_index]])
#values = np.concatenate((values, ndf.rlon[ndf.rlon_nearest_index].values.reshape(1, -1)), axis=0)
# values = np.concatenate((values, ndf.rlat[ndf.rlat_nearest_index].values.reshape(1, -1)), axis=0)
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance
from scipy.stats import linregress
