#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Standard lib
import sys
import glob
                
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter
import geopandas as gpd
from shapely.geometry import Point

import seaborn as sns
# Adjust plot font sizes
sns.set(font_scale=1.5)
sns.set_style("white")

import cartopy.crs as ccrs
import cartopy.feature
import pandas as pd

from sklearn import preprocessing

from scipy.interpolate import interp2d

from sklearn.decomposition import pca

import netCDF4 as nc


# In[3]:


# Set path to RCM data
PATH = './data/pr_24hr_50yrs/'
PATH_SUP = './support/'


# In[4]:


# Create a list of all files in PATH
nc_list = np.asarray(glob.glob(PATH+"*"))


# In[5]:


nc.Dataset(nc_list[0])


# In[1]:


# Create a list of Dataset objects using netCDF4
dataset_list = np.asarray([nc.Dataset(path) for path in nc_list])


# In[6]:


dataset_list.shape, nc_list.shape


# In[7]:


# Choose one random instance
# -------------------------- #
indx = np.random.randint(0, dataset_list.shape[0])
inst = dataset_list[indx]


# In[8]:


type(inst.variables)


# In[9]:


inst.variables['lat']


# In[10]:


rlat = inst.variables['rlat'][:]
rlon = inst.variables['rlon'][:]
print('rlat', rlat.shape, 'rlon', rlon.shape)


# In[11]:


# Create list of parameters in dataset list
#print(dataset_list[2].variables['lat'])
#lis = [full for full in dataset_list]
print(type(inst.variables['lat']))
for full in dataset_list:
    lat = full.variables['lat'][:, :]
    lon = full.variables['lon'][:, :]
    pr = full.variables['pr'][:, :]
    
#lat = [full.variables['lat'] for full in dataset_list]
#lon = [full.variables['lon'][:, :] for full in dataset_list]
#pr = [full.variables['pr'][:, :] for full in dataset_list]

# Construct 3D pandas dataframe with each data
# file as a layer in the 3rd dimension
pr={}
for full in dataset_list:
    pr[full] = full.variables['pr'][:, :]

df_pr = pd.Panel(pr)

# Create tuple of 3d df and rotated lat/lon
CanRCM4 = {'df_pr': df_pr, 
           'rlat': rlat, 
           'rlon': rlon
          }


# In[12]:


dfx = CanRCM4['df_pr'][dataset_list[1]].T

prx = {'pr': dfx.values.flatten(),
       'rlon': np.tile(CanRCM4['rlon'], len(rlat)),
       'rlat': np.tile(CanRCM4['rlat'], len(rlon))
      }

idx = np.argwhere(~np.isnan(prx['pr']))[:,0]

print(prx['pr'][idx])

prx_maskd = {'pr': prx['pr'][idx],
            'rlon': prx['rlon'][idx],
            'rlat': prx['rlat'][idx]
            }


# In[13]:


df_prx = pd.DataFrame.from_dict(prx_maskd).sample(frac=0.02)


# In[14]:


### START OF CLASS

# Finds closest value to res in array
resLibrary = [50, 25, 10, 5, 1]
resLib = resLibrary[np.argmin(np.abs(np.subtract(res, res_library)))]
resFactor = resLib/res

#method <- match.arg(method)

# organize (35) maps to (35) vectors
xtreme = np.asarray(CanRCM4['df_pr'])
rlon = CanRCM4['rlon']
rlat = CanRCM4['rlat']


# In[15]:


# Get dimensions of input
xtreme_dim = xtreme.shape
nrlon = rlon.shape[0]
nrlat = rlat.shape[0]
nrun = xtreme.shape[0] # Number of files

# initialize empty array
xtremeMatrix = np.zeros((nrun, nrlon*nrlat))

# Sklearn prep
lons, lats = np.meshgrid(rlon, rlat)


# In[16]:


# Reshape 3D array into 2D combined array, i.e. 35 x 18476.
# Each value in array is pr
X = np.reshape(xtreme, (xtreme.shape[0], nrlat * nrlon), order='F')

# Wherever there is NaN in array, that is a non-land value. 
# Create masked array
X = ma.masked_array(X, np.isnan(X))

# Mask the ocean points
ocean = X.sum(0).mask

# Declare anything not masked as land
# i.e. Compliment of ocean is land values in
# boolean masked array
land = ~ocean

# Get land values only from array
X = X[:, land]


# In[ ]:


# Standardize data
scaler  = preprocessing.StandardScaler()
scaler_sst = scaler.fit(X)

# Apply standardization
X = scaler_sst.transform(X)

# Should be gaussian centered on zero
self.mean = X.mean()
self.std = X.std()

# Find principal components of our array
skpca = pca.PCA()
skpca.fit(X)

# Number of PC to explain 70% of variance
ipc = np.where(skpca.explained_variance_ratio_.cumsum() >= 0.70)[0][0]

PCs = skpca.transform(X)

# Get Empirical Orthogonal Functions components
EOFs = skpca.components_

# Select EOFs up to 70% of explained variance
EOFs = EOFs[:ipc, :] 
self.eof_shape = EOFs.shape

# Create a new "dimension-reduced" PCA'd array
EOF_recons = np.ones((ipc, len(rlat) * len(rlon))) * -999.
for i in range(ipc): 
    # Fill each land portion of grid with
    # transformed principal components up to ipc
    EOF_recons[i,land] = EOFs[i,:] 

EOF_recons = ma.masked_values(np.reshape(EOF_recons, (ipc, len(rlat), len(rlon)), order='F'), -999.)

EOF_recons *= 100

self.EOF_recons = EOF_recons

proj = "+proj=ob_tran +o_proj=longlat +lon_0=-97 +o_lat_p=42.5 +a=1 +to_meter=0.0174532925199 +no_defs"

rp = ccrs.RotatedPole(pole_longitude=-97 - 180,
                      pole_latitude=42.5,
                      globe=ccrs.Globe(semimajor_axis=6370000,
                                       semiminor_axis=6370000))
proj = {'proj':'ob_tran', 'o_proj':'longlat', 'lon_0':-97, 'o_lat_p':42.5,
        'a':1, 'to_meter':0.0174532925199}

rlon_itrp = np.linspace(np.min(rlon), np.min(rlon), len(rlon)*10)
rlat_itrp = np.linspace(np.min(rlat), np.min(rlat), len(rlat)*10)

Rlon, Rlat = np.meshgrid(rlon, rlat)

f = interp2d(Rlon, Rlat, EOF_recons[0, :, :])
EOF_new = f(rlon_itrp, rlat_itrp)
self.EOF_new = EOF_new

plt.figure(figsize=(15, 15))

ax = plt.axes(projection=rp)
#ax.coastlines('110m', linewidth=0.8)
ax.pcolormesh(rlon_itrp, rlat_itrp, EOF_new[:,:], transform=rp, cmap=plt.cm.rainbow)


# In[17]:


class MapXtremePCIC:
    """ 
    MapXtremePCIC maps design values over North America
    ====================================================
    Arguments
     CanRCM4.lens : information data list for CanRCM4 modelled design values over North America
     obs : data array of observed design values over North America, [lon, lat, data] three columns 
     res : resolution (in km) of the map
     method : whether EOF or SOM-based method is employed for mapping
     
    Value
     rlon : vector of longitude coordinates of the map
     rlat : vector of latitude coordinates of the map
     xtreme : data array of the mapped design values
     sp.basis : data array of the spatial basis functions estimated from CanRCM4 modelled data
     obs.grid : data array of the gridded observations 
    
    Note: the coordinate system is in polar rotated projection for all involved arrays. The projection
     is "+proj=ob_tran +o_proj=longlat +lon_0=-97 +o_lat_p=42.5 +a=1 +to_meter=0.0174532925199 +no_defs"
     
    Author : Nic Annau at PCIC, University of Victoria, nannau@uvic.ca
    """

    def __init__(self, CanRCM4, obs, res, method):
        
        self.CanRCM4 = CanRCM4
        self.obs = obs
        self.res = res
        self.method = method
        
        if (type(method) != type('string')):
            raise ValueError('Method argument requires {} got {}'.format(type('string'), type(method)))
        if (method != 'eof' and method != 'som'):
            raise Exception('MapXtremePCIC requires specified \'som\' or \'eof\'. Got {}'.format(method))
        
        if (obs.shape[0] < 100):
            raise Exception('Observed design values sample size of {} is too small (<100).'.format(obs.shape[0]))

        
        # Set default res if not specified
        if (res == None):
            print("Res not specified. Setting default res = 50.")
            self.res = 50
        
        # Check map resolution    
        if (type(res) != type(1)):
            raise ValueError('Mapping resolution requires {}, got {}'.format(type(1), type(res)))
        
        # Finds closest value to res in array
        resLibrary = [50, 25, 10, 5, 1]
        resLib = resLibrary[np.argmin(np.abs(np.subtract(res, res_library)))]
        resFactor = resLib/res

        #method <- match.arg(method)

        # organize (35) maps to (35) vectors
        xtreme = np.asarray(CanRCM4['df_pr'])
        rlon = CanRCM4['rlon']
        rlat = CanRCM4['rlat']

        # Get dimensions of input
        xtreme_dim = xtreme.shape
        nrlon = rlon.shape[0]
        nrlat = rlat.shape[0]
        nrun = xtreme.shape[0] # Number of files
                
        # initialize empty array
        xtremeMatrix = np.zeros((nrun, nrlon*nrlat))
        
        # Sklearn prep
        lons, lats = np.meshgrid(rlon, rlat)
        
        # Reshape 3D array into 2D combined array, i.e. 35 x 18476.
        # Each value in array is pr
        X = np.reshape(xtreme, (xtreme.shape[0], nrlat * nrlon), order='F')

        # Wherever there is NaN in array, that is a non-land value. 
        # Create masked array
        X = ma.masked_array(X, np.isnan(X))

        # Mask the ocean points
        ocean = X.sum(0).mask
        
        # Declare anything not masked as land
        # i.e. Compliment of ocean is land values in
        # boolean masked array
        land = ~ocean
        
        # Get land values only from array
        X = X[:, land]
        
        # Standardize data
        scaler  = preprocessing.StandardScaler()
        scaler_sst = scaler.fit(X)
        
        # Apply standardization
        X = scaler_sst.transform(X)
        
        # Should be gaussian centered on zero
        self.mean = X.mean()
        self.std = X.std()

        # Find principal components of our array
        skpca = pca.PCA()
        skpca.fit(X)
        
        # Number of PC to explain 70% of variance
        ipc = np.where(skpca.explained_variance_ratio_.cumsum() >= 0.70)[0][0]
        
        PCs = skpca.transform(X)
        
        # Get Empirical Orthogonal Functions components
        EOFs = skpca.components_
        
        # Select EOFs up to 70% of explained variance
        EOFs = EOFs[:ipc, :] 
        self.eof_shape = EOFs.shape

        # Create a new "dimension-reduced" PCA'd array
        EOF_recons = np.ones((ipc, len(rlat) * len(rlon))) * -999.
        for i in range(ipc): 
            # Fill each land portion of grid with
            # transformed principal components up to ipc
            EOF_recons[i,land] = EOFs[i,:] 
        
        EOF_recons = ma.masked_values(np.reshape(EOF_recons, (ipc, len(rlat), len(rlon)), order='F'), -999.)

        EOF_recons *= 100
        
        self.EOF_recons = EOF_recons
        
        proj = "+proj=ob_tran +o_proj=longlat +lon_0=-97 +o_lat_p=42.5 +a=1 +to_meter=0.0174532925199 +no_defs"
        
        rp = ccrs.RotatedPole(pole_longitude=-97 - 180,
                              pole_latitude=42.5,
                              globe=ccrs.Globe(semimajor_axis=6370000,
                                               semiminor_axis=6370000))
        proj = {'proj':'ob_tran', 'o_proj':'longlat', 'lon_0':-97, 'o_lat_p':42.5,
                'a':1, 'to_meter':0.0174532925199}
        
        rlon_itrp = np.linspace(np.min(rlon), np.min(rlon), len(rlon)*10)
        rlat_itrp = np.linspace(np.min(rlat), np.min(rlat), len(rlat)*10)
        
        Rlon, Rlat = np.meshgrid(rlon, rlat)
                
        f = interp2d(Rlon, Rlat, EOF_recons[0, :, :])
        EOF_new = f(rlon_itrp, rlat_itrp)
        self.EOF_new = EOF_new
        
        plt.figure(figsize=(15, 15))
        
        ax = plt.axes(projection=rp)
        #ax.coastlines('110m', linewidth=0.8)
        ax.pcolormesh(rlon_itrp, rlat_itrp, EOF_new[:,:], transform=rp, cmap=plt.cm.rainbow)


        # In order to reproduce the extent, we can't use cartopy's smarter
        # "set_extent" method, as the bounding box is computed based on a transformed
        # rectangle of given size. Instead, we want to emulate the "lower left corner"
        # and "upper right corner" behaviour of basemap.
        #xs, ys, zs = rp.transform_points(pc,
        #                                 np.array([-129.08, 22.66]),
        #                                 np.array([-10.59, 46.59])).T
        #ax.set_xlim(xs)
        #ax.set_ylim(ys)

        plt.show()
        # mapping
        #map.xtreme <- map.tp.basis[1]
        #for (i in 1:4) {
        #map.xtreme <- map.xtreme+map.sp.basis[, , i]*map.tp.basis[i+1]
        #}

        # output
        #return(list(rlon=map.rlon,
        #          rlat=map.rlat,
        #          xtreme=map.xtreme,
        #          sp.basis=map.sp.basis,
        #          obs.grid=obs.grid))
        


# In[18]:


res = 50
method = 'som'
res_library = [50, 25, 10, 5, 1]

xtreme = CanRCM4['df_pr']

rlon = CanRCM4['rlon']

mappy = MapXtremePCIC(CanRCM4, df_prx, res, method)
mappy.EOF_new


# In[ ]:


def landMask(lon, lat):
    """
    Creates land mask for North America
    Arguments
     lon : vector of longitude coordinates for making the land mask
     lat : vector of latitude coordinates for making the land mask
    Returns
     Value: a matrix with TRUE for land grid cells
    """
    nlon = len(lon)

    # all grid cells
    lon = np.tile(lon, len(lat))
    lat = np.tile(lat, len(lon))

    d = {'lon': lon,
         'lat': lat
        }

    pts = pd.DataFrame.from_dict(d)

    # pts.idx <- rownames(pts)

    # grid cells over land

    namCoastShp = coastlineCanRCM4() # Need to write this function

    worldBound = gpd.read_file(landPath)

    #pointList = [Point()]
    """
    coordinates(pts) <- ~lon+lat
    projection(pts) <- proj4string(nam.coast.shp)
    pts.land <- pts[!is.na(over(pts, as(nam.coast.shp, "SpatialPolygons"))),]
    pts.land <- data.frame(pts.land@coords)

    # where are the land grid cells 
    mask <- pts.idx %in% rownames(pts.land)

    # format as a matrix
    mask <- matrix(mask, nrow=num.lon, byrow=TRUE)
    """
    return()


# In[ ]:


def plot_field(m, X, lats, lons, vmin, vmax, step, cmap=plt.get_cmap('jet'),                ax=False, title=False, grid=False):
    if not ax: 
        f, ax = plt.subplots(figsize=(8, (X.shape[0] / float(X.shape[1])) * 8))
    m.ax = ax
    im = m.contourf(lons, lats, X, np.arange(vmin, vmax+step, step),                     latlon=True, cmap=cmap, extend='both', ax=ax)
    m.drawcoastlines()
    if grid: 
        m.drawmeridians(np.arange(0, 360, 30), labels=[0,0,0,1])
        m.drawparallels(np.arange(-80, 80, 20), labels=[1,0,0,0])
    m.colorbar(im)
    if title: 
        ax.set_title(title)


# In[ ]:


################

# Snippet from earlier...

################
# flatten the rlon and rlat into one column for each file
""" 
xtremeMatrix = np.asarray([xtreme[i, :, :].T.flatten() for i in range(nrun)])

# Copies of array with replicates of _m
rlonMatrix = np.tile(rlon, nrlat)
rlatMatrix = np.tile(rlat, nrlon)

# Indexes of NaN
idx = np.argwhere(~np.isnan(xtremeMatrix))
idxx = idx[:, 0]
idxy = idx[:, 1]

# Mask grid cells with missing values, e.g., oceans
self.xtremeMatrix = xtremeMatrix[:, ~np.isnan(xtremeMatrix).any(axis=0)]
self.rlonMatrix = rlonMatrix[~np.isnan(xtremeMatrix).any(axis=0)]
self.rlatMatrix = rlatMatrix[~np.isnan(xtremeMatrix).any(axis=0)]

# Spatial basis function
if (method == 'eof'):
# create a solver class, taking advantage of built-in weighting
solver = Eof(xtremeMatrix, weights='coslat')

# retrieve the first two EOFs from the solver class
eofs = solver.eofs(neofs=2)

"""


# In[ ]:





# In[ ]:





# In[ ]:




