![logo.png](https://images.zenhubusercontent.com/5bc02597fcc72f27390ed1f9/c2cf2ba4-edb1-4b47-856e-20338712d4a7)
# climpyrical

climpyrical is a Python tool for reconstructing design value fields using meteorological station observations 

and ensembles of design value fields provided in CanRCM4 models.

### Requirements
climpyrical relies on the great work done by the following open source projects:
* [Numpy](https://www.numpy.org/)
* [Cartopy](https://scitools.org.uk/cartopy/docs/latest/)
* [xarray](http://xarray.pydata.org/en/stable/)
* [matplotlib](https://matplotlib.org/)
* [scipy](https://www.scipy.org/)
* [pandas](https://pandas.pydata.org/)

And many more. To install all of the dependencies used by climpyrical, install from requirements file found in `requirements.txt`

via 

```bash
pip install -r requirements.txt
```

### Getting started
There are a series of helper functions for conducting the empirical orthogonal function analysis. The software requires a list of NetCDF model files all contained within a single directory, and assmebles them into a datacube.

```python
>>> from datacube import read_data

>>> PATH = '/path/to/climate/models'

>>> ds = read_data(PATH)

<xarray.Dataset>
Dimensions:       (bnds: 2, rlat: 130, rlon: 155, run: 35)
Coordinates:
    time          object 1983-07-17 23:30:00
    lon           (rlat, rlon) float64 232.9 233.3 233.7 ... 335.5 335.9 336.4
    lat           (rlat, rlon) float64 12.36 12.52 12.68 ... 59.77 59.46 59.15
  * rlon          (rlon) float64 -33.88 -33.44 -33.0 -32.56 ... 33.0 33.44 33.88
  * rlat          (rlat) float64 -28.6 -28.16 -27.72 ... 27.28 27.72 28.16
...
```
please see `demo.ipynb` for more information and example usage on a climate field.

## Authors
* **Nic Annau** - [Pacific Climate Impacts Consortium](https://www.pacificclimate.org/)
