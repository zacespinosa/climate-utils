import pickle
import json
import os
import traceback
from typing import Any, Dict, Union

import numpy as np
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib
import matplotlib.animation as animation
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
    
# # -------------   Plotting Style --------------------------
# SMALL_SIZE = 20
# MEDIUM_SIZE = 20
# BIGGER_SIZE = 24

# styles = ["-", "--", "-.", ":", "-"]
# plt.style.use("seaborn-whitegrid")
# plt.rcParams["figure.figsize"] = [15, 10]
# plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
# plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
# plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
# plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
# plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc("lines", linewidth=4)
# plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.Paired.colors)

def to_pickle(
    path: Union[str, os.PathLike], 
    obj: Any,
) -> None:
    """
    Save obj to path
    """
    try: 
        with open(path, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:  
        traceback.print_exc()


def from_pickle(path: Union[str, os.PathLike]) -> Any:
    """
    Load pickle object from path 
    """
    try: 
        with open(path, 'rb') as handle:
            b = pickle.load(handle)
    except:  
        traceback.print_exc()
    else: 
        return b


def convert_coords(lat, lon, og_data, ccrs_grid):
    """convert data to appropriate coordinates for ccrs plot"""

    grid = ccrs_grid
    coords = grid.transform_points(ccrs.PlateCarree(), np.array(lon), np.array(lat))

    xs = np.ma.masked_invalid(coords[..., 0])
    ys = np.ma.masked_invalid(coords[..., 1])
    data = np.ma.masked_invalid(og_data)
    data.mask = np.logical_or(data.mask, xs.mask, ys.mask)

    xs = xs.filled(0)
    ys = ys.filled(0)

    return xs, ys, data


def plot_stationary_sp(minLon=-180, maxLon=180):
    """
    Create a stationary centered on south pole (Longitude: [minLon,maxLon]; Latitude: [-90,-60]
    Returns:
        ax, fig
    """
    fig = plt.figure(figsize=[20, 10])
    ax1 = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.04, right=0.95, wspace=0.02)

    # Limit the map to -60 degrees latitude and below.
    ax1.set_extent([minLon, maxLon, -90, -50], ccrs.PlateCarree())

    ax1.gridlines(draw_labels=True, color="black", linestyle="dashed", zorder=101)
    ax1.tick_params(which="both", zorder=103)

    ax1.add_feature(cartopy.feature.LAND, facecolor="grey", edgecolor="black", zorder=3)
    ax1.add_feature(cartopy.feature.OCEAN, facecolor="white")

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax1.set_boundary(circle, transform=ax1.transAxes)

    return fig, ax1


def xarray_time_to_monthly(ds):
    """
    Converts xarray from dims (time of type np.datatype64[M]) to (year, month) where year are integers and month are integers from 1 to 12

    Arguments:
    -----------
        ds [Dataset, DataArray](..., time)

    Returns:
    --------
        ds [Dataset, DataArray](..., year, month)
    """
    year = ds.time.dt.year
    month = ds.time.dt.month

    # assign new coords
    ds = ds.assign_coords(year=("time", year.data), month=("time", month.data))

    # reshape the array to (..., "month", "year")
    return ds.set_index(time=("year", "month")).unstack("time")


def xarray_monthly_to_time(df):
    """
    Converts xarray from dims (year, month) where year are integers and month are integers from 1 to 12 to (time of type np.datatype64[M])

    Arguments:
    -----------
        ds [Dataset, DataArray](..., year, month)

    Returns:
    --------
        ds [Dataset, DataArray](..., time)
    """
    # get first and last year
    firstYr, lastYr = df.year[0], df.year[-1] + 1
    
    # create time dimension
    df = df.stack(time=["year", "month"])
    
    # set time dimensions using first and last yr
    df["time"] = np.arange(f"{firstYr}-01", f"{lastYr}-01", dtype="datetime64[M]")
    
    return df


def xarray_time_to_daily(ds):
    """
    Converts xarray from dims (time of type np.datatype64[M]) to (year, day) where year are integers and day are integers from 1 to 366

    Arguments:
    -----------
        ds [Dataset, DataArray](..., time)

    Returns:
    --------
        ds [Dataset, DataArray](..., year, day)
    """
    year = ds.time.dt.year
    day = ds.time.dt.dayofyear

    # assign new coords
    ds = ds.assign_coords(year=("time", year.data), day=("time", day.data))

    # reshape the array to (..., "day", "year")
    return ds.set_index(time=("year", "day")).unstack("time")


def xarray_daily_to_time(ds, time=False):
    """
    Converts xarray from dims (year, day) where year are integers and day are integers from 1 to 366 to (time of type np.datatype64[M])

    Arguments:
    -----------
        ds [Dataset, DataArray](..., year, day)

    Returns:
    --------
        ds [Dataset, DataArray](..., time)
    """
    import calendar
    
    # get first and last year
    firstYr, lastYr = ds.year[0], ds.year[-1] + 1
    
    # create time dimension
    ds = ds.stack(time=["year", "day"])
    
    # Remove leap years
    leapYears = [i for i in range(1980, 2023) if not calendar.isleap(i)] # this is a hack because drop_sel doesn't work with MultiIndex => solution is convert to str
    ds["time"] = [str(i) for i in ds.time.values]
    for i in leapYears: 
        ds = ds.drop_sel({'time': f'({i}, 366)'})
    
    if time:
        ds["time"] = time
    else:
        # set time dimensions using first and last yr
        ds["time"] = np.arange(f"{firstYr}-01", f"{lastYr}-01", dtype="datetime64[D]")
    
    return ds


def xarray_latlon_to_space(df):
    """
    Converts xarray from dims (lat, lon) to space which is a tuple of floats (lat, lon)

    Arguments:
    -----------
        ds [Dataset, DataArray](..., lat, lon)

    Returns:
    --------
        ds [Dataset, DataArray](..., space)
    """
    # get first and last year
    lat, lon = df.lat, df.lon
    
    # create time dimension
    df = df.stack(space=["lat", "lon"])
    
    return df


def get_season(df, season):
    """
    Calculate the season average from an xarray with dims [...,time].
    season = Union["DJF", "SON", etc...]
    
    Arguments:
    -----------
        df [Dataset, DataArray](..., time)

    Returns:
    --------
        ds [Dataset, DataArray](..., season)
    """
    # select season
    df_season = df.sel(time=df.time.dt.season==season)
    # calculate mean per year
    return df_season.groupby(df_season.time.dt.year).mean("time")


def get_season_anchored(df, season="DJF"):
    """
    Get seasonal averages where annual year is anchored on December. Solves the problem in xarray where DJF gets discontinuous december. 
    (e.g. season=DJF and D is in 2021, F is in 2022, df would be indexed on 2022)
    
    Arguments:
    -----------
        df [Dataset, DataArray](..., time)

    Returns:
    --------
        df_season [Dataset, DataArray](..., season)
    """
    # Define season to index order
    seasons = {"MAM":(1,3), "JJA":(2,6), "SON": (3,9), "DJF":(4,12)}
    sidx, mon = seasons[season]
    
    # Anchor beginning of year in December
    df_season = df.resample(time='QS-DEC') 
    
    # Generate groups based on season
    df_season = df_season.mean(["time"])
    times = [t.values for t in df_season.time if t.dt.month == mon]
    df_season = df_season.sel(time=times)
    
    # Reindex time based on year of last month 
    # (i.e. so season=DJF and D is in 2021, F is in 2022, df would be indexed on 2022)
    df_season["time"] = df_season.time.dt.year + 1
    df_season = df_season.rename({"time": "year"})
    
    return df_season


def calc_sia_and_sie_nsidc(siconc, grid):
    """
    Calculate sea ice area and sea ice extent from nsidc data
    
    Arguments:
    -----------
        siconc [Dataset, DataArray](..., latitude, longitude)

    Returns:
    --------
        si [Dataset](..., sia, sie)
    """
    
    sia = (siconc * shgrid.areacello).sum(["latitude", "longitude"], skipna=True, min_count=100) / 1e6
    sie = xr.where(siconc >= 0.15, grid.areacello, 0).sum(["latitude", "longitude"], skipna=True, min_count=100) / 1e6
    
    si = xr.Dataset(data_vars={"sia": sia, "sie": sie})
    si.attrs['processed'] = 'SIA and SIE computed by Zac Espinosa'
    
    return si


def detrend_data(data, x, deg=1, trends=False):
    """
    Detrend data using n-degree least squares fit

    Arguments:
    -----------
        data [DataArray](..., x): data to detrend
        x [str]: name of dimension to detrend along 
        deg [int]: degree of polynomial to fit

    Returns:
    --------
        da [DataArray]: detrended data
        results [Dataset](..., ): linear trends (a) and offsets (b) in y = ax + b
    """
    results = data.polyfit(dim=x, skipna=True, deg=deg)
    new_data = data - xr.polyval(data[x], results.polyfit_coefficients)
    da = xr.DataArray(new_data, coords=data.coords, dims=data.dims, attrs=data.attrs)
    
    if trends: 
        return da, results
    
    return da


def detrend_blocks(da, dim, deg=1, fit=False): 
    """
    Detrend data using n-degree least squares fit and using blocks (much more efficient)

    Arguments:
    -----------
        data [DataArray](..., x): data to detrend
        dim [str]: name of dimension to detrend along 
        deg [int]: degree of polynomial to fit
        fit [bool]: return fit 

    Returns:
    --------
        da [DataArray]: detrended data
        fit [Dataset](..., ): linear trends (a) and offsets (b) in y = ax + b
    """
    import xarray as xr
    
    def _detrend(dab):
        # needed to workaround xarray's check with zero dimensions
        # https://github.com/pydata/xarray/issues/3575
        if len(dab[dim]) == 0:
            return dab

        # detrend along a single dimension
        p = dab.polyfit(dim=dim, deg=deg)
        res = xr.polyval(dab[dim], p.polyfit_coefficients)
        
        if fit: 
            return (dab - res, res)
        
        return dab - res
    
    return xr.map_blocks(_detrend, da, template=da)

                         
def anoms_blocks_daily(da):
    """

    Arguments:
    -----------
        da [DataArray](..., x): data to detrend

    Returns:
    --------
        anoms [DataArray]: anomalies along dim
    """
    import xarray as xr
    
    def _anoms(da):
        if "day" in da.dims: 
            return da - da.mean(dim="day")
        else: 
            gb = da.groupby('time.dayofyear')
            clim = gb.mean(dim='time')
            return gb - clim
                        
                         
    return xr.map_blocks(_anoms, da, template=da)


