import numpy as np
import xarray as xr
import os
import pop_tools

def global_integral(da,
                    mask = None,
                   surface_area = False):
    '''
    Perform an integral over the equidistant lon-lat field da. This should return 4 pi if da is one everywhere. If surface_area = True, the result is multiplied by the Earth's surface area in m^2 per solid angle. If da is a kind of 'surface density' (units X/m^2) then the result is the 'mass' of the surface (units X). For example if da is an energy flux of units W/m^2, then the result is the integrated energy flux in W.
    '''

    assert mask is None, 'Using a mask not currently supported. Just multiply input DataArray by your mask.'
    
    integral = global_mean(da,mask=mask,integrate=True)
    if surface_area:
        r_e = 6.378 * 10**6 # Earth radius in meters
        integral = integral * r_e**2
    return integral

def global_mean(da,
                mask=None,
               integrate=False):

    """
        Takes an xarray DataArray defined over a lat/lon field
        and returns global mean computed with correct weight factors.
        
        Can also work on an exclusively meridional field (only lat coordinates)
        
        To compute regional means, supply a mask.
        
        Performs an integration rather than averaging when 'integrate=True'. Use the wrapper global_integral for this purpose
        
        Parameters:
        ---
        da (DataArray): DataArray containing a field defined over latitudes (lat) and longitudes (lon) and optionally time
        mask (DataArray): DataArray with dimensions of 'da' and time dimension removed. Values must either be 1 (included in the mask) or 0 (cropped)

        Returns:
        ---
        float or DataArray: global mean of the field (over time)
    """

    assert not (not mask is None and integrate), 'Can not currently use masks while integrating'

    # apply mask
    if not mask is None:
        da = da * mask
    
    # detect if data has longitudinal component
    if 'lon' in list(da.coords):
        zonal = False
    else:
        zonal = True
    
    # determine the grid size
    # note: this requires an equally spaced coordinate grid
    dlat = np.deg2rad(np.diff(da.lat.data)[0])
    
    # zonal integration
    if not zonal:
        dlon = np.deg2rad(np.diff(da.lon.data)[0])
        da_zonal = da.sum(dim="lon") * dlon
        if mask is None: # apply zonal normalization if no mask is given. this is always 1/(2pi)
            da_zonal *= 1.0/(2.0*np.pi)
    else:
        da_zonal = da

    # meridional integration
    
    if mask is None: # mask needs to be in lon/lat format
        norm_factor = 2
    else:
        dlon = np.deg2rad(np.diff(da.lon.data)[0])
        mask_zonal = mask.sum(dim="lon") * dlon
        # only compute the mask normalization when we are averaging mode rather than integration mode
        if not integrate: 
            mask_int = ((np.cos(np.deg2rad(da.lat))) * dlat * mask_zonal).sum(dim="lat")
            norm_factor = float(mask_int)

            
    if integrate:
        # Do not perform an integration when in integration mode 'integrate=True'
        # The factor 1/(2*pi) undoes the normalization that occurs during zonal averaging
        norm_factor = 1/(2*np.pi)
        
    weight = 1.0/norm_factor * np.cos(np.radians(da.lat)) * dlat # diff. element for meridional int. (higher latitudes give lower contribution)
    da_weighted = da_zonal * weight

    return da_weighted.sum(dim="lat")

def meridional_L(da,n):
    """
        Takes an xarray DataArray defined over a lat/lon field
        and returns nth component of meridional Legendre decomposition.

        Parameters:
        da (DataArray): data array containing a field defined over latitudes (lat) and longitudes (lon) and optionally time
        n (int): specifies which Legendre component to compute (either 0, 1 or 2)

        Returns:
        float or DataArray: 0th component of Legendre decomposition of the field (over time)
    """
    
    # detect if data has longitudinal component
    if 'lon' in list(da.coords):
        zonal = False
    else:
        zonal = True
    
    assert isinstance(da,xr.DataArray), 'Input must be xarray.DataArray'
    assert n in [0,1,2], 'Specify n=0,1,2'

    # determine the grid size
    # note: this requires an equally spaced coordinate grid
    dlat = np.deg2rad(np.diff(da.lat.data)[0])

    # zonal integration
    if not zonal:
        dlon = np.deg2rad(np.diff(da.lon.data)[0])
        da_zonal = 1.0/(2.0*np.pi) * da.sum(dim="lon") * dlon
    else:
        da_zonal = da

    basis_func = { # Legendre polynomials of nth degree (evaluated along lat grid)
        0: da.lat**0,
        1: np.sin(np.deg2rad(da_zonal['lat'])),
        2: 0.5*(3.0 * np.sin(np.deg2rad(da_zonal['lat']))**2 - 1.0)
    }

    weight = 0.5 * np.cos(np.radians(da.lat)) * dlat # diff. element for meridional int. (higher latitudes give lower contribution)
    da_weighted = da_zonal * weight * basis_func[n]

    component = da_weighted.sum(dim="lat")

    return component

def sea_ice_extent(da,mode='global',threshold=0.15):
    '''
    Computes global sea ice extent as defined by total area of grid cells with ice fraction above 15% (if threshold not otherwise specified). 
    Units in km^2.
    
    Parameters:
    ---
    da (xr.DataArray): Ice fraction DataArray with equidistant lon-lat grid and optional time dimension.
    mode (string): Either 'global', 'nh' or 'sh' for global, northern or southern hemisphere sea ice extent
    threshold (float): Value between 0 and 1  that determines threshold above which a grid cell's area is counted.
    
    Returns:
    ---
    xr.DataArray: Sea ice extent in km^2
    '''
    
    assert mode in ['global','nh','sh'], '\'mode\' keyword-argument must either be \'global\', \'nh\' or \'sh\''
    
    # Global mask: all values set to 1
    if 'time' in da.dims:
        mask = xr.ones_like(da.isel(time=0))
    else:
        mask = xr.ones_like(da)
    
    # Select southern or northern hemisphere depending user-specified mode
    if mode == 'nh':
        mask = mask.where(mask.lat>0).fillna(0)
    elif mode == 'sh':
        mask = mask.where(mask.lat<0).fillna(0)
        
    da = da * mask

    # Set values to 1 if ICEFRAC > 0.15
    da = (da.where(da>threshold)/da.where(da>threshold)).fillna(0)
    
    extent = global_integral(da,surface_area=True) / 10**6
    
    return extent
        
# Ocean tools (for POP2 output)

 
# to-do: use pop-tools rather than hardcoded data file!
file_dir_path = os.path.dirname(os.path.realpath(__file__))
pop_ds_rel_loc = './grids/POP_gx1v7.nc'
pop_ds_loc = os.path.join(file_dir_path,pop_ds_rel_loc)
if not os.path.isfile(pop_ds_loc):
    pop_ds = pop_tools.get_grid('POP_gx1v7')
    pop_ds.to_netcdf(pop_ds_loc)
else:
    pop_ds = xr.open_dataset(pop_ds_loc)

    #pop_ds = xr.open_dataset('/home/dpfluger/links/leo_archive/b.e21.BSSP585cmip6.f09_g17.control.01/ocn/hist/b.e21.BSSP585cmip6.f09_g17.control.01.pop.h.2021-01.nc')
tarea = pop_ds.TAREA * 10**(-4) # convert cm^2 to m^2
uarea = pop_ds.UAREA * 10**(-4)
grids = {'T': tarea, 'U': uarea, 'norm': float(tarea.sum())}
ocn_area = tarea.sum()
def global_mean_ocn(da,grid_id='T',grids=grids,ocn_area=ocn_area):
    assert grid_id in grids.keys(), 'Must choose valid grid identifier (default T or U)'
    
    weighted_sum = (da * grids[grid_id]).sum(['nlat','nlon'])
    mean = weighted_sum/ocn_area
    
    return mean   
