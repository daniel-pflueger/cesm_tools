import numpy as np
import xarray as xr

def global_mean(da,mask=None):

    """
        Takes an xarray DataArray defined over a lat/lon field
        and returns global mean computed with correct weight factors.
        Can also work on an exclusively meridional field (only lat coordinates)

        Parameters:
        da (DataArray): data array containing a field defined over latitudes (lat) and longitudes (lon) and optionally time

        Returns:
        float or DataArray: global mean of the field (over time)
    """

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
        mask_int = ((np.cos(np.deg2rad(da.lat))) * dlat * mask_zonal).sum(dim="lat")
        norm_factor = float(mask_int)
        
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