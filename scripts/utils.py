from astropy.io import fits
from astropy.table import Table
import numpy as np
import pandas as pd


def import_lightcurve(filepath,drop_bad_points=True,return_type='astropy',return_meta_as_dict=False):
    """Importing a lightcurve given a FITS format file.
    
    Parameters: 
    filepath (str): Path to the FITS file
    drop_bad_points (bool): Whether to drop bad points or not

    Returns:
    data (astropy.table.Table): Table containing the lightcurve data
    meta (astropy.io.fits.header.Header): Header containing the metadata

    
    
    """
    lc = fits.open(filepath)

    meta = lc[0].header
    data = lc[1].data
    if drop_bad_points:
        data = data[data['QUALITY'] == 0]

    return_types = ['astropy','pandas','pd']
    lc.close()

    data = Table(data)
    
    if return_type == 'pandas' or return_type == 'pd':
        data = Table(data).to_pandas()

    if return_meta_as_dict:
        meta = dict(meta)
        
    return data, meta

def normalise(flux):
    """Median-normalises the flux to 1"""
    return flux / np.median(flux)
