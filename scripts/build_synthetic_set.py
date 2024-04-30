import os
import math
import numpy as np
from utils import import_lightcurve
from scipy.optimize import curve_fit
from astropy.table import Table

"""Building the synthetic data for the training set"""


### some parameter space
#depth = 10 ** np.random.uniform(-4, -1, 1)[0] # range from 0.0001 to 0.1
#skewness = np.random.uniform(0, 30, 0.001)[0] # range from 0 to 30

def gauss(t,A,t0,sigma):
    """
    Creates a Gaussian function.

    Parameters:
        :t (float or array): Time or array of times at which to evaluate the Gaussian function.
        :A (float): Amplitude of the Gaussian peak.
        :t0 (float): Mean or centre of the Gaussian distribution.
        :sigma (float): Standard deviation or width of the Gaussian distribution.

    Returns:
        float or array: Value of the Gaussian function at the given time(s)."""

    return abs(A)*np.exp( -(t - t0)**2 / (2 * sigma**2) )


def comet_curve(t,A,t0,sigma=3.28541476e-01,tail=3.40346173e-01):
    ### add the Beta Pic parameters
    """
    Creates an exocomets light curve model.

    Notes: These are the Beta Pic parameters:
        - A = 8.84860547e-04, t0 = 1.48614591e+03, sigma = 3.28541476e-01, alpha (skewness) = 1.43857307e+00, tail = 3.40346173e-01

    Parameters:
        t (array): Independent variable (time) values.
        A (float): Amplitude of the Gaussian curve.
        t0 (float): Mean (centre) of the Gaussian curve.
        sigma (float): Standard deviation of the Gaussian curve.
        tail (float): Tail parameter controlling decay rate after t0.

    Returns:
        array: The computed values of the asymmetric Gaussian."""

    x = np.zeros(len(t))
    for i in range(len(t)):
        if t[i] < t0:
            x[i] = gauss(t[i],A,t0,sigma)
        else:
            x[i] = A*math.exp(-abs(t[i]-t0)/tail)
                
    return x

def comet_curve_fit(x,y):
    """Fits the exocomet light curve model to the data."""
    # Initial parameters guess
    # x = time
    # y = flux
    i = np.argmax(y)

    width = x[-1]-x[0]

    params_init = [y[i],x[i],width/3,width/3]

    params_bounds = [[0,x[0],0,0], [np.inf,x[-1],width/2,width/2]]
    params,cov = curve_fit(comet_curve,x,y,params_init,bounds=params_bounds)
    return params, cov

def create_transit_model(time, depth, t0, sigma=3.02715600e-01, tail=3.40346173e-01):
    """Creates an exocomet light curve model. This does not return the flux, 
    you will have to multiply by the flux of your original lightcurve."""
    return 1 - comet_curve(time, depth, t0, sigma, tail)

def inject_lightcurve(table, time=None,depth=None):
    """Injects the exocomet light curve model into the light curve.
    
    Inputs:
    - file: the light curve filepath
    - time: the time array of the light curve. If None, a random time is calculated.
    - depth: the depth of the transit. If None, a random depth between 0.1% and 0.0001% is calculated.
    
    """
    time = lc['TIME']
 
    if time is not None:
        depth = 10 ** np.random.uniform(-4, -2, 1)[0]
        t0 = np.random.uniform(time[0], time[-1])

    lc['INJFLUX'] = create_transit_model(time, depth, t0)

    return lc

def save_lightcurve(data, lc_info, format='fits'):
    """
    Saves lightcurve as either a FITS file or a NumPy array in .npz format based on the specified format.

    Parameters:
        data: Astropy Table
            Data to be saved.
        format: str, optional
            Format in which to save the data. Can be 'fits' for FITS files or 'npz' for NumPy arrays in .npz format.
            Default is 'fits'.
    """
    # # Save the data as a FITS file
    # if format == 'fits':
    #     if isinstance(data, Table):
    #         filename = f'K2_{lc_info['KEPLERID']}.fits'
    #         data.write(filename, format='fits')
    #     else:
    #         raise ValueError("Data must be an Astropy Table for FITS files.")
    # # Convert the Astropy Table to a NumPy array and save as .npz format
    # elif format == 'npz':
    #     if isinstance(data, Table):
    #         filename = f'K2_{lc_info['KEPLERID']}.npz'
    #         column_dict = {}
    #         for col in data.dtype.names:
    #             # Extract the column corresponding to the current field
    #             column_data = data[col]
                
    #             # Add the column data to the dictionary with the field name as the key
    #             column_dict[col] = column_data

    #         # Save the dictionary containing all columns as a single .npz file
    #         np.savez(f'{lc_info['KEPLERID']}.npz', **column_dict)
    #     else:
    #         raise ValueError("Data must be an Astropy Table for .npz format.")
    # else:
    #     raise ValueError("Unsupported format. Use 'fits' for FITS files or 'npz' for NumPy arrays.")