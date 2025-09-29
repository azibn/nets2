"""Converts FITS files (in a directory) to .npy files for the CNN"""


import os
import sys
import pickle
import numpy as np
import argparse
import tqdm
import pandas as pd
from astropy.table import Table
sys.path.insert(1,'scripts')
sys.path.insert(1, 'stella')

from utils import *

parser = argparse.ArgumentParser(description='Convert FITS files to NPY files.')
parser.add_argument('--fits_dir', type=str, 
                    help='Directory containing FITS files')
parser.add_argument('--npy_dir', type=str,
                    help='Directory to save .npy files')

args = parser.parse_args()
    


def fits_to_npy(file):
    lc, lc_info = import_lightcurve(file)
    time = np.array(lc['TIME'])
    flux = np.array(lc['PCA_FLUX'])
    flux_err = np.array(lc['FLUX_ERR'])

    #### MADE TWO RANDOM TIMES FOR SVC IN MIDDLE OF THE REAL DATA BITS
    t0 = np.random.uniform(lc['TIME'][0]+2,lc['TIME'][0] + 7)
    t2 = np.random.uniform(lc['TIME'][-1] - 7,lc['TIME'][-1]-2)
    q = np.array(lc['QUALITY']) 
    return time, flux, flux_err, lc_info['TIC_ID'], q, t0, t2

def scale_lightcurve(time, flux, flux_error):
    # Calculate the scaled flux
    f = np.array((flux / np.nanmedian(flux)) - 1)
    mask = ~np.isnan(flux)
    t = time[mask]
    f = f[mask]
    flux_error = flux_error[mask]

    f = (f - np.min(f)) / (np.max(f) - np.min(f))
    del mask
    return t, f, flux_error


def convert_fits_to_npy(fits_dir, npy_dir):
    os.makedirs(npy_dir, exist_ok=True)
    times = []
    tics = []
    for file in tqdm.tqdm(os.listdir(fits_dir)):
        if file.endswith('.fits'):
            time, flux, flux_err, tic, q, t0, t2 = fits_to_npy(os.path.join(fits_dir, file))
            time, flux, flux_err = scale_lightcurve(time, flux, flux_err)
            np.save(os.path.join(args.npy_dir,f'{tic}_sector07.npy'), [time, flux, flux_err, q])
            times.append(t0)
            times.append(t2)
            tics.append(tic)
            tics.append(tic)



    data = pd.DataFrame({'TIC':tics, 'tpeak':times})
    data.TIC = data.TIC.astype(int)
    t = Table.from_pandas(data)
    t.write(f'{args.npy_dir}.txt', format='ascii', overwrite=True) 




if __name__ == '__main__':

    convert_fits_to_npy(args.fits_dir, args.npy_dir)