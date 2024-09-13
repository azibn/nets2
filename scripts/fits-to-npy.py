"""Converts FITS files (in a directory) to .npy files for the CNN"""


import os
import sys
import pickle
import numpy as np
import argparse
import tqdm
import pandas as pd
from astropy.table import Table

current_dir = os.getcwd()
while os.path.basename(current_dir) != 'nets2':
    current_dir = os.path.dirname(current_dir)
    if current_dir == os.path.dirname(current_dir): 
        raise Exception("'nets2' directory not found in parent directories")
    
sys.path.insert(1, os.path.join(current_dir, 'scripts'))
sys.path.insert(1, os.path.join(current_dir, 'stella'))

from utils import *

parser = argparse.ArgumentParser(description='Convert FITS files to NPY files.')
parser.add_argument('--fits_dir', type=str, default=os.path.join(current_dir, 'models/svc-fits'),
                    help='Directory containing FITS files')
parser.add_argument('--npy_dir', type=str, default=os.path.join(current_dir, 'models/svc'),
                    help='Directory to save .npy files')

args = parser.parse_args()
    


def fits_to_npy(file):
    lc, lc_info = import_lightcurve(file)
    time = np.array(lc['TIME'])
    flux = np.array(lc['PCA_FLUX'])
    flux_err = np.array(lc['FLUX_ERR'])
    t0 = np.random.uniform(lc['TIME'][0],lc['TIME'][-1])
    return time, flux, flux_err, lc_info['TIC_ID'], t0

def convert_fits_to_npy(fits_dir, npy_dir):
    os.makedirs(npy_dir, exist_ok=True)
    times = []
    tics = []
    for file in tqdm.tqdm(os.listdir(fits_dir)):
        if file.endswith('.fits'):
            time, flux, flux_err, tic, _ = fits_to_npy(os.path.join(fits_dir, file))
            np.save(os.path.join(args.npy_dir,f'{tic}_sector07.npy'), [time, flux, flux_err])
            t0 = 1513
            times.append(t0)
            tics.append(tic)



    data = pd.DataFrame({'TIC':tics, 'tpeak':times})
    data.TIC = data.TIC.astype(int)
    t = Table.from_pandas(data)
    t.write(f'{args.npy_dir}.txt', format='ascii', overwrite=True) 




if __name__ == '__main__':

    convert_fits_to_npy(args.fits_dir, args.npy_dir)