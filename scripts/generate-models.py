import argparse
import os
import build_synthetic_set as bss
import numpy as np
import glob
from wotan import flatten
from tqdm import tqdm
from utils import import_lightcurve
import batman

parser = argparse.ArgumentParser(description='Create a synthetic set of exocomet transits injected into real lightcurves. This code injects one exoplanet/exocomet transit per lightcurve.')
parser.add_argument('folder', help="target directory(s)", nargs="+")
parser.add_argument('-n', '--number', help="number of files to inject exocomet models in.", type=int)
parser.add_argument('-o', '--output', help="output directory")

args = parser.parse_args()


def create_synthetic_set(folder, number, output, modeltype='exocomet'):
    """
    This function generates synthetic transits by injecting the models into real lightcurves.

    Notes: 
    - The exoplanet models are generated with `batman`.
    - The exocomet models are generated using the half-Gaussian, half-exponential model in Kennedy et al. 2019 and Norazman et al. 2024.

    Parameters:
    - folder (list): A list of paths containing light curve files (FITS format).
    - number (int): The number of of transits to generate.
    - output (str): Directory of where to save the models. Will be saved in `.npy` format.
    - modeltype (str, optional): The type of model to use for injecting into light curves. Default is 'exocomet'.

    Returns:
    - None
    """
    
    
    os.makedirs(output, exist_ok=True)
    fails = []
    times = []
    ticid = []
    
    paths = []
    for path in folder:
        paths.append(os.path.expanduser(path))
    
    files = []
    for path in paths:
        files.extend(glob.glob(os.path.join(path, "**/*lc.fits"), recursive=True))
    
    sample = np.random.choice(files, number, replace=False)
    
    for i in tqdm(sample):
        try:
            lc, info = import_lightcurve(i)
            tic = info['TIC_ID']
            flat_flux = flatten(lc['TIME'], lc['PCA_FLUX'], method='median', window_length=0.5)
            rms = np.std(flat_flux)
        
            ### Finds where there are large gaps in the lightcurve (default is set to 1 day).
            #### For reference, 30-minute cadences are 0.02 days
            diff = np.diff(lc['TIME'])
            large_gaps_indices = np.where(diff > 1)[0]
        
            min_snr = 5
            max_snr = 20 
            random_snr = np.random.uniform(min_snr, max_snr)
            
            A = rms * random_snr
            
            # Initialize a flag to indicate whether a valid time has been found
            valid_time_found = False
        
            while not valid_time_found:
                t0 = np.random.uniform(lc['TIME'][0], lc['TIME'][-1])
            
                # Check if the current random start time falls within any large gap or within 1.5 days before or after a gap
                for index in large_gaps_indices:
                    start_time = lc['TIME'][index] - 1
                    end_time = lc['TIME'][index + 1] + 1
                    if start_time <= t0 <= end_time:
                        # Current random start time falls within a data gap or within 1.5 days before or after a gap, select a new one
                        break
                    elif index < len(lc['TIME']) - 1 and diff[index] > 0.5 and abs(t0 - lc['TIME'][index + 1]) < 1.5:
                        # Current random start time is within 1.5 days after a data gap, select a new one
                        break
                    elif index > 0 and diff[index - 1] > 0.5 and abs(t0 - lc['TIME'][index]) < 1.5:
                        # Current random start time is within 1.5 days before a data gap, select a new one
                        break
                    elif t0 <= lc['TIME'][0] + 1:
                        # Current random start time is within one day after the beginning of the lightcurve, select a new one
                        break
                    elif t0 >= lc['TIME'][-1] - 1.5:
                        # Current random start time is within two days before the end of the lightcurve, select a new one
                        break
                else:
                    # Current random start time doesn't fall within any data gap, 1.5 days before or after a gap, or special conditions, set the flag to True
                    valid_time_found = True
        
                model = 1 - models.comet_curve(lc['TIME'], A, t0)
                flux = model * flat_flux
                fluxerror = np.array(lc['FLUX_ERR'] / np.nanmedian(lc['PCA_FLUX']))
            
                time = np.array(lc['TIME'])
                flux = np.array(flux)
                times.append(t0)
                ticid.append(tic)
                np.save(os.path.join(output, f"{info['TIC_ID']}_sector07.npy"), np.array([time, flux, fluxerror]))
        except TypeError as e:
            fails.append(i)
            print(f"Exception occurred for file {i}: {e}. Continuing...")
            continue


def exoplanet_params(di):
    """Exoplanet model parameters to use for `batman` model injection
    
    Parameters: 
    - di: a dictionary containing the desired exoplanet parameters.
    
    """


    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = 0.                        #time of inferior conjunction
    params.per = 5                      #orbital period
    params.rp = 0.1                       #planet radius (in units of stellar radii)
    params.a = 15.                        #semi-major axis (in units of stellar radii)
    params.inc = 87.                      #orbital inclination (in degrees)
    params.ecc = 0.                       #eccentricity
    params.w = 90.                        #longitude of periastron (in degrees)
    params.limb_dark = "nonlinear"        #limb darkening model
    params.u = [0.5, 0.1, 0.1, -0.1]    

if __name__ == "__main__":


