import os, sys
sys.path.insert(1, '/Users/azib/Documents/open_source/nets2/stella/')
sys.path.insert(1, '/Users/azib/Documents/open_source/nets2/scripts/')
import numpy as np
from utils import *
import build_synthetic_set as models
import matplotlib.pyplot as plt
from astropy.table import Table
from glob import glob
from tqdm import tqdm
import wotan
import lightkurve as lk
import stella
import pandas as pd
import random
import batman 
import warnings
import argparse
from astroquery.mast import Catalogs
import astropy.constants as const
import time as ti
import re
import pickle
import signal

files = glob('../data/eleanor/**/*.fits', recursive=True)
random.shuffle(files)

folder = '../eleanor-lite-models' # do not add the dash after folder name here as it is done in the loop for saving files

# parser = argparse.ArgumentParser(description='Create synthetic transits for exocomets, exoplanets, or eclipsing binaries.')
# parser.add_argument(help='Choose what transits you want. Option: exocomets, exoplanets, EB', nargs='+',default='exocomet')



def calculate_timestep(table):
    """
    Function: Calculates the median value of the time differences between data points in a given table. 
    Provides an estimate of the timestep (or time delta) between consecutive data points.

    Parameters:
    :table (array or pandas.DataFrame): The input table containing time-series data.

    Returns:
    :dt (float): The estimated time interval or timestep between consecutive data points."""

    try:
        dt = [ table[i+1][0] - table[i][0] for i in range(len(table)-1) ] # calculates difference between (ith+1) - (ith) point 
        dt.sort()
        return dt[int(len(dt)/2)] # median of them.
    except:
        return np.median(np.diff(table['time'])) ## change this to account for any time column names

    

def clean_data(table):
    """
    Function: Interpolating missing data points, ensuring equal time gaps between points. 
    Returns five numpy arrays: time, flux, quality, real, and flux_error. Real is 0 if data point interpolated, 1 otherwise.

    Parameters:
    :table (astropy.table.table): The input table containing time-series data.
    
    Returns:
    :time (numpy.ndarray): An array of timestamps for each data point, including the interpolated points.
    :flux (numpy.ndarray): An array of flux values for each data point, including the interpolated points.
    :quality (numpy.ndarray): An array indicating the quality of each data point, including the interpolated points.
    :real (numpy.ndarray): An array indicating whether each data point is real (1) or interpolated (0).
    :flux_error (numpy.ndarray): An array of flux error values for each data point, including the interpolated points."""


    time = []
    flux = []
    quality = []
    real = []
    flux_error = []
    timestep = calculate_timestep(table)


    ### this scale factor ensures that you can use any cadence of lightcurves. 48 cadences = 1 day.
    factor = ((1/48)/timestep)

    for row in table:
        ti, fi, qi, fei = row

        if len(time) > 0:
            steps = int(round( (ti - time[-1])/timestep * factor)) # (y2-y1)/(x2-x1)
            if steps > 1:
                fluxstep = (fi - flux[-1])/steps
                fluxerror_step = (fei - flux_error[-1])/steps

                # For small gaps, pretend interpolated data is real.
                if steps > 4:
                    set_real=0
                else:
                    set_real=1

                for _ in range(steps-1):
                    time.append(timestep + time[-1])
                    flux.append(fluxstep + flux[-1])
                    flux_error.append(fluxerror_step + flux_error[-1])

                    quality.append(0)
                    real.append(set_real)
        time.append(ti)
        flux.append(fi)
        quality.append(qi)
        real.append(1)
        flux_error.append(fei)

    return [np.array(x) for x in [time,flux,quality,real,flux_error]]

os.makedirs(folder, exist_ok=True)



def timeout_handler(signum, frame):
    raise TimeoutError("Timeout reached.")

def inject_valid_time(time,real):
     # INITIALISE FLAG TO FIND IF TIME IS GOOD TO INJECT IN (THIS IS BASED ON THE TIMES OF THE ORIGINAL LIGHTCURVE)
        valid_time_found = False

        while not valid_time_found:
            t0 = np.random.uniform(time[0], time[-1])

            # Check if t0 avoids large gaps
            valid_t0 = False
            for index in large_gaps_indices:
                start_time = time[index] - 1
                end_time = time[index + 1] + 1
                if start_time <= t0 <= end_time:
                    break
                elif index < len(time) - 1 and diff[index] > 0.5 and abs(t0 - time[index + 1]) < 2:
                    break
                elif index > 0 and diff[index - 1] > 0.5 and abs(t0 - time[index]) < 2:
                    break
                elif t0 <= time[0] + 2:
                    break
                elif t0 >= time[-1] - 2:
                    break
            else:
                valid_t0 = True

            if valid_t0:
                # Check if all data points within the window are non-interpolated
                window_start = np.argmin(np.abs(time - (t0 - window_size * np.median(np.diff(time)))))
                window_end = np.argmin(np.abs(time - (t0 + window_size * np.median(np.diff(time))))) + 1
                if np.all(real[window_start:window_end] == 1):
                    valid_time_found = True

def planet(A, time):
    alpha = 2.0
    period_min = 5
    period_max = 700.0

    num_planets = 1
    max_retries = 5
    retry_delay = 1
    timeout_duration = 5

    TIC_table = Catalogs.query_object(f'TIC 270577175', catalog="TIC")
    r_star = TIC_table['rad'][0]
    m_star = TIC_table['mass'][0]

    for i in np.arange(0, num_planets):
        retries = 0
        while retries < max_retries:
            try:
                params = batman.TransitParams()
                random_value = np.random.uniform(0, 1)
                params.per = period_min * (period_max / period_min) ** (random_value ** (1 / alpha))
                params.rp = np.sqrt(A)
                params.a = (((const.G.value * m_star * const.M_sun.value * (params.per * 86400.) ** 2) / (4. * (np.pi ** 2))) ** (1. / 3)) / (r_star * const.R_sun.value)
                params.inc = 90  # np.random.uniform(88.0,90.0)
                params.ecc = 0.
                params.w = 90.
                params.limb_dark = "linear"

                ld_coeff = np.random.uniform(0, 1)
                params.u = [ld_coeff]

                # lcp = lc.to_pandas()

                m = batman.TransitModel(params, time, fac=0.02)
                batman_flux = m.light_curve(params)

                # Check if the minimum value of batman_flux is 1 (flat line)
                if np.min(batman_flux) == 1:
                    retries += 1
                    if retries == max_retries:
                        failed_iterations += 1
                    else:
                        ti.sleep(retry_delay)
                    continue
                break  # Exit the while loop if the condition is not met

            except TimeoutError:
                signal.alarm(0)  # Reset the alarm
                failed_iterations += 1
                break  # Exit the while loop

            except Exception as e:
                if "Convergence failure in calculation of scale factor for integration step size" in str(e):
                    retries += 1
                    if retries == max_retries:
                        failed_iterations += 1
                    else:
                        ti.sleep(retry_delay)
                else:
                    raise e

        elapsed_time = ti.time() - start_time
        if elapsed_time > timeout_duration:
            print(f"Timeout reached for planet {i + 1}. Skipping this planet.")
            break  # Exit the for loop
        signal.alarm(0)
        
fails, times, rmsfails, ticid = [], [], [], []

min_snr = 5
max_snr = 20
window_size = 84  # Number of cadences representing the window size (3.5 days)
target_ID = [int(re.search(r"(\d{16})", filename).group(1).lstrip('0')) for filename in files]
print(target_ID)
break

for i in tqdm(files[0:1000]):
    try:
        ### READ IN LIGHTCURVE
        lc, lc_info = import_lightcurve(i, drop_bad_points=True)
        sector = f"{lc_info['sector']:02d}"

        ### FLATTEN THE ORIGINAL LIGHTCURVE
        flat_flux = wotan.flatten(lc['TIME'], lc['PCA_FLUX'], method='median', window_length=1)

        ### GET RMS OF FLATTENED ORIGINAL LIGHTCURVE
        rms = np.nanstd(flat_flux)
        if np.isnan(rms):
            rmsfails.append(rms)
            continue

        ### IDENTIFY LARGE GAPS IN ORIGINAL LIGHTCURVE
        diff = np.diff(lc['TIME'])
        large_gaps_indices = np.where(diff > 1)[0]

        ### CREATE COPY OF LIGHTCURVE
        lcc = lc.copy()
        lcc = lcc[lcc['QUALITY'] == 0]
        tic = lc_info['TIC_ID']
        lcc = lcc['TIME', 'PCA_FLUX', 'QUALITY', 'FLUX_ERR']

        ### INTERPOLATE THE COPIED LIGHTCURVE
        time, flux, quality, real, flux_error = clean_data(lcc)

        ### CHOOSE RANDOM SNR VALUE
        random_snr = np.random.uniform(min_snr, max_snr)
        A = rms * random_snr

        ### SELECTING TIME TO INJECT LIGHTCURVE. USED ORIGINAL TIME ARRAY TO IDENTIFY GAPS
        t0 = inject_valid_time(lc['TIME'], real)

        ### CREATE MODEL BASED ON THE INTERPOLATED LIGHTCURVE TIME ARRAY

        model = 1 - models.comet_curve(time, A, t0)

        ### INJECT MODEL INTO INTERPOLATED LIGHTCURVE
        f = model * (flux/np.nanmedian(flux))
        fluxerror = flux_error/flux

        ### APPEND TIMES AND TIC ID FOR THE CATALOG
        times.append(t0)
        ticid.append(tic)

        ### SAVE INTO NUMPY FOLDER
        np.save(f"{folder}/{lc_info['TIC_ID']}_sector{sector}.npy", np.array([time, f, fluxerror, real]))

    except TypeError as e:
        fails.append(i)
        print(f"Exception occurred for file {i}: {e}. Continuing...")
        continue
