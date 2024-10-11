"""
Creates models of exocomets/exoplanets/eclipsing binaries, and injecting them into real lightcurves. 

Exocomet models were created using custom functions in `models.py`, while exoplanet and binary models were created using the `batman` package.
"""

import os
import sys
import time as ti
import argparse
import random
import time as ti
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from astropy.table import Table
import wotan
import batman
import astropy.constants as const
from astroquery.mast import Catalogs
import signal
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

sys.path.insert(1, 'scripts')
sys.path.insert(1, 'stella')

from utils import *
import models
import stella

def calculate_timestep(table):
    """
    Function: Calculates the median value of the time differences between data points in a given table.
    Provides an estimate of the timestep (or time delta) between consecutive data points.

    Parameters:
    :table (array or pandas.DataFrame): The input table containing time-series data.

    Returns:
    :dt (float): The estimated time interval or timestep between consecutive data points.
    """

    try:
        dt = [
            table[i + 1][0] - table[i][0] for i in range(len(table) - 1)
        ]  # calculates difference between (ith+1) - (ith) point
        dt.sort()
        return dt[int(len(dt) / 2)]  # median of them.
    except:
        return np.median(
            np.diff(table["time"])
        )  ## change this to account for any time column names


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
    :flux_error (numpy.ndarray): An array of flux error values for each data point, including the interpolated points.
    """

    time = []
    flux = []
    quality = []
    real = []
    flux_error = []
    timestep = calculate_timestep(table)

    ### this scale factor ensures that you can use any cadence of lightcurves. 48 cadences = 1 day.
    factor = (1 / 48) / timestep

    for row in table:
        ti, fi, qi, fei = row

        if len(time) > 0:
            steps = int(round((ti - time[-1]) / timestep * factor))  # (y2-y1)/(x2-x1)
            if steps > 1:
                fluxstep = (fi - flux[-1]) / steps
                fluxerror_step = (fei - flux_error[-1]) / steps

                # For small gaps, pretend interpolated data is real.
                if steps > 2:
                    set_real = 0
                else:
                    set_real = 1

                for _ in range(steps - 1):
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

    return [np.array(x) for x in [time, flux, quality, real, flux_error]]


def timeout_handler(signum, frame):
    raise TimeoutError("Timeout reached.")


def prepare_lightcurve(file):
    lc, lc_info = import_lightcurve(file, drop_bad_points=True)
    lcc = lc.copy()
    lcc = lcc[lcc["QUALITY"] == 0]
    lcc = lcc["TIME", "PCA_FLUX", "QUALITY", "FLUX_ERR"]
    time, flux, quality, real, flux_error = clean_data(lcc)

    flat_flux = wotan.flatten(
        lc["TIME"], lc["PCA_FLUX"], method="median", window_length=1
    )
    rms = np.nanstd(flat_flux)

    diff = np.diff(lc["TIME"])
    large_gaps_indices = np.where(diff > 1)[0]

    return {
        "lc": lc,
        "lc_info": lc_info,
        "time": time,
        "flux": flux,
        "quality": quality,
        "real": real,
        "flux_error": flux_error,
        "flat_flux": flat_flux,
        "rms": rms,
        "diff": diff,
        "large_gaps_indices": large_gaps_indices,
    }


def SNR(rms, min_snr, max_snr):
    random_snr = np.random.uniform(min_snr, max_snr)
    A = rms * random_snr
    return {"snr": random_snr, "amplitude": A}


def is_valid_t0(t0, time, large_gaps_indices, diff):
    """
    Finds a valid time to insert the transit.

    t0: time of injection
    time: time array
    large_gaps_indices: indices of large gaps identified using np.diff(t)
    diff: the timestep between each cadence

    """
    for index in large_gaps_indices:
        if time[index] - 1 <= t0 <= time[index + 1] + 1:
            return False
        if (
            index < len(time) - 1
            and diff[index] > 0.5
            and abs(t0 - time[index + 1]) < 1.5
        ):
            return False
        if index > 0 and diff[index - 1] > 0.5 and abs(t0 - time[index]) < 1.5:
            return False
    if t0 <= time[0] + 1.5 or t0 >= time[-1] - 1.5:
        return False
    return True


def find_valid_injection_time(lc, window_size, max_attempts=20):
    for _ in range(max_attempts):
        t0 = np.random.uniform(lc["lc"]["TIME"][0], lc["lc"]["TIME"][-1])

        if is_valid_t0(t0, lc["lc"]["TIME"], lc["large_gaps_indices"], lc["diff"]):
            window_start = np.argmin(
                np.abs(lc["time"] - (t0 - window_size * np.median(np.diff(lc["time"]))))
            )
            window_end = (
                np.argmin(
                    np.abs(
                        lc["time"] - (t0 + window_size * np.median(np.diff(lc["time"])))
                    )
                )
                + 1
            )

            ### Checks if all the points in the comet are "real".
            if np.all(lc["real"][window_start:window_end] == 1):
                return {"t0": t0}


def scale_relative_to_baseline(flux):
    baseline = np.median(flux) 
    scaled_flux = (flux - baseline) / baseline
    return (scaled_flux - np.min(scaled_flux)) / (np.max(scaled_flux) - np.min(scaled_flux))

def normalise_depth(flux):
    median = np.median(flux)
    min_flux = np.min(flux)
    abs_depth = median - min_flux
    depth_normalised_lightcurve = ((flux - median) / abs_depth + 1)
    return depth_normalised_lightcurve

def comet(
    file,
    folder,
    min_snr=7,
    max_snr=20,
    window_size=84,
    max_retries=50,
    method=None,
    save_model=True,
):
    """
    Creates a comet profile and injects it into a lightcurve.

    file: path to file
    folder: folder to save the output lightcurve
    min_snr: Minimum SNR (default SNR=5).
    max_snr: Maximum SNR (default SNR=20).
    window_size: Number of cadences representing the window size (default 84, corresponding to 3.5 days)
    max_retries: Maximum number of retries for model creation (default 50)
    method: Method to create the comet model ("comet_curve" or "skewed_gaussian")
    save_model: Saves the lightcurve model when exporting the `.npy` file too.
    """
    lc = prepare_lightcurve(file)

    if np.isnan(lc["rms"]):
        return None

    snr = SNR(lc["rms"], min_snr, max_snr)

    valid_model_found = False
    retry_count = 0

    while not valid_model_found and retry_count < max_retries:
        injection_time = find_valid_injection_time(lc, window_size)
        
        if injection_time is None:
            retry_count += 1
            continue

        t0 = injection_time["t0"]

        if method == "comet_curve" or method is None:
            sigma = np.round(np.random.uniform(0.25, 0.7), 3)
            tail = np.round(np.random.uniform(0.35, 0.5), 3)
            shape = np.round(np.random.uniform(1.5, 4), 3)
            model = 1 - models.comet_curve2(lc["time"], snr["amplitude"], t0, sigma=sigma, tail=tail, shape=shape)
        
        elif method == "skewed_gaussian":
            skew = 3
            duration = 0.2
            model = models.skewed_gaussian(lc["time"], alpha=skew, t0=t0, sigma=duration, depth=snr["amplitude"])

        f = model * (lc["flux"] / np.nanmedian(lc["flux"]))
        f = normalise_depth(f)

        valid_model_found = True

    if not valid_model_found:
        print(f"Failed to create a valid model for file {file} after {max_retries} attempts. Skipping...")
        return None

    fluxerror = lc["flux_error"] / lc["flux"]

    sector = f"{lc['lc_info']['sector']:02d}"

    if save_model:
        np.save(
            f"{folder}/{lc['lc_info']['TIC_ID']}_sector{sector}_{args.transit}.npy",
            np.array(
                [
                    lc["time"][lc["real"] == 1],
                    f[lc["real"] == 1],
                    fluxerror[lc["real"] == 1],
                    lc["real"][lc["real"] == 1],
                    model[lc["real"] == 1],
                ]
            ),
        )
    else:
        np.save(
            f"{folder}/{lc['lc_info']['TIC_ID']}_sector{sector}_{args.transit}.npy",
            np.array(
                [
                    lc["time"][lc["real"] == 1],
                    f[lc["real"] == 1],
                    fluxerror[lc["real"] == 1],
                    lc["real"][lc["real"] == 1],
                ]
            ),
        )

    return [{"tic": lc['lc_info']['TIC_ID'], "time": t0, "snr": snr, "rms": lc['rms']}]

def exoplanet(file, folder, m_star, r_star, period_min=3, period_max=700, binary=False):
    min_snr = 3
    max_snr = 20
    window_size = 84
    max_retries = 50

    try:
        # Read in lightcurve
        lc = prepare_lightcurve(file)
        sector = f"{lc['lc_info']['sector']:02d}"
        snr = SNR(lc["rms"], min_snr, max_snr)

        valid_model_found = False
        retry_count = 0

        while not valid_model_found and retry_count < max_retries:
            try:
                injection_time = find_valid_injection_time(lc, window_size)
                
                if injection_time is None:
                    retry_count += 1
                    continue

                t0 = injection_time["t0"]

                # Create transit model
                params = batman.TransitParams()
                params.t0 = t0
                random_value = np.random.uniform(0, 1)

                if binary:
                    params.u = [np.random.uniform(0.1, 0.9)]
                    alpha = 1.7  # a wider spread of periods for binaries
                else:
                    params.u = [np.random.uniform(0.2, 0.8)] 
                    alpha = 1.2  # slightly more biased to shorter periods for exoplanets

                params.per = period_min * (period_max / period_min) ** (random_value ** (1 / alpha))
                params.rp = np.sqrt(snr['amplitude'])
                params.a = ((params.per * 86400.) ** 2 * const.G.value * m_star * const.M_sun.value / 
                            (4 * np.pi**2)) ** (1/3) / (r_star * const.R_sun.value)
                params.inc = 90
                params.ecc = 0
                params.w = 90
                params.limb_dark = "linear"

                m = batman.TransitModel(params, lc['time'], fac=0.02)
                model = m.light_curve(params)

                injected_flux = model * (lc['flux'] / np.nanmedian(lc['flux']))
                injected_flux = normalise_depth(injected_flux)

                if np.all(injected_flux >= 0):
                    valid_model_found = True
                else:
                    
                    plt.show()
                    break
                    retry_count += 1
                    

            except Exception as e:
                if "Convergence failure" in str(e):
                    retry_count += 1
                else:
                    raise e

        if not valid_model_found:
            print(f"Failed to create a valid model after {max_retries} attempts. Skipping...")
            return None

        fluxerror = np.array(lc["flux_error"]) / np.nanmedian(lc["flux"])
        tic = lc["lc_info"]["TIC_ID"]
        np.save(f"{folder}/{tic}_sector{sector}_{args.transit}.npy", 
                np.array([lc['time'][lc['real'] == 1], injected_flux[lc['real'] == 1], fluxerror[lc['real'] == 1], lc['real'][lc['real'] == 1], model[lc['real'] == 1]]))

        return [{"tic": tic, "time": t0, "snr": snr, "rms": lc['rms']}]

    except Exception as e:
        print(f"Exception occurred: {e}. Continuing...")
        return None

def is_valid_sine_time(t, time, window_size=4):
    if t < time[0] + window_size or t > time[-1] - window_size:
        return False
    
    window_start = np.searchsorted(time, t - window_size/2)
    window_end = np.searchsorted(time, t + window_size/2)
    
    min_points = 160
    if window_end - window_start < min_points:
        return False
    
    return True


def sines(file, folder, min_period=1.25, max_period=3, 
                       min_amplitude=0.005, max_amplitude=0.01, 
                       prominence_factor=0.01, min_distance_days=1):
    
    max_attempts = 5
    
    lc = prepare_lightcurve(file)
    time = lc['time']
    flux = lc['flux'] / np.nanmedian(lc['flux'])
    num_waves = 1 #np.random.randint(1, 3)
    
    # Inject sine waves
    for _ in range(num_waves):
        period = np.random.uniform(min_period, max_period)
        amplitude = np.random.uniform(min_amplitude, max_amplitude)
        phase = np.random.uniform(0, 2*np.pi)
        
        sine_wave = amplitude * np.sin(2 * np.pi * time / period + phase)
        
        flux *= (1 + sine_wave)
    
    ### Written this way to make it easier to use the other normalisation methods if desired
    normalised_flux = normalise_depth(flux) 
    
    #def find_troughs(time, flux, min_distance_days=1, prominence_factor=0.01):
    # Invert the flux to turn troughs into peaks
    inverted_flux = -flux

    # Smooth the inverted flux
    smoothed_flux = savgol_filter(inverted_flux, window_length=11, polyorder=3)

    # Estimate signal properties
    amplitude = np.max(smoothed_flux) - np.min(smoothed_flux)
    timestep = np.median(np.diff(time))
    distance = int(min_distance_days / timestep)


    prominence = max(prominence_factor, amplitude * 0.1)

    # Find peaks (troughs in original signal)
    troughs, properties = find_peaks(smoothed_flux, 
                                     prominence=prominence, 
                                     distance=distance, 
                                     width=10)


    times = time[troughs]

    if len(times) > 5:
        times = np.random.choice(times, 5, replace=False)

    valid_times = [t for t in times if is_valid_sine_time(t, lc['lc']['TIME'])]


    # valid_times = []
    # invalid_times = []
    # for t in times:
    #     if is_valid_t0(t, lc['lc']['TIME'], lc['large_gaps_indices'], lc['diff']):
    #         valid_times.append(t)
    #     else:
    #         invalid_times.append(t)


    tic = lc["lc_info"]["TIC_ID"]
    sector = f"{lc['lc_info']['sector']:02d}"


    np.save(f"{folder}/{tic}_sector{sector}_{args.transit}.npy", 
        np.array([lc['time'], normalised_flux, lc['flux_error'], lc['real']]))
    
    return [{"tic": tic, "time": t, "snr": None, "rms": None} for t in valid_times]


def process_results(results):
    tic = []
    times = []
    snr_cat = []
    rms_cat = []
    for result in results:
        tic.append(result["tic"])
        times.append(result["time"])
        snr_cat.append(result["snr"])
        rms_cat.append(result["rms"])
    return tic, times, snr_cat, rms_cat

def load_ds():
    ds = stella.FlareDataSet(args.folder,args.catalog,frac_balance=1,cadences=168)

    return ds


def save_training_set_plots(ds, folder_name):
    """plots the training set data as the stella input lightcurves (i.e: the chopped up lightcurves)"""

    os.makedirs(folder_name, exist_ok=True)

    ind_pc = np.where(ds.train_labels == 1)[0]
    ind_nc = np.where(ds.train_labels != 1)[0]

    ### PLOTTING POSITIVE CLASS
    data = ds.train_data[ind_pc]


    num_sets = data.shape[0] // 100

    for set_index in tqdm(range(num_sets),desc='Saving plots'):
        start_index = set_index * 100
        end_index = min((set_index + 1) * 100, data.shape[0])

        fig, axs = plt.subplots(10, 10, figsize=(20, 20))
        axs = axs.flatten()

        for i in range(start_index, end_index):
            plot_index = i % 100
            axs[plot_index].plot(data[i, :, 0])
            axs[plot_index].set_title(f"Plot {i}")

        for j in range(end_index - start_index, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.savefig(f'{folder_name}/{start_index}-{end_index}.png', dpi=200, bbox_inches='tight')
        plt.close()

def main(args):
    # files = # Your list of files or target IDs

    files = glob(f"{args.dir}/*.fits")
    random.shuffle(files)

    os.makedirs(args.folder, exist_ok=True)


    if args.transit != "exocomet":
        TIC_table = Catalogs.query_object(f'TIC 270577175', catalog="TIC")
        r_star = TIC_table['rad'][0]
        m_star = TIC_table['mass'][0]
        del TIC_table

    failed_ids = []
    # Map model names to functions
    model_functions = {
        "exocomet": lambda target_ID: comet(target_ID, folder=args.folder),
        "exoplanet": lambda target_ID: exoplanet(target_ID, folder=args.folder,r_star=r_star, m_star=m_star),
        "binary": lambda target_ID: exoplanet(
            target_ID, folder=args.folder, r_star=r_star,m_star=m_star,binary=True
        ),
        "sines": lambda target_ID: sines(target_ID, folder=args.folder)
    }

    tic = []
    times = []
    snr_cat = []
    rms_cat = []

    for target_ID in tqdm(files[0 : args.number]):
        if args.transit in model_functions:
            try:
                results = model_functions[args.transit](target_ID)
                new_tic, new_times, new_snr, new_rms = process_results(results)
                tic.extend(new_tic)
                times.extend(new_times)
                snr_cat.extend(new_snr)
                rms_cat.extend(new_rms)
            except Exception as e:
                print(f"Failed for TIC {target_ID}: ", e)
                failed_ids.append(target_ID)
                continue

    data = pd.DataFrame(data=[tic, times, snr_cat, rms_cat]).T
    data.columns = ["TIC", "tpeak", "SNR", "RMS"]
    data.TIC = data.TIC.astype(int)
    t = Table.from_pandas(data)
    t.write(f"{args.catalog}", format="ascii", overwrite=True)

    if len(failed_ids) > 0:
        print(f"Failed IDs: {len(failed_ids)}")


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create exocomet/exoplanet/eclipsing binary transits."
    )

    parser.add_argument(
        help="The target directory of lightcurves to use for model injection.",
        dest="dir",
    )
    parser.add_argument("-f", "--folder", help="target output folder.", dest="folder")
    parser.add_argument(
        "-c",
        "--catalog-name",
        help="target catalog file. Saved in a .txt format",
        dest="catalog",
    )

    parser.add_argument("-n", "--number", default=5000, dest="number", type=int)

    parser.add_argument(
        "-t",
        "--transit-type",
        help='Select the transit type. Options: "exocomet", "exoplanet","binary","sines". Default is "exocomet".',
        dest="transit",
        default='exocomet'
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Select the model used to create exocomets. Options: 'comet_curve', 'skewed_gaussian'. Default 'comet_curve'.",
        dest="model",
    )

    parser.add_argument(
        "-pf",
        "--plot-folders",
        help="Saves the training set plots in the specified folder.",
        dest="plotfoldername",
    )

    args = parser.parse_args()

    main(args)
    ds = load_ds()
    save_training_set_plots(ds,folder_name=args.plotfoldername)
    print("Injections complete.")
