"""
Creates models of exocomets/exoplanets/eclipsing binaries, and injecting them into real lightcurves. 
"""

import os, sys
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
from scipy.stats import skewnorm

import models


current_dir = os.getcwd()
while os.path.basename(current_dir) != 'nets2':
    current_dir = os.path.dirname(current_dir)
    if current_dir == os.path.dirname(current_dir): 
        raise Exception("'nets2' directory not found in parent directories")
    
sys.path.insert(1, os.path.join(current_dir, 'scripts'))
sys.path.insert(1, os.path.join(current_dir, 'stella'))

from utils import *


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
                if steps > 3:
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


def find_valid_injection_time(lc, window_size, max_attempts=100):
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


def comet(
    file,
    folder,
    min_snr=5,
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
    window_size: Half of the window size of the lightcurve cutout to use for CNN (default 84, corresponding to a total of 128 cadences (3.5 days).)
    save_model: Saves the lightcurve model when exporting the `.npy` file too.

    """
    lc = prepare_lightcurve(file)

    if np.isnan(lc["rms"]):
        return None

    snr = SNR(lc["rms"], min_snr, max_snr)

    injection_time = find_valid_injection_time(lc, window_size)

    retries = 0
    while retries < max_retries:
        injection_time = find_valid_injection_time(lc, window_size)
        if injection_time["t0"] is None:
            retries += 1
            continue
        else:
            break


#     model_functions = {
#     "comet_curve": lambda lc, snr, t0: 1 - models.comet_curve(lc["time"], snr["amplitude"], t0["t0"]),
#     "skewed_gaussian": lambda lc, snr, t0: models.skewed_gaussian(
#         lc["time"],
#         depth=snr["amplitude"],
#         alpha=int(np.random.uniform(1, 4)),
#         sigma=0.74,
#         t0=t0["t0"],
#     ),
# }


    if method == "comet_curve" or method is None:
        shape = np.round(np.random.uniform(1.5,4),3)
        sigma = np.round(np.random.uniform(0.25,0.7),3)
        tail = np.round(np.random.uniform(0.35,0.5),3)

        ## t0, A, sigma, tail, shape
        model = 1 - models.comet_curve2(
            lc["time"], snr["amplitude"], injection_time["t0"], sigma = sigma, tail = tail, shape = shape)
        
    elif method == "skewed_gaussian":
        alpha = int(np.random.uniform(1, 4))
        model = models.skewed_gaussian(
            lc["time"],
            depth=snr["amplitude"],
            alpha=alpha,
            sigma=0.74,
            t0=injection_time["t0"],
        )  ## sigma can change

    f = model * (lc["flux"] / np.nanmedian(lc["flux"]))

    ### PERFORM SCALING


    fluxerror = lc["flux_error"] / lc["flux"]

    sector = f"{lc['lc_info']['sector']:02d}"

    if save_model:
        np.save(
            f"{folder}/{lc['lc_info']['TIC_ID']}_sector{sector}_model.npy",
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
            f"{folder}/{lc['lc_info']['TIC_ID']}_sector{sector}.npy",
            np.array(
                [
                    lc["time"][lc["real"] == 1],
                    f[lc["real"] == 1],
                    fluxerror[lc["real"] == 1],
                    lc["real"][lc["real"] == 1],
                ]
            ),
        )

    return lc["lc_info"]["TIC_ID"], injection_time["t0"], snr["snr"], lc["rms"]


def exoplanet(
    file,
    folder,
    min_snr=5,
    max_snr=20,
    window_size=84,
    period_min=3,
    period_max=700,
    alpha=1.7,
    m_star=1,
    r_star=1,
    max_retries=50,
    retry_delay=1,
    timeout_duration=2,
    binary=False,
):

    lc = prepare_lightcurve(file)

    snr = SNR(lc["rms"], min_snr, max_snr)

    retries = 0
    while retries < max_retries:
        try:
            params = batman.TransitParams()
            random_value = np.random.uniform(0, 1)
            params.per = period_min * (period_max / period_min) ** (
                random_value ** (1 / alpha)
            )

            injection_time = find_valid_injection_time(lc, window_size)
            if injection_time["t0"] is None:
                retries += 1
                continue

            params.t0 = injection_time["t0"]
            params.rp = np.sqrt(snr["amplitude"])
            params.a = (
                (
                    (
                        const.G.value
                        * m_star
                        * const.M_sun.value
                        * (params.per * 86400.0) ** 2
                    )
                    / (4.0 * (np.pi**2))
                )
                ** (1.0 / 3)
            ) / (r_star * const.R_sun.value)
            params.inc = 90
            params.ecc = 0.0
            params.w = 90.0
            params.limb_dark = "linear"
            if binary:
                params.u = [np.random.uniform(0.1, 0.9)]
            else:
                params.u = [np.random.uniform(0.2, 0.8)]

            m = batman.TransitModel(params, lc["time"], fac=0.02)
            batman_flux = m.light_curve(params)

            if np.min(batman_flux) == 1:
                retries += 1
                continue

            injected_flux = batman_flux * (lc["flux"] / np.nanmedian(lc["flux"]))
            fluxerror = np.array(lc["flux_error"]) / np.nanmedian(lc["flux"])

            sector = f"{lc['lc_info']['sector']:02d}"
            np.save(
                f"{folder}/{lc['lc_info']['TIC_ID']}_sector{sector}.npy",
                np.array(
                    [
                        lc["time"][lc["real"] == 1],
                        injected_flux[lc["real"] == 1],
                        fluxerror[lc["real"] == 1],
                        lc["real"][lc["real"] == 1],
                    ]
                ),
            )

            # return {
            #     "t0": params.t0,
            #     "tic_id": lc["lc_info"]["TIC_ID"],
            #     "time": lc["time"][lc["real"] == 1],
            #     "injected_flux": injected_flux[lc["real"] == 1],
            #     "fluxerror": fluxerror[lc["real"] == 1],
            #     "period": params.per,
            # }

        except Exception as e:
            retries += 1
            if retries == max_retries:
                return None
            ti.sleep(retry_delay)

    return lc["lc_info"]["TIC_ID"], injection_time["t0"], snr["snr"], lc["rms"]


def main(args):
    # files = # Your list of files or target IDs

    files = glob(f"{args.dir}/*.fits")
    random.shuffle(files)

    os.makedirs(args.folder, exist_ok=True)

    results = []
    failed_ids = []
    # Map model names to functions
    model_functions = {
        "exocomet": lambda target_ID: comet(target_ID, folder=args.folder),
        "exoplanet": lambda target_ID: exoplanet(target_ID, folder=args.folder),
        "binary": lambda target_ID: exoplanet(
            target_ID, folder=args.folder, binary=True
        ),
    }

    tic = []
    times = []
    snr_cat = []
    rms_cat = []

    for target_ID in tqdm(files[0 : args.number]):

        if args.transit in model_functions:
            #try:
            tic_id, time, snrs, rms = model_functions[args.transit](target_ID)
            tic.append(tic_id)
            times.append(time)
            snr_cat.append(snrs)
            rms_cat.append(rms)
            #except Exception as e:
            #    failed_ids.append(target_ID)

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

    parser.add_argument("--number", default=5000, dest="number", type=int)

    parser.add_argument(
        "-t",
        "--transit-type",
        help='Select the transit type. Options: "exocomet", "exoplanet","binary".',
        dest="transit",
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Select the model used to create exocomets. Options: 'comet_curve', 'skewed_gaussian'. Default 'comet_curve'.",
        dest="model",
    )

    args = parser.parse_args()

    main(args)
    print("Injections complete.")
