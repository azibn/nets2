"""
Creates models of exocomets/exoplanets/eclipsing binaries, and injecting them into real lightcurves.
"""

import os, sys
import numpy as np
import build_synthetic_set as models
import matplotlib.pyplot as plt
from astropy.table import Table
from glob import glob
from tqdm import tqdm
import wotan
import lightkurve as lk
import random
import batman
import warnings
import argparse
import astropy.constants as const
import time as ti
import signal

sys.path.insert(1, "/Users/azib/Documents/open_source/nets2/stella/")
sys.path.insert(1, "/Users/azib/Documents/open_source/nets2/scripts/")
from utils import *


parser = argparse.ArgumentParser(
    description="Create exocomet/exoplanet/eclipsing binary transits."
)
parser.add_argument(
    description="The target directory of lightcurves to use for model injection.",
    dest="dir",
)
parser.add_argument(
    "-of", "--output-folder", description="target output folder.", dest="of"
)
parser.add_argument("--number", default=5000, dest="number")

parser.add_argument(
    "-m",
    "--model",
    description='Select the model type. Options: "exocomet", "exoplanet","binary".',
    dest="model",
)


args = parser.parse_args()

files = glob(f"{args.dir}/**/*.fits", recursive=True)
random.shuffle(files)

os.makedirs(args.of, exist_ok=True)

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
                if steps > 4:
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
            if np.all(lc["real"][window_start:window_end] == 1):
                return {"t0": t0}


def comet(file, folder, min_snr=3, max_snr=20, window_size=84):
    """
    Creates a comet profile and injects it into a lightcurve.

    file: path to file
    folder: folder to save the output lightcurve
    min_snr: Minimum SNR (default SNR=3).
    max_snr: Maximum SNR (default SNR=20).
    window_size: Half of the window size of the lightcurve cutout to use for CNN (default 84, corresponding to a total of 128 cadences (3.5 days).)

    """
    lc = prepare_lightcurve(file)

    if np.isnan(lc["rms"]):
        return None

    snr = SNR(lc["rms"], min_snr, max_snr)

    injection_time = find_valid_injection_time(lc, window_size)
    if injection_time["t0"] is None:
        return None

    model = 1 - models.comet_curve(lc["time"], snr["amplitude"], injection_time["t0"])

    f = model * (lc["flux"] / np.nanmedian(lc["flux"]))
    fluxerror = lc["flux_error"] / lc["flux"]

    sector = f"{lc['lc_info']['sector']:02d}"
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

    return injection_time["t0"]


def exoplanet(
    file,
    min_snr=5,
    max_snr=20,
    window_size=84,
    period_min=3,
    period_max=700,
    alpha=1.7,
    m_star=1,
    r_star=1,
    max_retries=5,
    retry_delay=1,
    timeout_duration=30,
    binary=False
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

            return {
                "t0": params.t0,
                "tic_id": lc["lc_info"]["TIC_ID"],
                "time": lc["time"][lc["real"] == 1],
                "injected_flux": injected_flux[lc["real"] == 1],
                "fluxerror": fluxerror[lc["real"] == 1],
                "period": params.per,
            }

        except Exception as e:
            retries += 1
            if retries == max_retries:
                return None
            ti.sleep(retry_delay)

    return None


def main():
    # files = # Your list of files or target IDs
    results = []
    failed_iterations = 0

    # Map model names to functions
    model_functions = {
        'exocomet': comet,
        'exoplanet': exoplanet,
        'binary': lambda target_ID, files: exoplanet(target_ID, files, binary=True)

        # Add more models here if needed
    }

    # Get the function for the selected model
    model_function = model_functions.get(args.model)

    if model_function is None:
        raise ValueError(f"Unknown model: {args.model}")
        return

    for target_ID in tqdm(files[0:2000]):
        result = model_function(target_ID, files)
        if result:
            results.append(result)
            # Save the data
        else:
            failed_iterations += 1

    print(f"Completed. Failed iterations: {failed_iterations}")
    return results


if __name__ == "__main__":
    main()
