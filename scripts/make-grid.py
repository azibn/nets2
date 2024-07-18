"""This script makes a bunch of lightcurve models based on skewed Gaussian curves (these models are separate from the training set) based on skew, duration (sigma) and snr."""




import argparse
import os
import sys
import itertools
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skewnorm

sys.path.insert(1, "/Users/azib/Documents/open_source/nets2/scripts")
sys.path.insert(1, "/Users/azib/Documents/open_source/nets2/stella")
from utils import *
from astropy.table import Table
from glob import glob
from wotan import flatten

parser = argparse.ArgumentParser(
    description="Runs an injection recovery CNN thing on a parameter space defined by user."
)
# parser.add_argument(help="Target directory", dest="path")
# parser.add_argument(help="catalog", dest="catalog")
parser.add_argument(
    "-n",
    "--number",
    help="Number of models to create",
    type=int,
    dest="n",
    default=1000,
)

parser.add_argument(
    "-s",
    "--save",
    help="Save the models. Default False.",
    action="store_true",
    dest="s",
)
parser.add_argument(
    "-sp",
    "--save-path",
    help='Path to directory to save models. Default is "."',
    default=".",
    dest="sp",
)

parser.add_argument(
    "-t",
    "--types",
    help='Type of injection. Options are "noiseless" or "injected". Default is "injected"',
    default="injected",
    dest="types",
)

args = parser.parse_args()


def skewed_gaussian(x, alpha=1, t0=1496.5, sigma=1, A=0.001):
    """Creates a skewed Gaussian model transit.

    Parameters:
        x: Time array.
        alpha: Skewness parameter (0 for symmetric).
        t0: Mean of the Gaussian.
        sigma: Standard deviation of the Gaussian.
        A: Amplitude of the Gaussian.

    Returns:
        y: The value of the skewed Gaussian at each input data point x.
    """
    pdf = skewnorm.pdf(x, alpha, loc=t0, scale=sigma)
    normalized_pdf = pdf / pdf.max()
    return 1 - A * normalized_pdf


skew = np.arange(-15, 15, 1)
# duration_range = np.arange(0.1,1.2,0.1) ## linear space
min_duration = 0.1
max_duration = 1.3
num_duration_points = 10
duration_range = np.logspace(
    np.log10(min_duration), np.log10(max_duration), num_duration_points
)  ## logspace


def generate_models(n_models=args.n, save=args.s, save_path=args.sp, types=args.types):
    """Generates a set of skewed Gaussian models.

    Params:
        n_models: Number of models you want to create. Default is 1000 per combination.
        save: Option to save the models in directory. Default is True.
        save_path: Path to directory you want to save the models. Default is '.'

    Returns:
        models: The models returned as a list.

    """
    # models = []
    id = []
    snr_range = np.arange(3, 15, 2)
    # types = ['noiseless','injected']

    if types == "noiseless":
        for i, (skewness, duration, model_index) in tqdm.tqdm(
            enumerate(itertools.product(skew, duration_range, range(n_models)))
        ):
            time = np.load("time-array-s7.npy")
            duration = np.round(duration, 2)
            flux = skewed_gaussian(
                time, alpha=skewness, t0=1500, sigma=duration, A=0.001
            )
            fluxerror = np.zeros(len(flux))

            if save:
                cfolder = f"skew_{skewness}-duration-{duration}-snr-{snr}"
                folder_path = os.path.join(save_path, cfolder)
                os.makedirs(folder_path, exist_ok=True)
                file_path = os.path.join(
                    folder_path, f"exocomet_model_{model_index}.npy"
                )
                np.save(file_path, np.array([time, flux, fluxerror]))

            id.append(i)

    elif types == "injected":
        print("globbing directory for real lightcurves")
        lightcurves = glob("data/eleanor/s0007/*.fits", recursive=True)
        lightcurves = np.random.choice(lightcurves, size=5000, replace=False)
        print("globbing complete.")

    for skewness in skew:
        for duration in duration_range:
            for snr in snr_range:
                for _ in tqdm.tqdm(
                    range(n_models),
                    desc=f"Skew: {skewness}, Duration: {duration:.2f}, SNR: {snr}",
                ):
                    lightcurve = np.random.choice(
                        lightcurves
                    )  # Randomly choose a lightcurve for each model
                    lc, lc_info = import_lightcurve(lightcurve, drop_bad_points=True)
                    duration = np.round(duration, 2)
                    time, flux, fluxerror, model = exocomet(
                        lc=lc,
                        lc_info=lc_info,
                        snr=snr,
                        duration=duration,
                        t0=1496.5,
                        folder="exocomet-grid-models",
                        skew=skewness,
                    )

                    if save:
                        cfolder = f"injected-skew_{skewness}-duration-{duration:.2f}-snr-{snr}"
                        folder_path = os.path.join(save_path, cfolder)
                        os.makedirs(folder_path, exist_ok=True)
                        file_path = os.path.join(
                            folder_path,
                            f"exocomet_model_{lc_info['TIC_ID']}_sector07.npy",
                        )
                        np.save(file_path, np.array([time, flux, fluxerror, model]))

                    id.append(lc_info["TIC_ID"])

    catalogtime = np.full_like(time, 1500)
    df = create_catalog(id, catalogtime)

    # return models


def create_catalog(id, time, catalog_name="catalog.txt"):
    df = pd.DataFrame(data=[id, time]).T
    df.columns = ["TIC", "tpeak"]
    # df.to_csv(f"{catalog_name}",index=None)
    table = Table.from_pandas(df)
    table.write(f"{catalog_name}", format="ascii", overwrite=True)

    return df


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


def exocomet(
    lc, lc_info, snr, duration, t0=1500, folder="exocomet-grid-models", skew=skew
):

    # os.makedirs(folder, exist_ok=True)
    fails = []
    times = []
    rmsfails = []
    ticid = []
    snrlist = []
    rms_cat = []

    window_size = 84  # Number of cadences representing the window size (3.5 days)

    ### FLATTEN THE ORIGINAL LIGHTCURVE
    flat_flux = flatten(lc["TIME"], lc["PCA_FLUX"], method="median", window_length=1)

    ### GET RMS OF FLATTENED ORIGINAL LIGHTCURVE
    rms = np.nanstd(flat_flux)

    ### CREATE COPY OF LIGHTCURVE
    lcc = lc.copy()
    lcc = lcc[lcc["QUALITY"] == 0]
    tic = lc_info["TIC_ID"]
    lcc = lcc["TIME", "PCA_FLUX", "QUALITY", "FLUX_ERR"]

    ### INTERPOLATE THE COPIED LIGHTCURVE
    time, flux, quality, real, flux_error = clean_data(lcc)

    A = rms * snr  # gives the depth of the transit given the SNR

    # Check if all data points within the window are non-interpolated
    window_start = np.argmin(
        np.abs(time - (t0 - window_size * np.median(np.diff(time))))
    )
    window_end = (
        np.argmin(np.abs(time - (t0 + window_size * np.median(np.diff(time))))) + 1
    )
    if np.all(real[window_start:window_end] == 1):
        valid_time_found = True

    ### CREATE MODEL BASED ON THE INTERPOLATED LIGHTCURVE TIME ARRAY

    model = skewed_gaussian(time[real == 1], alpha=skew, t0=t0, sigma=duration, A=A)

    ### INJECT MODEL INTO INTERPOLATED LIGHTCURVE
    f = model * (flux[real == 1] / np.nanmedian(flux[real == 1]))
    fluxerror = flux_error[real == 1] / flux[real == 1]

    # assert f>= 0, "All data points must be above 0."

    ### APPEND TIMES AND TIC ID FOR THE CATALOG
    times.append(t0)
    ticid.append(tic)
    snrlist.append(snr)
    rms_cat.append(rms)

    return time[real == 1], f, fluxerror, model
    ### SAVE INTO NUMPY FOLDER
    # np.save(f"{folder}/{lc_info['TIC_ID']}_sector{sector}.npy",np.array([time[real == 1], f[real == 1], fluxerror[real == 1], real[real == 1],model[real == 1]]))


if __name__ == "__main__":

    generate_models(args.n)
    #sys.exit()

    # Generate the combinations
    combinations = list(itertools.product(skew, duration_range, range(args.n)))

    # # Extract the skewness and duration values
    # skew_values = [comb[0] for comb in combinations]
    # duration_values = [comb[1] for comb in combinations]

    # # Create a 2D histogram
    # hist, xedges, yedges = np.histogram2d(
    #     skew_values, duration_values, bins=[len(skew), len(duration_range)]
    # )

    # # Plot the heatmap
    # plt.figure(figsize=(8, 6))
    # plt.imshow(
    #     hist,
    #     interpolation="nearest",
    #     origin="lower",
    #     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    # )
    # plt.colorbar(label="Number of iterations")
    # plt.xlabel("Skewness")
    # plt.ylabel("Duration")
    # plt.grid(True)
    # plt.title("Number of iterations per skewness-duration combination")
    # plt.show()


# def inject_model():
#     """Injects a model transit into a light curve."""

# ## Make a for loop that loops over each "skewness" parameter and the lightcurve duration between a range of -5 -> 5 skewness
# ## and 1 hour duration to 1.5 days. I think this will be a double for loop

# for i, skewness in enumerate(skew):
#     for j, duration in enumerate(duration_range):
#         recovered = 0

#         for _ in range(1000):  # Loop over 1000 random lightcurves
#             # Generate a random lightcurve (you'll need to implement this)
#             lc, lc_info = import_lightcurve()

#             # Generate the skewnorm Gaussian model
#             model = skewed_gaussian(lc['TIME'], amplitude=1, duration=duration, skewness=skewness, loc=1416)

#             # Inject model and check if recovered
#             if inject_and_recover(lightcurve, model):
#                 recovered += 1

#         # Calculate and store the recovery fraction
#         recovery_fractions[i, j] = recovered / 1000

# # Plot the results
# plt.imshow(recovery_fractions, extent=[-5, 5, 1, 36], aspect='auto', origin='lower')
# plt.colorbar(label='Recovery Fraction')
# plt.xlabel('Skewness')
# plt.ylabel('Duration (hours)')
# plt.title('Recovery Fraction vs Skewness and Duration')
# plt.show()
