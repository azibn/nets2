"""This script makes a bunch of lightcurve models based on skewed Gaussian curves (these models are separate from the training set) based on skew, duration (sigma) and snr."""

import argparse
import os
import sys
import itertools
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stella
from scipy.stats import skewnorm
import models as m
import modelmaker
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




skew = np.arange(-15, 15, 1)
# duration_range = np.arange(0.1,1.2,0.1) ## linear space
min_duration = 0.1
max_duration = 1.3
num_duration_points = 10
duration_range = np.logspace(
    np.log10(min_duration), np.log10(max_duration), num_duration_points
)  ## logspace


def initialise_cnn():

    exoplanets = stella.FlareDataSet(
    fn_dir="/Users/azib/Documents/open_source/nets2/models/exoplanets1k/",
    catalog="/Users/azib/Documents/open_source/nets2/catalogs/exoplanets1k.txt",
    cadences=168,
    training=0.8,
    validation=0.1,
    frac_balance=1,
    )
    fbinaries = stella.FlareDataSet(
        fn_dir="/Users/azib/Documents/open_source/nets2/models/binaries1k/",
        catalog="/Users/azib/Documents/open_source/nets2/catalogs/fakebinaries1k.txt",
        cadences=168,
        training=0.8,
        validation=0.1,
        frac_balance=1,
    )
    rbinaries = stella.FlareDataSet(
        fn_dir="/Users/azib/Documents/open_source/nets2/models/binaries-s7/",
        catalog="/Users/azib/Documents/open_source/nets2/catalogs/binaries-catalog-s7.txt",
        cadences=168,
        training=0.65,
        validation=0.1,
        frac_balance=1,
    )
    ds = stella.FlareDataSet(
        fn_dir="/Users/azib/Documents/open_source/nets2/models/comets5k/",
        catalog="/Users/azib/Documents/open_source/nets2/catalogs/comets.txt",
        cadences=168,
        training=0.8,
        validation=0.1,
        merge_datasets=True,
        frac_balance=0.7,
        other_datasets=[exoplanets, fbinaries, rbinaries],
        other_datasets_labels=[2, 3, 4],
        augment_portion=0.4
    )

    cnn = stella.ConvNN(
        output_dir="/Users/azib/Documents/open_source/nets2/cnn-models/", ds=ds
    )

    return cnn

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
            flux = modelmaker.skewed_gaussian(
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

    modelname = "/Users/azib/Documents/open_source/nets2/cnn-models/ensemble_s0002_i0200_b0.7.h5"
    cnn = initialise_cnn()

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
                        skew=skewness,
                    )

                    try:
            
                        cnn.predict(modelname=modelname,times=time, fluxes=flux, errs=fluxerror)
                        
                        time = cnn.predict_time[0]
                        flux = cnn.predict_flux[0]
                        errs = cnn.predict_err[0]
                        predictions = cnn.predictions[0]
                    

                        if save:
                            cfolder = f"injected-skew_{skewness}-duration-{duration:.2f}-snr-{snr}"
                            folder_path = os.path.join(save_path, cfolder)
                            os.makedirs(folder_path, exist_ok=True)
                            file_path = os.path.join(
                                folder_path,
                                f"exocomet_model_{lc_info['TIC_ID']}_sector07.npy",
                            )
                            np.save(file_path, np.array([time, flux, errs, predictions, model]))
                        
                    except IndexError:
                        print("cannot predict. moving on...")

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


def exocomet(
    lc, lc_info, snr, duration, t0=1496.5, skew=skew
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
    time, flux, quality, real, flux_error = modelmaker.clean_data(lcc)

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

    model = 1 - m.comet_curve(time, A=A, t0=t0)
   # model = models.skewed_gaussian(time[real == 1], alpha=skew, t0=t0, sigma=duration, depth=A)

    ### INJECT MODEL INTO INTERPOLATED LIGHTCURVE
    f = model * (flux / np.nanmedian(flux))
    fluxerror = flux_error / flux

    # assert f>= 0, "All data points must be above 0."

    ### APPEND TIMES AND TIC ID FOR THE CATALOG
    times.append(t0)
    ticid.append(tic)
    snrlist.append(snr)
    rms_cat.append(rms)

    return time[real == 1], f[real==1], fluxerror[real==1], model[real == 1]
    ### SAVE INTO NUMPY FOLDER
    # np.save(f"{folder}/{lc_info['TIC_ID']}_sector{sector}.npy",np.array([time[real == 1], f[real == 1], fluxerror[real == 1], real[real == 1],model[real == 1]]))



if __name__ == "__main__":

    generate_models(args.n)
    sys.exit()
