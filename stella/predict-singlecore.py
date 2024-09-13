import sys
import os
import pickle
import glob
import time
import argparse
import numpy as np
import stella


current_dir = os.getcwd()
while os.path.basename(current_dir) != 'nets2':
    current_dir = os.path.dirname(current_dir)
    if current_dir == os.path.dirname(current_dir): 
        raise Exception("'nets2' directory not found in parent directories")
    
sys.path.insert(1, os.path.join(current_dir, 'scripts'))
sys.path.insert(1, os.path.join(current_dir, 'stella'))


from tensorflow import keras
from utils import *
os.nice(7)


parser = argparse.ArgumentParser(description="Predict CNN on lightcurve data")
parser.add_argument(help="Target directory of lightcurves", dest="path")
parser.add_argument("-m", "--model", type=str, help="Path to model")
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help='Name of output file in pickle format. Default is "output.pkl"',
    default="output.pkl",
    dest="o",
)
parser.add_argument(
    "-p",
    "--pipeline",
    type=str,
    help='Lightcurve pipeline. Default is "eleanor-lite"',
    default="eleanor-lite",
    dest="p",
)
parser.add_argument(
    "-th",
    "--threshold",
    type=float,
    help="Threshold for interesting predictions",
    default=0.7,
)


args = parser.parse_args()

### PIPELINE OPTIONS
PIPELINE = {
    "eleanor-lite": {
        "time": "TIME",
        "flux": "PCA_FLUX",
        "flux_err": "FLUX_ERR",
        "id": "TIC_ID",
    },
    "SPOC": {
        "time": "TIME",
        "flux": "PDCSAP_FLUX",
        "flux_err": "PDCSAP_FLUX_ERR",
        "id": "TICID",
    },

    "K2": {"time": "TIME", "flux": "FLUX", "flux_err": "FRAW_ERR", "id": "KEPLERID"},
    # CONSIDER CHANGING FLUX TO FCOR (THE CBV DETRENDED FLUX)
    # Add more pipeline configurations as needed
}


def load_model(cnn, model):
    return cnn.load_model(model)


def load_lightcurves(path):
    print("globbing files")
    return glob.glob(f"{path}/**/*.fits", recursive=True)


def process_lightcurve(path, pipeline):
    lc, info = import_lightcurve(path)

    # def get_column(lc, primary, fallback):
    #     """Fallback option if primary column is not found as e.g: some SPOC lightcurves
    #     do not have a PDCSAP."""
    #     return np.array(lc[primary] if primary in lc.columns else lc[fallback])

    # Use the function to get the columns
    # time = get_column(lc, pipeline['time'], 'TIME')
    # flux = get_column(lc, pipeline['flux'], 'FLUX')
    # flux_error = get_column(lc, pipeline['flux_err'], 'FLUX_ERR')
    time, flux, flux_error = (
        lc[pipeline["time"]],
        lc[pipeline["flux"]],
        lc[pipeline["flux_err"]],
    )
    time, flux, flux_error = scale_lightcurve(time, flux, flux_error)
    return info[pipeline["id"]], time, flux, flux_error


def scale_lightcurve(time, flux, flux_error):
    # Calculate the scaled flux
    f = np.array((flux / np.nanmedian(flux)) - 1)
    mask = ~np.isnan(flux)
    t = time[mask]
    f = f[mask]
    flux_error = flux_error[mask]

    f = (f - np.min(f)) / (np.max(f) - np.min(f))

    return t, f, flux_error


def load_predictions(file_path):
    """Load the pickle file"""
    data = []
    with open(file_path, "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    return data


if __name__ == "__main__":

    start_time = time.time() 
    with open("ds.pkl", "rb") as file:
        ds = pickle.load(file)

    cnn = stella.ConvNN(
        output_dir="/Users/azib/Documents/open_source/nets2/cnn-models/", ds=ds
    )
    load_model(cnn, args.model)

    lightcurves = load_lightcurves(args.path)
    pred_t = []
    pred = []

    pipeline = PIPELINE[args.p]
    print("Lightcurve data product: ", pipeline)

    with open(args.o, "ab") as file:
        for lc in lightcurves:

            try:
                source_id, t, flux, flux_error = process_lightcurve(lc, pipeline)

                try:
                    cnn.predict(
                        modelname=args.model, times=t, fluxes=flux, errs=flux_error
                    )
                except ValueError:
                    print("Error predicting lightcurve: empty.")
                    continue

                arg = np.argmax(cnn.predictions[0])

                #### THIS SAVES ONLY THE HIGHEST PREDICTION, NOT THE WHOLE ARRAY
                # pred.append(cnn.predictions[0][arg])
                # pred_t.append(cnn.predict_time[0][arg])

                pred = cnn.predictions[0][arg]
                t_pred = cnn.predict_time[0][arg]

                # Determine if this prediction is interesting
                is_interesting = 1 if pred > args.threshold else 0

                # Store data as a dictionary
                results = {
                    "ID": source_id,
                    "t_pred": t_pred,
                    "pred": pred,
                    "is_interesting": is_interesting,
                }

                # Include full time and flux arrays only if the prediction is interesting
                if is_interesting:
                    results["time"] = np.array(t)
                    results["flux"] = np.array(flux)
                    results["predictions"] = np.array(cnn.predictions[0])

                # Save the prediction data directly to the pickle file
                pickle.dump(results, file)
                del source_id, t, flux, flux_error, pred, t_pred

            except FileNotFoundError:
                print("File not found")
                continue

    end_time = time.time()  
    elapsed_time = (end_time - start_time) / 60  
    print(f"Script executed in {elapsed_time:.2f} minutes")  

    sys.exit(0)