import sys
import os
import pickle
import glob
import time
import argparse
import numpy as np
import stella
import concurrent.futures
from tqdm import tqdm
from itertools import islice
import multiprocessing



current_dir = os.getcwd()
while os.path.basename(current_dir) != 'nets2':
    current_dir = os.path.dirname(current_dir)
    if current_dir == os.path.dirname(current_dir): 
        raise Exception("'nets2' directory not found in parent directories")
    
sys.path.insert(1, os.path.join(current_dir, 'scripts'))
sys.path.insert(1, os.path.join(current_dir, 'stella'))


from utils import *
os.nice(17)

# config = tf.compat.v1.ConfigProto(
#     intra_op_parallelism_threads=40,  # Parallelism within individual operations
#     inter_op_parallelism_threads=2    # Parallelism between independent operations
# )

# # Create a session with the above configuration
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)


parser = argparse.ArgumentParser(description="Predict CNN on lightcurve data")
parser.add_argument(help="Target directory of lightcurves", dest="path")
parser.add_argument("-m", "--model", type=str, help="Path to CNN model directory. Assumes models are in its own directory.", dest="model")
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

parser.add_argument('-t', '--threads', type=int, help='Number of threads to use', default=20, dest='threads')

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

def process_single_lightcurve(lc, pipeline, models, threshold):
    try:
        source_id, time, flux, flux_error = process_lightcurve(lc, pipeline)
        cnn = stella.ConvNN(output_dir=f"{os.path.join(current_dir)}/cnn-models/", ds=ds)

        preds = np.zeros((len(models), len(time)))  
        ### LOOPS OVER CNN MODELS
        for i, model in enumerate(models):
            try:
                cnn.predict(modelname=model, times=time, fluxes=flux, errs=flux_error)
                preds[i] = cnn.predictions[0]
            except ValueError:
                print("Error predicting lightcurve: empty.")
                preds[i] = np.nan


        avg_pred = np.nanmedian(preds, axis=0)
        arg = np.argmax(avg_pred)
        pred = avg_pred[arg]
        t_pred = cnn.predict_time[0][arg]
        is_interesting = 1 if pred > threshold else 0

        results = {
            "ID": source_id,
            "t_pred": t_pred,
            "pred": pred,
            "is_interesting": is_interesting,
        }

        if is_interesting:
            results["time"] = np.array(time)
            results["flux"] = np.array(flux)
            results["predictions"] = np.array(avg_pred)
        del time, flux, flux_error, preds, avg_pred, pred, t_pred, is_interesting
        return results

    except FileNotFoundError:
        print("File not found")
        return None


def find_models(path):
    """
    Returns the CNN model(s) as a list of paths.
    If path is a directory, it globs for .h5 files.
    If path is a file, it returns a list with that single file.
    """
    if os.path.isdir(path):
        return glob.glob(f"{path}/*.h5")
    elif os.path.isfile(path) and path.endswith('.h5'):
        return [path]
    else:
        return



def load_lightcurves(path):
    print("globbing files")
    return glob.glob(f"{path}/**/*.fits", recursive=True)


def process_lightcurve(path, pipeline):
    lc, info = import_lightcurve(path)


    time, flux, flux_error = (
        lc[pipeline["time"]],
        lc[pipeline["flux"]],
        lc[pipeline["flux_err"]],
    )
    time, flux, flux_error = scale_lightcurve(time, flux, flux_error)
    del lc
    return info[pipeline["id"]], time, flux, flux_error


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

def load_lightcurves_generator(path):
    for file in glob.glob(f"{path}/**/*.fits", recursive=True):
        yield file

def process_batch(batch, pipeline, models, threshold):
    results = []
    for lc in batch:
        result = process_single_lightcurve(lc, pipeline, models, threshold)
        if result is not None:
            results.append(result)



def process_batch(batch_args):
    batch, pipeline, models, threshold = batch_args
    results = []
    for lc in batch:
        result = process_single_lightcurve(lc, pipeline, models, threshold)
        if result is not None:
            results.append(result)
    return results

def main(ds):
    start_time = time.time()

    lightcurves = load_lightcurves_generator(args.path)
    pipeline = PIPELINE[args.p]
    print("Lightcurve data product: ", pipeline)

    models = find_models(args.model)

  
    total_results = 0
    with open(args.o, "ab") as output_file:
        with multiprocessing.Pool(processes=args.threads) as pool:
            # Create an iterator of arguments for each lightcurve
            lc_args = ((lc, pipeline, models, args.threshold) for lc in lightcurves)
            
            # Process lightcurves in parallel with a dynamic progress bar
            pbar = tqdm(desc="Processing lightcurves", unit=" lightcurves")
            for result in pool.imap_unordered(process_single_lightcurve, lc_args):
                if result is not None:
                    pickle.dump(result, output_file)
                    output_file.flush()
                    total_results += 1
                pbar.update(1)
            pbar.close()

    print(f"Total results processed: {total_results}")

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Script executed in {elapsed_time:.2f} minutes")

if __name__ == '__main__':
    with open("ds.pkl", "rb") as file:
        ds = pickle.load(file)
    main(ds)
    sys.exit(0)