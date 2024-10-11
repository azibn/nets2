import sys
import os
import pickle
import glob
import time
import argparse
import numpy as np
import stella
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from itertools import islice
import gc
import cProfile
import pstats
import io
from memory_profiler import profile
sys.path.insert(1, "../scripts")
sys.path.insert(1, "../stella")


from utils import *

os.nice(12)

# config = tf.compat.v1.ConfigProto(
#     intra_op_parallelism_threads=40,  # Parallelism within individual operations
#     inter_op_parallelism_threads=2    # Parallelism between independent operations
# )

# # Create a session with the above configuration
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)


parser = argparse.ArgumentParser(description="Predict CNN on lightcurve data")
parser.add_argument(help="Target directory of lightcurves", dest="path")
parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="Path to CNN model directory. Assumes models are in its own directory.",
    dest="model",
)
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

parser.add_argument(
    "-t",
    "--threads",
    type=int,
    help="Number of threads to use",
    default=20,
    dest="threads",
)

parser.add_argument(
    "-ds",
    "--dataset",
    help="The dataset used in training, in .pkl form.",
    default="dsv2.pkl",
    dest="ds",
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


def load_lightcurves_generator(path):
    for file in glob.glob(f"{path}/**/*.fits", recursive=True):
        yield file


def find_models(path):
    """
    Returns the CNN model(s) as a list of paths.
    If path is a directory, it globs for .h5 files.
    If path is a file, it returns a list with that single file.
    """
    if os.path.isdir(path):
        return glob.glob(f"{path}/*.h5")
    elif os.path.isfile(path) and path.endswith(".h5"):
        return [path]
    else:
        return


def process_lightcurve(path, pipeline):
    try:
        lc, info = import_lightcurve(path)
    except OSError: 
        return None

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


# @profile
def process_single_lightcurve(args):
    lc_path, pipeline, models, threshold = args
    try:
        source_id, time, flux, flux_error = process_lightcurve(lc_path, pipeline)
    except TypeError:
        return None
    
    try:
        cnn = stella.ConvNN(
            output_dir=f"{os.path.join(current_dir)}/cnn-models/", ds=ds
        )

        preds = np.zeros((len(models), len(time)))
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
            results["time"] = time
            results["flux"] = flux
            results["predictions"] = avg_pred

        del time, flux, flux_error, preds, avg_pred
        gc.collect()
        return results

    except FileNotFoundError:
        print("File not found")
        return None
    
    except OSError:
        return None


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


def main():
    start_time = time.time()

    lightcurves = load_lightcurves_generator(args.path)
    pipeline = PIPELINE[args.p]
    print("Lightcurve data product: ", pipeline)

    models = find_models(args.model)

    total_results = 0
    try:
        with open(args.o, "ab") as f:
            with multiprocessing.Pool(processes=args.threads) as pool:

                lc_args = ((lc, pipeline, models, args.threshold) for lc in lightcurves)

                tqdmbar = tqdm(desc="Processing lightcurves", unit=" lightcurves")
                for result in pool.imap_unordered(process_single_lightcurve, lc_args):
                    if result is None:
                        continue
                    pickle.dump(result, f)
                    f.flush()
                    total_results += 1
                    tqdmbar.update(1)
                tqdmbar.close()

                pool.close()
                pool.join()
    except KeyboardInterrupt:
        print("Script interrupted by user. Exiting...")
        pool.terminate()
        pool.join()

    print(f"Total results processed: {total_results}")
    pool.close()
    pool.join()
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Script executed in {elapsed_time:.2f} minutes")


if __name__ == "__main__":
    with open(args.ds, "rb") as file:
        ds = pickle.load(file)

    pr = cProfile.Profile()
    pr.enable()

    main()

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    sys.exit(0)