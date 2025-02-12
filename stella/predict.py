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

os.nice(4)

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
    nargs='+',
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
    default="ds.pkl",
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

def init_cnn(ds_path):
    """Initialise CNN once per worker process"""
    global cnn
    with open(ds_path, "rb") as file:
        dataset = pickle.load(file)
        ds = dataset['dataset']
    cnn = stella.ConvNN(output_dir=f"/cnn-models/", ds=ds)


def load_lightcurves_generator(path):
    for extension in ['.fits', '.npy']:
        pattern = f"{path}/**/*{extension}"
        for file in glob.glob(pattern, recursive=True):
            yield file


def find_models(path):
    """
    Returns the CNN model(s) as a list of paths.
    If path is a directory, it globs for .h5 files.
    If path is a file, it returns a list with that single file.
    """
    model_paths = []
    for p in path:
        if os.path.isdir(p):
            model_paths.extend(glob.glob(f"{p}/*.h5"))
        elif os.path.isfile(p) and p.endswith(".h5"):
            model_paths.append(p)
    return model_paths


def process_lightcurve(path, pipeline):
    """
    Import lightcurve and normalise the flux to be between 0 and 1.
    
    Params:
    -------
    path: str
        Path to the lightcurve file.
    pipeline: dict
        Dataset/mission pipeline configuration. Currently supports eleanor-lite and SPOC for TESS, and EVEREST for K2.

    Returns:
    --------
    ID: int
        ID of the target.
    time: np.array
        Time array.
    flux: np.array
        Scaled flux array.
    flux_error: np.array    
        Flux error array.
        
    """

    if path.endswith('.fits'):
        try:
            lc, info = import_lightcurve(path)
        except OSError:
            return None
        time, flux, flux_error = (
            lc[pipeline["time"]],
            lc[pipeline["flux"]], 
            lc[pipeline["flux_err"]]
        )
        time, flux, flux_error = scale_lightcurve(time, flux, flux_error)
    else:  
        try:
            data = np.load(path, allow_pickle=True)
            time, flux, flux_error = data[0], data[1], data[2]
            info = {'TIC_ID': int(path.split('/')[-1].split('_')[0])}
            return info['TIC_ID'], time, flux, flux_error
        except:
            return None
            
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

    global cnn

    lc_path, pipeline, models, threshold = args
    try:
        source_id, time, flux, flux_error = process_lightcurve(lc_path, pipeline)
    except TypeError:
        return None
    
    try:


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

        #if is_interesting:
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

    pipeline = PIPELINE[args.p]
    models = find_models(args.model)

    total_results = 0
    pool = None  # Initialize pool variable outside try block
    
    try:
        with open(args.o, "ab") as f:
            pool = multiprocessing.Pool(
                processes=args.threads,
                initializer=init_cnn,
                initargs=(args.ds,)
            )
            
            total_files = sum(1 for _ in load_lightcurves_generator(args.path))
            lc_args = ((lc, pipeline, models, args.threshold) 
                      for lc in load_lightcurves_generator(args.path))

            tqdmbar = tqdm(desc="Processing lightcurves", 
                         unit=" lightcurves",
                         total=total_files)
            
            for result in pool.imap_unordered(process_single_lightcurve, lc_args):
                if result is None:
                    continue
                pickle.dump(result, f)
                f.flush()
                total_results += 1
                tqdmbar.update(1)
            
            tqdmbar.close()

    except KeyboardInterrupt:
        print("Script interrupted by user. Exiting...")
        if pool:
            pool.terminate()
    finally:
        if pool:
            pool.close()
            pool.join()

    print(f"Total results processed: {total_results}")
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Script executed in {elapsed_time:.2f} minutes")

if __name__ == "__main__":
    with open(args.ds, "rb") as file:
        dataset = pickle.load(file)
        ds = dataset['dataset']

    pr = cProfile.Profile()
    pr.enable()

    main()

    sys.exit(0)