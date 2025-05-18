import sys
import os
import pickle
import glob
import time
import argparse
import numpy as np
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from itertools import islice
import gc
import cProfile
import pstats
import io
from memory_profiler import profile
sys.path.insert(1, "scripts")
sys.path.insert(1, "stella")
import stella

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
        del dataset
        gc.collect()
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
    # Only load and keep what's absolutely necessary
    try:
        try:
            lc, info = import_lightcurve(path)
        except OSError:
            return None
            
        time = np.array(lc[pipeline["time"]])
        flux = np.array(lc[pipeline["flux"]])
        flux_error = np.array(lc[pipeline["flux_err"]])
        
        # Handle NaN values more efficiently
        mask = ~np.isnan(time) & ~np.isnan(flux) & ~np.isnan(flux_error)
        time = time[mask]
        flux = flux[mask]
        flux_error = flux_error[mask]
        
        # Keep original data only for interesting events
        original_time = time.copy()
        original_flux = flux / np.nanmedian(flux)
        
        # Scale the flux
        flux = flux / np.nanmedian(flux) - 1
        flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))
        
        return info[pipeline["id"]], time, flux, flux_error, original_flux, original_time

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


# def process_single_lightcurve(args):
#     global cnn
#     lc_path, pipeline, models, threshold = args
    
#     try:
#         result = process_lightcurve(lc_path, pipeline)
#         if result is None:
#             return None
            
#         source_id, time, flux, flux_error, original_flux, original_time = result
        
#         # Pre-allocate array for predictions
#         preds = np.zeros((len(models), len(time)))
        
#         for i, model in enumerate(models):
#             try:
#                 # Clear previous predictions
#                 if hasattr(cnn, 'predictions'):
#                     del cnn.predictions
                    
#                 cnn.predict(modelname=model, times=time, fluxes=flux, errs=flux_error)
#                 preds[i] = cnn.predictions[0]
                
#                 # Force garbage collection after each model prediction
#                 gc.collect()
#                 del cnn.predictions
#             except Exception as e:
#                 print(f"Error with model {model}: {e}")
#                 preds[i] = np.nan
        
#         # Find best prediction
#         avg_pred = np.nanmedian(preds, axis=0)
#         arg = np.argmax(avg_pred)
#         pred = avg_pred[arg]
#         t_pred = time[arg]  # Simplify this - no need to use cnn.predict_time
#         is_interesting = 1 if pred > threshold else 0
        
#         # Include all data for every lightcurve
#         results = {
#             "ID": source_id,
#             "t_pred": t_pred,
#             "pred": pred,
#             "is_interesting": is_interesting,
#             "original_time": original_time,
#             "original_flux": original_flux,
#             "time": time,
#             "flux": flux,
#             "predictions": avg_pred
#         }
        
#         # Explicit cleanup
#         del time, flux, flux_error, preds, avg_pred, result
#         gc.collect()
        
#         return results
#     except Exception as e:
#         print(f"Failed to process {lc_path}: {e}")
#         return None

def process_single_lightcurve(args):
    global cnn
    lc_path, pipeline, models, threshold = args
    
    try:
        result = process_lightcurve(lc_path, pipeline)
        if result is None:
            return None
            
        source_id, time, flux, flux_error, original_flux, original_time = result
        
        # Process one model at a time to reduce peak memory usage
        all_preds = []
        for i, model in enumerate(models):
            try:
                # Clear previous predictions
                if hasattr(cnn, 'predictions'):
                    del cnn.predictions
                    
                # Make prediction with current model
                cnn.predict(modelname=model, times=time, fluxes=flux, errs=flux_error)
                
                # Store this model's predictions
                all_preds.append(cnn.predictions[0])
                
                # Force garbage collection after each model
                gc.collect()
            except Exception as e:
                print(f"Error with model {model}: {e}")
                all_preds.append(np.full(len(time), np.nan))
        
        # Calculate median predictions across models
        # Using nanmedian to handle any NaN values from failed models
        avg_pred = np.nanmedian(all_preds, axis=0)
        
        # Find maximum prediction
        arg = np.argmax(avg_pred)
        pred = avg_pred[arg]
        t_pred = time[arg]
        is_interesting = 1 if pred > threshold else 0
        
        # Create result dictionary with full arrays
        # But convert to more memory-efficient data types where possible
        results = {
            "ID": source_id,
            "t_pred": t_pred,
            "pred": pred,
            "is_interesting": is_interesting,
            "time": time,
            "flux": flux,
            "predictions": avg_pred.astype(np.float32)  # Use float32 instead of float64
        }
        
        # Original flux is only needed if you're plotting later
        if is_interesting:
            results["original_time"] = original_time
            results["original_flux"] = original_flux
        
        # Explicit cleanup before returning
        del time, flux, flux_error, all_preds, avg_pred, result
        gc.collect()
        
        return results
    except Exception as e:
        print(f"Failed to process {lc_path}: {e}")
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


# 

def main():
    start_time = time.time()
    pipeline = PIPELINE[args.p]
    models = find_models(args.model)
    
    # Collect all lightcurve paths first
    print("Collecting lightcurve paths...")
    all_lightcurves = list(load_lightcurves_generator(args.path))
    total_files = len(all_lightcurves)
    print(f"Found {total_files} lightcurves to process")
    
    # Set the batch size - adjust based on your memory constraints
    batch_size = 100
    total_batches = (total_files + batch_size - 1) // batch_size  # Ceiling division
    
    # Process in batches
    total_results = 0
    
    with open(args.o, "ab") as output_file:
        for batch_idx in range(total_batches):
            # Calculate the start and end indices for this batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_files)
            
            print(f"Processing batch {batch_idx + 1}/{total_batches} (files {start_idx} to {end_idx-1})")
            
            # Extract the batch of lightcurves to process
            batch_lightcurves = all_lightcurves[start_idx:end_idx]
            
            # Create a fresh pool for this batch
            with multiprocessing.Pool(
                processes=args.threads,
                initializer=init_cnn,
                initargs=(args.ds,)
            ) as pool:
                # Create arguments for each lightcurve in the batch
                batch_args = [(lc, pipeline, models, args.threshold) 
                             for lc in batch_lightcurves]
                
                # Process the batch with a progress bar
                batch_tqdm = tqdm(
                    desc=f"Batch {batch_idx + 1}/{total_batches}",
                    total=len(batch_lightcurves),
                    unit=" lightcurves"
                )
                
                # Process each lightcurve in the batch
                for result in pool.imap_unordered(process_single_lightcurve, batch_args):
                    if result is not None:
                        pickle.dump(result, output_file)
                        output_file.flush()
                        total_results += 1
                    batch_tqdm.update(1)
                
                batch_tqdm.close()
            
            # Explicitly clear memory after each batch
            gc.collect()
            
            # Optional: Add a brief pause between batches to let system recover
            time.sleep(1)
    
    print(f"Total results processed: {total_results}")
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Script executed in {elapsed_time:.2f} minutes")

if __name__ == "__main__":
    with open(args.ds, "rb") as file:
        dataset = pickle.load(file)
        ds = dataset['dataset']


    main()

    sys.exit(0)