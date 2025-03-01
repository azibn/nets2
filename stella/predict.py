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
sys.path.insert(1, 'scripts')
sys.path.insert(1, 'stella')

import stella
import psutil

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

# @profile
def process_lightcurve(path, pipeline):
    # Only load and keep what's absolutely necessary
    try:
        try:
            lc, info = import_lightcurve(path)
        except OSError:
            return None
            
        # Extract only the columns we need to reduce memory usage
        time = np.array(lc[pipeline["time"]], dtype=np.float32)  # Use float32 instead of float64
        flux = np.array(lc[pipeline["flux"]], dtype=np.float32)
        flux_error = np.array(lc[pipeline["flux_err"]], dtype=np.float32)
        
        # Free memory from the original loaded data
        del lc
        
        # Handle NaN values more efficiently
        mask = ~np.isnan(time) & ~np.isnan(flux) & ~np.isnan(flux_error)
        time = time[mask]
        flux = flux[mask]
        flux_error = flux_error[mask]
        
        # Get source ID before clearing info dict
        source_id = info[pipeline["id"]]
        del info
        
        # Process flux in place to save memory
        flux_median = np.nanmedian(flux)
        original_flux = flux / flux_median
        
        # Scale the flux
        flux = (flux / flux_median) - 1
        flux_min, flux_max = np.min(flux), np.max(flux)
        flux = (flux - flux_min) / (flux_max - flux_min)
        
        return source_id, time, flux, flux_error, original_flux, time

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


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
        result = process_lightcurve(lc_path, pipeline)
        if result is None:
            return None
            
        source_id, time, flux, flux_error, original_flux, original_time = result
        
        # Use float32 to reduce memory usage
        preds = np.zeros((len(models), len(time)), dtype=np.float32)
        
        for i, model in enumerate(models):
            try:
                # Clear previous predictions
                if hasattr(cnn, 'predictions'):
                    del cnn.predictions
                    
                cnn.predict(modelname=model, times=time, fluxes=flux, errs=flux_error)
                preds[i] = cnn.predictions[0]
                
                # Force garbage collection after each model prediction
                gc.collect()
            except Exception as e:
                print(f"Error with model {model}: {e}")
                preds[i] = np.nan
        
        # Find best prediction
        avg_pred = np.nanmedian(preds, axis=0).astype(np.float32)
        arg = np.argmax(avg_pred)
        pred = float(avg_pred[arg])  # Convert to simple float to reduce memory
        t_pred = float(time[arg])    # Convert to simple float to reduce memory
        is_interesting = 1 if pred > threshold else 0
        
        # Include all data for every lightcurve
        results = {
            "ID": source_id,
            "t_pred": t_pred,
            "pred": pred,
            "is_interesting": is_interesting,
            "original_time": original_time,
            "original_flux": original_flux,
            "time": time,
            "flux": flux,
            "predictions": avg_pred
        }
        
        # Explicit cleanup
        del time, flux, flux_error, preds, avg_pred, result, original_time, original_flux
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

@profile
def main():
    start_time = time.time()
    
    pipeline = PIPELINE[args.p]
    models = find_models(args.model)
    
    # Reduce batch size to prevent excessive memory usage
    batch_size = 10000
    
    # Maximum number of tasks in the queue to prevent memory buildup
    max_tasks_per_child = 1000
    
    total_results = 0
    
    try:
        # Create pool with maxtasksperchild to prevent memory leaks
        pool = multiprocessing.Pool(
            processes=min(args.threads, multiprocessing.cpu_count()),
            initializer=init_cnn,
            initargs=(args.ds,),
            maxtasksperchild=max_tasks_per_child  # Recycle workers to prevent memory buildup
        )
        
        # Get all files to process
        print("Finding lightcurve files...")
        file_list = list(load_lightcurves_generator(args.path))
        total_files = len(file_list)
        print(f"Found {total_files} lightcurve files")
        
        # Calculate total number of batches
        num_batches = (total_files + batch_size - 1) // batch_size
        
        # Use a context manager for the output file
        with open(args.o, "ab") as f:
            # Process files in batches
            for batch_num in range(num_batches):
                # Calculate batch range
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, total_files)
                
                print(f"Processing batch {batch_num+1}/{num_batches} (files {start_idx} to {end_idx-1})")
                print(f"Current memory usage: {psutil.Process().memory_info().rss / (1024**3):.2f} GB")
                
                # Get files for this batch
                batch_files = file_list[start_idx:end_idx]
                
                # Prepare arguments for processing - generate on-the-fly with smaller chunks
                lc_args = ((lc_file, pipeline, models, args.threshold) for lc_file in batch_files)
                
                # Process batch with progress bar and chunksize to better control memory
                batch_results = 0
                chunksize = max(1, min(100, len(batch_files) // (pool._processes * 4)))
                
                with tqdm(total=len(batch_files), desc=f"Batch {batch_num+1}/{num_batches}") as pbar:
                    # Use imap instead of imap_unordered with specific chunksize to control memory better
                    for result in pool.imap(process_single_lightcurve, lc_args, chunksize=chunksize):
                        if result is not None:
                            pickle.dump(result, f)
                            # Only flush occasionally to reduce I/O overhead
                            if batch_results % 50 == 0:
                                f.flush()
                            batch_results += 1
                            total_results += 1
                        pbar.update(1)
                
                # Report memory usage after each batch
                memory_usage = psutil.Process().memory_info().rss / (1024**3)
                print(f"Batch {batch_num+1} complete: {batch_results}/{len(batch_files)} successful. Memory: {memory_usage:.2f} GB")
                
                # Force garbage collection between batches
                gc.collect()
                
                # Give the system a moment to clean up memory
                time.sleep(1)
        
        # Close the pool
        pool.close()
        pool.join()
        
    except KeyboardInterrupt:
        print("Script interrupted by user. Exiting...")
        if 'pool' in locals() and pool:
            pool.terminate()
            pool.join()
    except Exception as e:
        print(f"Error in main processing loop: {e}")
        if 'pool' in locals() and pool:
            pool.terminate()
            pool.join()
    
    print(f"Total results processed: {total_results}/{total_files}")
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

    pr.disable()

    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    sys.exit(0)