import os
import re
import sys
import cProfile
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial
import stella


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
    )

    cnn = stella.ConvNN(
        output_dir="/Users/azib/Documents/open_source/nets2/cnn-models/", ds=ds
    )
    return cnn


def process_file(
    file,
    cnn,
    modelname="/Users/azib/Documents/open_source/nets2/cnn-models/ensemble_s0002_i0200_b0.7.h5",
):
    try:
        time, flux, errs, _ = np.load(file, allow_pickle=True)
        cnn.predict(modelname=modelname, times=time, fluxes=flux, errs=errs)
        ### create function to save arrays
        time = cnn.predict_time[0]
        flux = cnn.predict_flux[0]
        errs = cnn.predict_err[0]
        predictions = cnn.predictions[0]

        np.save(file,np.array([time,flux,errs,predictions]))


        closest_index = np.argmin(np.abs(np.array(cnn.predict_time) - 1500))

        ## time boundary conditions
        start_index = max(closest_index - 4, 0)
        end_index = min(closest_index + 5, len(cnn.predictions[0]))

        return 1 if np.max(cnn.predictions[0][start_index:end_index]) > 0.9 else 0
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return 0


def process_folder(args):
    base_dir, skew, duration, cnn = args
    folder_pattern = f"injected-skew_{skew}-duration-{duration:.2f}-snr-5"
    files = glob(os.path.join(base_dir, folder_pattern, "*"))

    if not files:
        return 0

    labels = [process_file(file, cnn) for file in files]
    return np.mean(labels) if labels else 0


def main():
    base_dir = "exo9/"
    folders = [
        f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
    ]

    pattern = re.compile(
        r"injected-skew_(?P<skew>-?\d+(\.\d+)?)-duration-(?P<duration>\d+\.\d+)-snr-5"
    )

    skew_values = sorted(
        set(int(pattern.search(f).group("skew")) for f in folders if pattern.search(f))
    )
    duration_values = sorted(
        set(
            float(pattern.search(f).group("duration"))
            for f in folders
            if pattern.search(f)
        )
    )

    cnn = initialise_cnn()

    args_list = [
        (base_dir, skew, duration, cnn)
        for skew, duration in product(skew_values, duration_values)
    ]

    with Pool(processes=4) as pool:
        results = pool.map(process_folder, args_list)

    recovery_grid = np.array(results).reshape(len(duration_values), len(skew_values))

    plt.figure(figsize=(10, 8))
    plt.imshow(
        recovery_grid * 100,
        extent=[
            skew_values[0] - 0.5,
            skew_values[-1] + 0.5,
            duration_values[0],
            duration_values[-1],
        ],
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    plt.colorbar(label="Percentage of Recovered Injections")
    plt.xlabel("Skewness")
    plt.ylabel("Sigma")
    plt.title("2D Histogram of Recovered CNN Injections")
    plt.savefig("recovery-plot-time-and-ind-cond-all-wider-window.png", dpi=200)
    plt.show()
    plt.close()


if __name__ == "__main__":
    cProfile.run("main()", "output.prof")
    sys.exit()
