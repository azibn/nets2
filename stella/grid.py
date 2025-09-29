import argparse
import os
import sys
import itertools
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skewnorm
import modelmaker
sys.path.insert(1, "/Users/azib/Documents/open_source/nets2/scripts")
sys.path.insert(1, "/Users/azib/Documents/open_source/nets2/stella")
from utils import *

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


args = parser.parse_args()


def skewed_gaussian(x, alpha=1, t0=1500, sigma=1, A=0.001):
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


skew = np.arange(-8, 8, 1)
duration_range = np.arange(0.042, 5)  # make sure it goes up in the right increments.


def generate_models(n_models=args.n, save=args.s, save_path=args.sp):
    """Generates a set of skewed Gaussian models

    Params:
        n_models: Number of models you want to create. Default is 1000 per combination.
        save: Option to save the models in directory. Default is True.
        save_path: Path to directory you want to save the models. Default is '.'

    Returns:
        models: The models returned as a list.

    """
    models = []
    id = []
    for i, (skewness, duration, _) in tqdm.tqdm(
        enumerate(itertools.product(skew, duration_range, range(n_models)))
    ):
        model = skewed_gaussian(time, alpha=skewness, t0=1500, sigma=duration, A=0.001)
        models.append(model)

        if save:
            os.makedirs(save_path, exist_ok=True)
            path = os.path.join(save_path, f"exocomet_model_{i}.npy")
            np.save(path, model)

        id.append(i)

    catalogtime = np.array()
    df = create_catalog(
        id,
    )

    return models


def create_catalog(id, time, catalog_name="catalog.txt"):

    df = pd.DataFrame(data=[id, time], columns=["TIC", "tpeak"])
    df.to_csv(f"{catalog_name}", index=None)

    return df


if __name__ == "__main__":
    time = np.load("time-array-s7.npy")
    models = generate_models(args.n)

    # Generate the combinations
    combinations = list(itertools.product(skew, duration_range, range(args.n)))

    # Extract the skewness and duration values
    skew_values = [comb[0] for comb in combinations]
    duration_values = [comb[1] for comb in combinations]

    # Create a 2D histogram
    hist, xedges, yedges = np.histogram2d(
        skew_values, duration_values, bins=[len(skew), len(duration_range)]
    )

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(
        hist,
        interpolation="nearest",
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    plt.colorbar(label="Number of iterations")
    plt.xlabel("Skewness")
    plt.ylabel("Duration")
    plt.grid(True)
    plt.title("Number of iterations per skewness-duration combination")
    plt.show()


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
