"""
This module implements a Convolutional Neural Network (CNN) for exocomet detection.

It includes functions for initializing the CNN, training the model, and making predictions.
The CNN is built using the stella library and can be customized with various hyperparameters.
"""

import os
import sys

sys.path.insert(1, "/Users/azib/Documents/open_source/nets2/stella/")
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import stella
import optimise  ## NEED TO FIND OUT HOW TO IMPORT THIS SCRIPT PROPERLY


parser = argparse.ArgumentParser(
    description="Run a Convolutional Neural Network for exocomet detection. Enter your positional arguments as <path-to-directory> <path-to-catalog>."
)
parser.add_argument(help="Target directory", dest="path")
parser.add_argument(help="catalog", dest="catalog")
parser.add_argument(
    "-c",
    "--cadences",
    help="Cadences to use for CNN window",
    type=int,
    dest="c",
    default=168,
)
parser.add_argument(
    "--training", type=float, default=0.8, help="Training fraction (default: 0.8)"
)
parser.add_argument(
    "--validation", type=float, default=0.1, help="Validation fraction (default: 0.1)"
)
parser.add_argument(
    "--frac-balance", type=float, default=0.73, help="Fraction balance (default: 0.73)"
)
parser.add_argument(
    "-s",
    "--seed",
    nargs="*",
    default=[42],
    type=int,
    help="SEED(s) to use for CNN model. Default 42",
    dest="seed",
)  # List to account for multiple SEEDs.
parser.add_argument(
    "-e",
    "--epochs",
    help="Number of epochs for CNN. Default 100.",
    default=200,
    type=int,
    dest="e",
)
parser.add_argument(
    "--batch-size",
    help="Batch size for CNN. Default 32.",
    default=32,
    type=int,
    dest="batch_size",
)

parser.add_argument(
    "--optimise",
    help="Optimise the hyperparameters.",
    action="store_true",
    dest="optimise",
)

parser.add_argument("--merge", nargs="+", help="Paths to additional datasets to merge")
parser.add_argument(
    "--merge_catalogs",
    nargs="+",
    help="Paths to catalogs for additional datasets",
    dest="merge_catalogs",
)


def plot_metrics(cnn, seed):
    """
    Plots the output metrics from the CNN model for a single seed.
    """
    # Create a custom colormap
    custom_cmap = mcolors.ListedColormap(["yellow", "darkblue", "red", "green"])

    _, axes = plt.subplots(2, 2, figsize=(18, 12))
    formatted_seed = f"{seed:04}"

    # Plot ground truth
    sc = axes[0, 0].scatter(
        cnn.val_pred_table["tpeak"],
        cnn.val_pred_table[f"pred_s{formatted_seed}"],
        c=cnn.val_pred_table["labels"],
        cmap=custom_cmap,
        label=f"Seed {formatted_seed}",
    )
    axes[0, 0].set_xlabel("Tpeak [BJD - 2457000]")
    axes[0, 0].set_ylabel("Probability of Exocomet")
    plt.colorbar(
        sc, ax=axes[0, 0], ticks=np.arange(4), boundaries=np.arange(4 + 1) - 0.5
    )

    # Plot loss
    axes[0, 1].plot(
        cnn.history_table[f"loss_s{formatted_seed}"],
        label=f"Training Seed {formatted_seed}",
        lw=3,
    )
    axes[0, 1].plot(
        cnn.history_table[f"val_loss_s{formatted_seed}"],
        label=f"Validation Seed {formatted_seed}",
        lw=3,
    )
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()

    # Plot accuracy
    axes[1, 0].plot(
        cnn.history_table[f"accuracy_s{formatted_seed}"],
        label=f"Training Seed {formatted_seed}",
        lw=3,
    )
    axes[1, 0].plot(
        cnn.history_table[f"val_accuracy_s{formatted_seed}"],
        label=f"Validation Seed {formatted_seed}",
        lw=3,
    )
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()

    # Plot ground truth (gt)
    sc2 = axes[1, 1].scatter(
        cnn.val_pred_table["tpeak"],
        cnn.val_pred_table[f"pred_s{formatted_seed}"],
        c=cnn.val_pred_table["gt"],
        cmap=custom_cmap,
        label=f"Seed {formatted_seed}",
    )
    axes[1, 1].set_xlabel("Tpeak [BJD - 2457000]")
    axes[1, 1].set_ylabel("Probability of Exocomet")
    plt.colorbar(
        sc2,
        ax=axes[1, 1],
        ticks=np.arange(4),
        boundaries=np.arange(4 + 1) - 0.5,
    )

    plt.tight_layout()
    os.makedirs("plots/", exist_ok=True)
    plt.savefig(f"plots/metrics-s{seed}-acc-opt.png", dpi=300)
    plt.close()


def create_dataset(path, catalog, cadences, training, validation, frac_balance):
    return stella.FlareDataSet(
        fn_dir=path,
        catalog=catalog,
        cadences=cadences,
        training=training,
        validation=validation,
        frac_balance=frac_balance,
    )


args = parser.parse_args()


if __name__ == "__main__":

    # datasets = [dataset]
    datasets = []
    if args.merge:
        for additional_dir, additional_catalog in zip(args.merge, args.merge_catalogs):
            additional_dataset = create_dataset(
                additional_dir,
                additional_catalog,
                args.c,
                args.training,
                args.validation,
                frac_balance=1,
            )
            datasets.append(additional_dataset)

    dataset = stella.FlareDataSet(
        args.path,
        catalog=args.catalog,
        merge_datasets=True,
        other_datasets=datasets,
        other_datasets_labels=[2, 3, 4],
        cadences=args.c,
        training=args.training,
        validation=args.validation,
        frac_balance=args.frac_balance,  ### REMOVE ALL NEGATIVE CLASSES OF THE MERGING DATASETS
    )

    cnn = stella.ConvNN(
        output_dir="/Users/azib/Documents/open_source/nets2/cnn-models/",
        ds=dataset,
    )  # ,layers=layers)

    print("CNN initialised.")
    print("Training sample %:", args.training)
    print("Validation sample %:", args.validation)

    positive_train = len(np.where(dataset.train_labels == 1)[0])
    positive_val = len(np.where(dataset.val_labels == 1)[0])
    print("Positive classes in training set:", positive_train)
    print("Positive classes in validation set:", positive_val)

    decision = input("Proceed? ")

    if (decision == "y") or (decision == "yes"):
        for seed in args.seed:
            if args.optimise:
                print("Selected optimising hyperparameters...")
                best_params = optimise.optimise_hyperparameters(
                    cnn, n_trials=100
                )  # You can adjust n_trials as needed
                optimise.apply_best_params(cnn, best_params, seed=seed)
                print("Optimisation complete. Best parameters:", best_params)

                print("CNN initialised.")
                cnn.train_models(
                    seeds=seed, epochs=args.e, batch_size=args.batch_size, shuffle=True
                )

            else:
                cnn.train_models(
                    seeds=seed, epochs=args.e, batch_size=args.batch_size, shuffle=True
                )

            print("CNN complete. Plotting metrics.")
            plot_metrics(cnn, seed)
