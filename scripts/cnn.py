"""
This module implements a Convolutional Neural Network (CNN) for exocomet detection.

It includes functions for initializing the CNN, training the model, and making predictions.
The CNN is built using the stella library and can be customized with various hyperparameters.
"""

import os
import sys
import pickle
import argparse
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.regularizers import l2
from scikeras.wrappers import KerasClassifier
#sys.path.insert(1, '/home/astro/phrdhx/nets2/stella')
current_dir = os.getcwd()
while os.path.basename(current_dir) != 'nets2':
    current_dir = os.path.dirname(current_dir)
    if current_dir == os.path.dirname(current_dir): 
        raise Exception("'nets2' directory not found in parent directories")
    
sys.path.insert(1, os.path.join(current_dir, 'scripts'))
sys.path.insert(1, os.path.join(current_dir, 'stella'))

import stella
import optimise
os.nice(7)


parser = argparse.ArgumentParser(
    description="Run a Convolutional Neural Network for exocomet detection. Enter your positional arguments as <path-to-directory> <path-to-catalog>."
)

# Positional arguments
parser.add_argument(help="Target directory", dest="path", nargs='?')
parser.add_argument(help="catalog", dest="catalog", nargs='?')

# Optional arguments
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
    default=[49],
    type=int,
    help="SEED(s) to use for CNN model. Default 4",
    dest="seed",
)
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
    "--optimise-bayes",
    help="Optimise the hyperparameters using Bayesian optimisation.",
    action="store_true",
    dest="optimise_bayes",
)

parser.add_argument(
    "--optimise-RS",
    help="Optimise the hyperparameters using RandomSearchCV.",
    action="store_true",
    dest="optimise_RS",
)


parser.add_argument("--merge", nargs="+", help="Paths to additional datasets to merge",dest='merge')
parser.add_argument(
    "--merge_catalogs",
    nargs="+",
    help="Paths to catalogs for additional datasets",
    dest="merge_catalogs",
)
parser.add_argument(
    "--merge_labels",
    nargs="+",
    help="Labels for additional datasets",
    dest="merge_labels",
    type=int,
)
parser.add_argument(
    "-fp",
    "--flip-portion",
    help="Flips a portion of the positive class data from left-right. Insert value as fraction. Default is None.",
    default=None,
    type=float,
    dest = "flip_portion"
)

# Mutually exclusive group
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "-ld",
    "--load-data",
    help="Load a pre-defined dataset. Dataset must be in .pkl format.",
    action="store_true",
    dest="load_dataset"
)

args = parser.parse_args()

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
    plt.savefig(f"plots/cnn-metrics-s{seed}.png", dpi=300)
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

def model_RS(filter1, filter2, dense, dropout, learning_rate, kernel1, kernel2, pool, l2val):
    model = keras.models.Sequential([
        tf.keras.layers.Conv1D(
            filters=filter1,
            kernel_size=kernel1,
            activation='elu',
            padding="same",
            input_shape=(args.c, 1),
            kernel_regularizer=l2(l2val)
        ),
        tf.keras.layers.MaxPooling1D(pool_size=pool),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Conv1D(
            filters=filter2,
            kernel_size=kernel2,
            activation='elu',
            padding="same",
            kernel_regularizer=l2(l2val)
        ),
        tf.keras.layers.MaxPooling1D(pool_size=pool),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense, activation='elu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

args = parser.parse_args()


if __name__ == "__main__":

    # datasets = [dataset]
    if args.load_dataset:
        with open("ds.pkl", "rb") as file:
            dataset = pickle.load(file)
    
    else:
        datasets = []
        if args.merge:
            for additional_dir, additional_catalog in zip(args.merge, args.merge_catalogs):
                additional_dataset = create_dataset(
                    additional_dir,
                    additional_catalog,
                    args.c,
                    training=1,
                    validation=0,
                    frac_balance=1,
                )
                datasets.append(additional_dataset)

        dataset = stella.FlareDataSet(
            args.path,
            catalog=args.catalog,
            merge_datasets=True,
            other_datasets=datasets,
            other_datasets_labels=args.merge_labels,  # change this so that this becomes a parser argumnet
            cadences=args.c,
            training=args.training,
            validation=args.validation,
            frac_balance=args.frac_balance,  ### REMOVED ALL NEGATIVE CLASSES OF THE MERGING DATASETS
            augment_portion=args.flip_portion,  # make this a parser argument (default value is OK)
        )

    cnn_dir = os.path.join(os.getcwd(), 'cnn-models')

    cnn = stella.ConvNN(
        output_dir=cnn_dir,
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
            if args.optimise_bayes:
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

            elif args.optimise_RS:
                    print("Using RandomSearchCV to optimise hyperparameters...")

                    model = KerasClassifier(build_fn=model_RS, epochs=150, batch_size=128, verbose=0)

                    param_dist = {
                    'kernel1': [7, 9, 11, 13, 15],
                    'kernel2': [3, 5, 7],
                    'pool': [2, 3, 4, 5],
                    'filter1': [16, 32, 64, 128, 256, 512],
                    'filter2': [64, 128, 256, 512, 1024],
                    'dense': [32, 64, 128, 256, 512],
                    'dropout': [0.2, 0.3, 0.4, 0.5],
                    'l2val': loguniform(0.001, 0.01)
                }

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, n_jobs=33)
                    random_search_result = random_search.fit(dataset.train_data, dataset.train_labels)
                    best_params = random_search_result.best_params_
                    print("Best parameters found: ", best_params)

            else:
                cnn.train_models(
                    seeds=seed, epochs=args.e, batch_size=args.batch_size, shuffle=True
                )

            print("CNN complete. Plotting metrics.")
            plot_metrics(cnn, seed)
