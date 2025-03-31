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
from astropy.table import Table, Column

sys.path.insert(1, 'scripts')
sys.path.insert(1, 'stella')

import stella
import optimise
os.nice(1)


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
    help="Number of epochs for CNN. Default 200.",
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
    "--optimise-bayes-name",
    help="The study name of the optuna optimisation saved as an sqlite database. Default is 'cnn_optimisation.db'.",
    default='cnn_optimisation.db',
    dest="optimise_bayes_name",
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

parser.add_argument(
    "-dsn",
    "--ds-name",
    help="Save the dataset as a pkl file.",
    default=None,
    dest="dsn",
)

parser.add_argument(
    "-l", "--layers",
    help="Path to a Python file defining model architecture",
    type=str,
    dest="layers",
    default=None
)

# Mutually exclusive group
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "-ld",
    "--load-data",
    help="Load a pre-defined dataset. Dataset must be in .pkl format. If no filename specified, uses ds.pkl",
    nargs='?', 
    const="ds.pkl", 
    dest="load_dataset"
)

args = parser.parse_args()

def plot_metrics(cnn, seed):
    """
    Plots the output metrics from the CNN model for a single seed.
    """
    # Create a custom colormap
    custom_cmap = mcolors.ListedColormap(['yellow', 'darkblue', 'red', 'cyan'])
    formatted_seed = f"{seed:04}"
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    
    # Top left: Validation predictions colored by original labels
    sc = axes[0, 0].scatter(
        cnn.val_pred_table['tpeak'], 
        cnn.val_pred_table[f'pred_s{formatted_seed}'],
        c=cnn.val_pred_table['labels'], 
        cmap=custom_cmap, 
        label=f'Validation Seed {formatted_seed}',
        s=5, alpha=0.8
    )
    axes[0, 0].set_xlabel('Tpeak [BJD - 2457000]')
    axes[0, 0].set_ylabel('Probability of Exocomet')
    axes[0, 0].set_title('Probabilities (with the original labels)')
    axes[0, 0].legend()
    plt.colorbar(sc, ax=axes[0, 0], ticks=np.arange(4), boundaries=np.arange(4+1)-0.5)
    
    # Top right: Binary classification (gt)
    sc2 = axes[0, 1].scatter(
        cnn.val_pred_table['tpeak'], 
        cnn.val_pred_table[f'pred_s{formatted_seed}'],
        c=cnn.val_pred_table['gt'], 
        label=f'Validation Seed {formatted_seed}',
        s=5
    )
    axes[0, 1].set_xlabel('Tpeak [BJD - 2457000]')
    axes[0, 1].set_ylabel('Probability of Exocomet')
    axes[0, 1].set_title('Binary Classification')
    axes[0, 1].legend()
    plt.colorbar(sc2, ax=axes[0, 1])
    
    # Bottom left: Accuracy curves
    axes[1, 0].plot(
        cnn.history_table[f'accuracy_s{formatted_seed}'], 
        label=f'Training Seed {formatted_seed}', 
        lw=3
    )
    axes[1, 0].plot(
        cnn.history_table[f'val_accuracy_s{formatted_seed}'], 
        label=f'Validation Seed {formatted_seed}', 
        lw=3
    )
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy')
    axes[1, 0].legend()
    
    # Bottom right: Loss curves
    axes[1, 1].plot(
        cnn.history_table[f'loss_s{formatted_seed}'], 
        label=f'Training Seed {formatted_seed}', 
        lw=3
    )
    axes[1, 1].plot(
        cnn.history_table[f'val_loss_s{formatted_seed}'], 
        label=f'Validation Seed {formatted_seed}', 
        lw=3
    )
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Loss')
    axes[1, 1].legend()

    plt.tight_layout()
    os.makedirs("plots-es/", exist_ok=True)
    plt.savefig(f"plots-es/cnn-metrics-s{seed}.png", dpi=300)
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
        with open(args.load_dataset, "rb") as file:
            ds = pickle.load(file)
            dataset = ds['dataset']
    
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
            other_datasets_labels=args.merge_labels, 
            cadences=args.c,
            training=args.training,
            validation=args.validation,
            frac_balance=args.frac_balance,  ### REMOVED ALL NEGATIVE CLASSES OF THE MERGING DATASETS
            augment_portion=args.flip_portion, 
        )
        
        if args.dsn:
            ds_info = {
                'dataset': dataset,
                'frac_balance': args.frac_balance,
                'training_fraction': args.training,
                'validation_fraction': args.validation,
                'positive_train': len(np.where(dataset.train_labels == 1)[0]),
                'negative_train': len(np.where(dataset.train_labels != 1)[0]),
                'positive_val': len(np.where(dataset.val_labels == 1)[0]),
                'negative_val': len(np.where(dataset.val_labels != 1)[0]),
                'total_train': len(dataset.train_labels),
                'total_val': len(dataset.val_labels),
                'cadences': args.c,
                'flip_portion': args.flip_portion,
                'batch_size': args.batch_size,
                'seeds': args.seed
            }

            with open(f'{args.dsn}', 'wb') as file:
                pickle.dump(ds_info, file)

        

    cnn_dir = os.path.join(os.getcwd(), 'cnn-models-es')

    if args.layers:
        import importlib.util
        spec = importlib.util.spec_from_file_location("layer", args.layers)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        layers = module.create_model_layers(input_shape=(args.c, 1))
        cnn = stella.ConvNN(
            output_dir=cnn_dir,
            ds=dataset,
            layers=layers)
        print(f"CNN initialised with custom layers from {args.layers}")
    else:
        cnn = stella.ConvNN(
            output_dir=cnn_dir,
            ds=dataset)

    print("CNN initialised.")
    print("Training sample %:", args.training)
    print("Validation sample %:", args.validation)

    positive_train = len(np.where(dataset.train_labels == 1)[0])
    positive_val = len(np.where(dataset.val_labels == 1)[0])
    print("Positive classes in training set:", positive_train)
    print("Positive classes in validation set:", positive_val)

    decision = input("Proceed? ")

    if (decision == "y") or (decision == "yes"):
    
        if args.optimise_bayes:
            print("Optimising hyperparameters with Optuna...")
            best_params = optimise.optimise_hyperparameters(cnn, n_trials=50, name=args.optimise_bayes_name)
            
            print("Training final model with best parameters...")
            for seed in args.seed:
                final_model, history = optimise.train_final_model(
                    cnn, 
                    best_params, 
                    epochs=args.e, 
                    seed=seed,
                )
                    
                # Create and populate val_pred_table
                val_preds = final_model.predict(cnn.ds.val_data)
                val_preds = np.reshape(val_preds, len(val_preds))
                
                # Create tables
                cnn.history_table = Table()
                cnn.val_pred_table = Table([
                    cnn.ds.val_ids,
                    cnn.ds.val_labels,
                    cnn.ds.val_tpeaks,
                    cnn.ds.val_labels_ori,
                ], names=["tic", "gt", "tpeak", "labels"])
                
                formatted_seed = f"{seed:04}"
                
                # Add predictions to validation table
                cnn.val_pred_table.add_column(Column(val_preds, name=f"pred_s{formatted_seed}"))
                
                # Add history metrics to history table
                for metric, values in history.history.items():
                    cnn.history_table.add_column(Column(values, name=f"{metric}_s{formatted_seed}"))
                
                # Save model 
                fmt_tail = f"_s{seed:04d}_i{args.e:04d}_b{cnn.frac_balance}"
                model_fmt = "ensemble" + fmt_tail + ".h5"
                
                final_model.save(os.path.join(cnn.output_dir, model_fmt), overwrite=True)
                
                # histories and predictions are saved for the final optimised model (it is optional for the non-optimised ones)
                fmt_table = f"_i{args.e:04d}_b{cnn.frac_balance}.txt"
                hist_fmt = f"ensemble_histories_opt_{int(seed):04d}" + fmt_table
                pred_fmt = f"ensemble_predval_opt_{int(seed):04d}" + fmt_table
                
                cnn.history_table.write(os.path.join(cnn.output_dir, hist_fmt), format="ascii",overwrite=True)
                cnn.val_pred_table.write(
                    os.path.join(cnn.output_dir, pred_fmt),
                    format="ascii",
                    fast_writer=False,
                    overwrite=True
                )
                
                cnn.model = final_model
                cnn.history = history
                
                print("CNN complete. Plotting metrics.")
                plot_metrics(cnn, seed)

        else:
            for seed in args.seed:
                cnn.train_models(
                    seeds=seed, epochs=args.e, batch_size=args.batch_size, shuffle=True, pred_test=True,save=True
                )

                print("CNN complete. Plotting metrics.")
                plot_metrics(cnn, seed)
