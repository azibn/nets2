import tensorflow as tf
import optuna


def objective(trial, cnn_instance):
    # HYPERPARAMETERS TO TUNE
    filter1 = trial.suggest_int("filter1", 8, 64)
    filter2 = trial.suggest_int("filter2", 32, 128)
    dense = trial.suggest_int("dense", 16, 64)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 0.001,0.01, log=True)

    kernel_size1 = trial.suggest_int("kernel_size1", 3, 11, step=2)
    kernel_size2 = trial.suggest_int(
        "kernel_size2", 3, kernel_size1, step=2
    )  # MUST BE SMALLER THAN KERNEL_SIZE1
    pool_size1 = trial.suggest_int("pool_size1", 2, 4)
    pool_size2 = trial.suggest_int("pool_size2", 2, 4)

    # MODEL
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv1D(
                filters=filter1,
                kernel_size=7, #kernel_size1
                activation="leaky_relu",
                padding="same",
                input_shape=(cnn_instance.cadences, 1),
            ),
            tf.keras.layers.MaxPooling1D(pool_size=pool_size1),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Conv1D(
                filters=filter2,
                kernel_size=3, # kernel_size2
                activation="leaky_relu",
                padding="same",
            ),
            tf.keras.layers.MaxPooling1D(pool_size=pool_size2),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dense, activation="leaky_relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    # TRAIN MODEL
    history = model.fit(
        cnn_instance.ds.train_data,
        cnn_instance.ds.train_labels,
        epochs=20,  # Use fewer epochs for faster optimization
        batch_size=64,
        validation_data=(cnn_instance.ds.val_data, cnn_instance.ds.val_labels),
        verbose=0,
    )

    return history.history["val_accuracy"][-1]


def optimise_hyperparameters(cnn_instance, n_trials=10):
    storage = "sqlite:///optuna_study.db"

    # Create the study
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name="cnn_optimisation_v2",
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, cnn_instance), n_trials=n_trials,n_jobs=1)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study.best_params


def apply_best_params(cnn_instance, best_params, seed):
    """Updating the CNN with its optimal parameters."""
    cnn_instance.layers = [
        tf.keras.layers.Conv1D(
            filters=best_params["filter1"],
            kernel_size=best_params["kernel_size1"],
            activation="leaky_relu",
            padding="same",
            input_shape=(cnn_instance.cadences, 1),
        ),
        tf.keras.layers.MaxPooling1D(pool_size=best_params["pool_size1"]),
        tf.keras.layers.Dropout(best_params["dropout"]),
        tf.keras.layers.Conv1D(
            filters=best_params["filter2"],
            kernel_size=best_params["kernel_size2"],
            activation="leaky_relu",
            padding="same",
        ),
        tf.keras.layers.MaxPooling1D(pool_size=best_params["pool_size2"]),
        tf.keras.layers.Dropout(best_params["dropout"]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(best_params["dense"], activation="leaky_relu"),
        tf.keras.layers.Dropout(best_params["dropout"]),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
    cnn_instance.optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=best_params["learning_rate"]
    )
    cnn_instance.create_model(
        seed=seed
    )  