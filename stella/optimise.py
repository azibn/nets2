import tensorflow as tf
import optuna
import multiprocessing  

# def create_model_with_params(cnn_instance, params):

    
#     """Create model with specified parameters"""
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv1D(
#             filters=params['filter1'],
#             kernel_size=params['kernel1'],
#             activation="relu",
#             padding="same",
#             input_shape=(cnn_instance.cadences, 1),
#             kernel_regularizer=tf.keras.regularizers.l2(params['l2_lambda'])
#         ),
#         tf.keras.layers.MaxPooling1D(pool_size=2),
#         tf.keras.layers.Dropout(params['dropout']),
#         tf.keras.layers.Conv1D(
#             filters=params['filter2'],
#             kernel_size=params['kernel2'],
#             activation="relu",
#             padding="same",
#             kernel_regularizer=tf.keras.regularizers.l2(params['l2_lambda'])
#         ),
#         tf.keras.layers.MaxPooling1D(pool_size=2),
#         tf.keras.layers.Dropout(params['dropout']),

#         tf.keras.layers.Conv1D(
#             filters=params['filter3'],
#             kernel_size=params['kernel3'],
#             activation="relu",
#             padding="same",
#             kernel_regularizer=tf.keras.regularizers.l2(params['l2_lambda'])
#         ),
#         tf.keras.layers.MaxPooling1D(pool_size=2),
#         tf.keras.layers.Dropout(params['dropout']),



#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(32, activation="relu", 
#                             kernel_regularizer=tf.keras.regularizers.l2(params['l2_lambda'])),
#         tf.keras.layers.Dropout(params['dropout']),
#         tf.keras.layers.Dense(1, activation="sigmoid"),
#     ])

def create_model_with_params(cnn_instance, params):

    
    """Create model with specified parameters"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=7,
            activation="relu",
            padding="same",
            input_shape=(cnn_instance.cadences, 1),
            kernel_regularizer=tf.keras.regularizers.l2(params['l2_lambda'])
        ),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(params['dropout']),
        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(params['l2_lambda'])
        ),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(params['dropout']),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu", 
                            kernel_regularizer=tf.keras.regularizers.l2(params['l2_lambda'])),
        tf.keras.layers.Dropout(params['dropout']),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=params['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name='val_auc'),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.F1Score(threshold=0.5, average='micro'),
        ]
    )
    return model

def objective(trial, cnn_instance):

    #filter1 = trial.suggest_int("filter1", 16, 32, step=16)
    #filter2 = trial.suggest_int("filter2", 32, 128, step=16)
    #filter3 = trial.suggest_int("filter3", 64, 128, step=16)

    #kernel1 = trial.suggest_int("kernel1", 7,15, step=2)
    #kernel2 = trial.suggest_int("kernel2", 5,7, step=2)
    #kernel3 = trial.suggest_int("kernel3", 3,5, step=2)

    # Only tune regularization parameters
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    l2_lambda = trial.suggest_float("l2_lambda", 1e-6, 1e-2, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 128, 1024, step=256)

    params = {
        'dropout': dropout,
        'l2_lambda': l2_lambda,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        #'filter1': filter1,
        #'filter2': filter2,
        #'filter3': filter3,
        #'kernel1': kernel1,
        #'kernel2': kernel2,
        #'kernel3': kernel3
    }
    
    model = create_model_with_params(cnn_instance, params)

    train_labels = tf.cast(cnn_instance.ds.train_labels, tf.float32)
    val_labels = tf.cast(cnn_instance.ds.val_labels, tf.float32)

    # Train model
    history = model.fit(
        cnn_instance.ds.train_data,
        train_labels,
        epochs=200,
        batch_size=batch_size,
        validation_data=(cnn_instance.ds.val_data, val_labels),
        verbose=0,
    )

    return history.history["val_auc"][-1]

def optimise_hyperparameters(cnn_instance, n_trials=100,name='cnn_optimisation.db'):
    name = name
    storage = f"sqlite:///{name}" # must end with .db 
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{name}",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, cnn_instance), 
        n_trials=n_trials,
        n_jobs= int(multiprocessing.cpu_count()/2)
    )

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study.best_params

def train_final_model(cnn_instance, best_params, epochs, seed):
    """Train the final model using the best parameters"""
    tf.keras.backend.clear_session()  # Clear memory
    tf.random.set_seed(seed)
    
    model = create_model_with_params(cnn_instance, best_params)
    
    train_labels = tf.cast(cnn_instance.ds.train_labels, tf.float32)
    val_labels = tf.cast(cnn_instance.ds.val_labels, tf.float32)

    history = model.fit(
        cnn_instance.ds.train_data,
        train_labels,
        epochs=epochs,
        batch_size=best_params['batch_size'],
        validation_data=(cnn_instance.ds.val_data, val_labels),
        verbose=1
    )
    
    return model, history
