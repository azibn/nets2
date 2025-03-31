"""
Custom CNN/RNN hybrid architecture that can be read in to `cnn.py`. 
"""

import tensorflow as tf

def create_rnn_model(cadences=168, learning_rate=0.001):
    """
    Creates a hybrid CNN-LSTM model for exocomet detection.
    
    Parameters:
    -----------
    cadences : int
        Number of time steps in each input sequence
    learning_rate : float
        Learning rate for the Adam optimizer
        
    Returns:
    --------
    model : keras.Model
        Compiled Keras model
    """
    # Architecture parameters
    # CNN parameters
    filters1 = 16
    filters2 = 64
    kernel_size1 = 7
    kernel_size2 = 3
    pool_size = 2
    
    # LSTM parameters
    lstm_units1 = 64
    lstm_units2 = 32
    
    # Regularization parameters
    dropout_cnn = 0.25
    dropout_lstm = 0.3
    l2_reg = 0.001
    
    # Activation function
    activation = 'relu'
    
    # Create model
    model = tf.keras.models.Sequential([
        # Initial CNN layers for feature extraction
        tf.keras.layers.Conv1D(
            filters=filters1, 
            kernel_size=kernel_size1, 
            activation=activation, 
            padding="same",
            input_shape=(cadences, 1),
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        ),
        tf.keras.layers.MaxPooling1D(pool_size=pool_size),
        tf.keras.layers.Dropout(dropout_cnn),
        
        # Optional second CNN layer
        tf.keras.layers.Conv1D(
            filters=filters2, 
            kernel_size=kernel_size2, 
            activation=activation, 
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        ),
        tf.keras.layers.MaxPooling1D(pool_size=pool_size),
        tf.keras.layers.Dropout(dropout_cnn),
        
        # Add RNN layers
        tf.keras.layers.LSTM(
            units=lstm_units1, 
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        ),
        tf.keras.layers.Dropout(dropout_lstm),
        
        tf.keras.layers.LSTM(
            units=lstm_units2, 
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        ),
        tf.keras.layers.Dropout(dropout_lstm),
        
        # Output layer
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.F1Score(
                threshold=0.5,
                average='micro', 
                dtype=tf.float32
            )
        ]
    )
    
    return model