import tensorflow as tf

def create_model_layers(input_shape=(168, 1)):
    """
    Defines a hybrid CNN-LSTM model architecture for exocomet detection.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of the input data (time steps, features)
    
    Returns:
    --------
    layers : list
        List of Keras layers composing the model
    """
    # Architecture parameters
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
    
    layers = [
        # Initial CNN layers for feature extraction
        tf.keras.layers.Conv1D(
            filters=filters1, 
            kernel_size=kernel_size1, 
            activation=activation, 
            padding="same",
            input_shape=input_shape,
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
    ]
    
    return layers
