"""
Custom CNN architectures that can be read in to `cnn.py`. 
"""

import tensorflow as tf

def create_model_layers(input_shape=(168, 1)):

    filter1 = 64
    filter2 = 128
    filter3 = 256
    kernel_size1 = 7
    kernel_size2 = 7
    kernel_size3 = 7
    dilation1 = 1
    dilation2 = 2
    dilation3 = 4
    pool_size = 2
    dropout = 0.25
    l2val = 0.001
    activation = 'relu'
    
    
    layers = [
        ## LAYER 1
        tf.keras.layers.Conv1D(
            filters=filter1,
            kernel_size=kernel_size1,
            activation=activation,
            dilation_rate=dilation1,
            padding="same",
            input_shape=input_shape,
            kernel_regularizer=tf.keras.regularizers.l2(l2val)
        ),
        tf.keras.layers.MaxPooling1D(pool_size=pool_size),
        tf.keras.layers.Dropout(dropout),
        
        ## LAYER 2
        tf.keras.layers.Conv1D(
            filters=filter2,
            kernel_size=kernel_size2,
            activation=activation,
            dilation_rate=dilation2,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(l2val)
        ),
        tf.keras.layers.MaxPooling1D(pool_size=pool_size),
        tf.keras.layers.Dropout(dropout),
        
        ## LAYER 3
        tf.keras.layers.Conv1D(
            filters=filter3,
            kernel_size=kernel_size3,
            activation=activation,
            dilation_rate=dilation3,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(l2val)
        ),
        tf.keras.layers.MaxPooling1D(pool_size=pool_size),
        tf.keras.layers.Dropout(dropout),
        
        # Output layers
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(dense_units, activation=activation, 
        #                     kernel_regularizer=tf.keras.regularizers.l2(l2val)),
        # tf.keras.layers.Dropout(dropout),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ]
    
    return layers