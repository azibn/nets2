"""
Custom CNN architectures that can be read in to `cnn.py`. 
"""

import tensorflow as tf

def create_model_layers(input_shape=(168, 1)):

    filter1 = 32
    filter2 = 64
    filter3 = 128
    kernel_size1 = 9
    kernel_size2 = 5
    kernel_size3 = 3
    pool_size = 2
    dropout = 0.4  
    l2val = 0.005  
    activation = 'relu'

    layers = [
        # First conv layer
        tf.keras.layers.Conv1D(
            filters=filter1,
            kernel_size=kernel_size1,
            activation=activation,
            padding="same",
            input_shape=input_shape,
            #dilation_rate=8,
            kernel_regularizer=tf.keras.regularizers.l2(l2val),
        ),
        tf.keras.layers.MaxPooling1D(pool_size=pool_size),
        tf.keras.layers.Dropout(dropout),
        
        # Second conv layer
        tf.keras.layers.Conv1D(
            filters=filter2,
            kernel_size=kernel_size2,
            activation=activation,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(l2val),

        ),
        tf.keras.layers.MaxPooling1D(pool_size=pool_size),
        tf.keras.layers.Dropout(dropout),
        
        # Third conv layer
        tf.keras.layers.Conv1D(
            filters=filter3,
            kernel_size=kernel_size3,
            activation=activation,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(l2val)
        ),
        tf.keras.layers.MaxPooling1D(pool_size=pool_size),
        tf.keras.layers.Dropout(dropout),
        
        # Dense layers


        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ]
    
    return layers