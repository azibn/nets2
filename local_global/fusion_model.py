"""
Fusion model combining local CNN and global RNN for exocomet detection.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import sys
import os

# Add paths to import existing CNN components
sys.path.insert(1, '../stella')

from global_rnn import create_global_rnn

class ExocometFusionModel(Model):
    """
    Combined model using local CNN for shape detection and global RNN for stellar context.
    """
    
    def __init__(self,
                 local_window: int = 168,
                 global_window: int = 500,  # After downsampling
                 cnn_filters: list = [16, 64],
                 cnn_kernels: list = [7, 3],
                 rnn_hidden: int = 128,
                 rnn_type: str = 'variability',
                 fusion_units: list = [64, 32],
                 dropout_rate: float = 0.3,
                 l2_reg: float = 0.01,
                 use_existing_cnn: bool = False,
                 existing_cnn_path: str = None):
        """
        Parameters
        ----------
        local_window : int
            Size of local window
        global_window : int
            Size of global window (after downsampling)
        cnn_filters : list
            Number of filters for each CNN layer
        cnn_kernels : list
            Kernel sizes for each CNN layer
        rnn_hidden : int
            Hidden units for RNN
        rnn_type : str
            Type of global RNN ('variability', 'residual', 'hierarchical')
        fusion_units : list
            Units for fusion MLP layers
        dropout_rate : float
            Dropout rate
        l2_reg : float
            L2 regularization factor
        use_existing_cnn : bool
            Whether to load existing CNN weights
        existing_cnn_path : str
            Path to existing CNN model
        """
        super(ExocometFusionModel, self).__init__(name='exocomet_fusion')
        
        self.local_window = local_window
        self.global_window = global_window
        self.rnn_type = rnn_type
        
        # Build local CNN branch
        if use_existing_cnn and existing_cnn_path and os.path.exists(existing_cnn_path):
            self.local_cnn = self._load_existing_cnn(existing_cnn_path)
        else:
            if use_existing_cnn:
                print(f"Warning: Could not load CNN from {existing_cnn_path}. Building new CNN.")
            self.local_cnn = self._build_local_cnn(
                cnn_filters, cnn_kernels, dropout_rate, l2_reg
            )
        
        # Build global RNN branch
        self.global_rnn = create_global_rnn(
            rnn_type=rnn_type,
            hidden_units=rnn_hidden,
            num_layers=2,
            dropout_rate=dropout_rate,
            use_attention=True,
            feature_dim=32,
            l2_reg=l2_reg
        )
        
        # Build fusion network
        self.fusion_network = self._build_fusion_network(
            fusion_units, dropout_rate, l2_reg
        )
    
    def _build_local_cnn(self, filters, kernels, dropout_rate, l2_reg):
        """Build local CNN for transit shape detection."""
        cnn_layers = []
        
        # Conv layers
        for i, (f, k) in enumerate(zip(filters, kernels)):
            cnn_layers.extend([
                layers.Conv1D(
                    filters=f,
                    kernel_size=k,
                    activation='relu',
                    padding='same',
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                    name=f'local_conv1d_{i}'
                ),
                layers.MaxPooling1D(pool_size=2, name=f'local_maxpool_{i}'),
                layers.Dropout(dropout_rate, name=f'local_dropout_{i}')
            ])
        
        # Feature extraction
        cnn_layers.extend([
            layers.GlobalAveragePooling1D(name='local_global_pool'),
            layers.Dense(32, activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                        name='local_features')
        ])
        
        return tf.keras.Sequential(cnn_layers, name='local_cnn')
    
    def _load_existing_cnn(self, model_path):
        """Load pre-trained CNN and extract feature layers."""
        try:
            print(f"Loading existing CNN from {model_path}")
            
            # Load the full model
            full_model = tf.keras.models.load_model(model_path)
            
            # Extract layers up to the last dense layer (before sigmoid)
            feature_layers = []
            
            # Copy all layers except the final classification layer
            for layer in full_model.layers[:-1]:
                if isinstance(layer, layers.Dense) and layer.activation.__name__ == 'sigmoid':
                    # Skip sigmoid output layers
                    continue
                feature_layers.append(layer)
            
            # Add a feature extraction layer if needed
            if len(feature_layers) == 0 or not isinstance(feature_layers[-1], layers.Dense):
                feature_layers.extend([
                    layers.GlobalAveragePooling1D(name='loaded_global_pool'),
                    layers.Dense(32, activation='relu', name='loaded_features')
                ])
            
            feature_extractor = tf.keras.Sequential(feature_layers, name='loaded_local_cnn')
            
            # Optionally freeze pre-trained layers (keep last 2 layers trainable)
            for layer in feature_extractor.layers[:-2]:
                layer.trainable = False
            
            print(f"Loaded CNN with {feature_extractor.count_params():,} parameters")
            return feature_extractor
            
        except Exception as e:
            print(f"Failed to load CNN from {model_path}: {e}")
            print("Building new CNN instead...")
            return None
    
    def _build_fusion_network(self, fusion_units, dropout_rate, l2_reg):
        """Build fusion MLP for combining local and global features."""
        fusion_layers = []
        
        # MLP layers
        for i, units in enumerate(fusion_units):
            fusion_layers.extend([
                layers.Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                    name=f'fusion_dense_{i}'
                ),
                layers.Dropout(dropout_rate, name=f'fusion_dropout_{i}')
            ])
        
        # Output layer
        fusion_layers.append(
            layers.Dense(1, activation='sigmoid', name='fusion_output')
        )
        
        return tf.keras.Sequential(fusion_layers, name='fusion_network')
    
    def call(self, inputs, training=False):
        """
        Forward pass through the fusion model.
        
        Parameters
        ----------
        inputs : list of tf.Tensor
            [local_input, global_input] where:
            - local_input: shape (batch, local_window, 1)
            - global_input: shape (batch, global_window, 1)
        training : bool
            Whether in training mode
            
        Returns
        -------
        output : tf.Tensor
            Probability predictions of shape (batch, 1)
        """
        local_input, global_input = inputs
        
        # Process local view (transit shape)
        local_features = self.local_cnn(local_input, training=training)
        
        # Process global view (stellar context)  
        global_features = self.global_rnn(global_input, training=training)
        
        # Concatenate features
        combined_features = tf.concat([local_features, global_features], axis=-1)
        
        # Fusion and classification
        output = self.fusion_network(combined_features, training=training)
        
        return output
    
    def get_feature_representations(self, inputs):
        """
        Get intermediate feature representations for analysis.
        
        Parameters
        ----------
        inputs : list of tf.Tensor
            [local_input, global_input]
            
        Returns
        -------
        features : dict
            Dictionary containing local_features, global_features, and combined_features
        """
        local_input, global_input = inputs
        
        local_features = self.local_cnn(local_input, training=False)
        global_features = self.global_rnn(global_input, training=False)
        combined_features = tf.concat([local_features, global_features], axis=-1)
        
        return {
            'local_features': local_features,
            'global_features': global_features,
            'combined_features': combined_features
        }
    
    def get_attention_weights(self, inputs):
        """
        Get attention weights from global RNN if available.
        
        Parameters
        ----------
        inputs : list of tf.Tensor
            [local_input, global_input]
            
        Returns
        -------
        attention_weights : tf.Tensor or None
            Attention weights if global RNN supports it
        """
        _, global_input = inputs
        
        if hasattr(self.global_rnn, 'get_attention_weights'):
            return self.global_rnn.get_attention_weights(global_input)
        else:
            return None


class AttentionFusionModel(ExocometFusionModel):
    """
    Alternative fusion model using cross-attention instead of simple concatenation.
    """
    
    def __init__(self, **kwargs):
        super(AttentionFusionModel, self).__init__(**kwargs)
        
        # Replace simple fusion with attention-based fusion
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=kwargs.get('dropout_rate', 0.3),
            name='cross_attention'
        )
        
        self.attention_norm = layers.LayerNormalization(name='attention_norm')
        
        # Learnable query for aggregating local+global info
        self.fusion_query = self.add_weight(
            shape=(1, 1, 64),
            initializer='random_normal',
            trainable=True,
            name='fusion_query'
        )
        
        # Final classification layer
        self.classifier = layers.Dense(1, activation='sigmoid', name='attention_output')
    
    def call(self, inputs, training=False):
        """
        Forward pass with attention-based fusion.
        
        Parameters
        ----------
        inputs : list of tf.Tensor
            [local_input, global_input]
        training : bool
            Training mode
            
        Returns
        -------
        output : tf.Tensor
            Classification probability
        """
        local_input, global_input = inputs
        
        # Process branches
        local_features = self.local_cnn(local_input, training=training)
        global_features = self.global_rnn(global_input, training=training)
        
        # Reshape features for attention (add sequence dimension)
        local_features = tf.expand_dims(local_features, axis=1)
        global_features = tf.expand_dims(global_features, axis=1) 
        
        # Stack features as key-value pairs
        all_features = tf.concat([local_features, global_features], axis=1)
        
        # Apply cross-attention with learnable query
        batch_size = tf.shape(local_input)[0]
        query = tf.tile(self.fusion_query, [batch_size, 1, 1])
        
        attended_features = self.cross_attention(
            query, all_features, training=training
        )
        
        # Residual connection and normalization
        attended_features = self.attention_norm(query + attended_features)
        
        # Classification
        attended_features = tf.squeeze(attended_features, axis=1)
        output = self.classifier(attended_features)
        
        return output
    
    def get_feature_representations(self, inputs):
        """
        Get intermediate feature representations for attention model.
        """
        local_input, global_input = inputs
        
        local_features = self.local_cnn(local_input, training=False)
        global_features = self.global_rnn(global_input, training=False)
        
        # For attention model, combined features are the attention output
        local_features_seq = tf.expand_dims(local_features, axis=1)
        global_features_seq = tf.expand_dims(global_features, axis=1)
        all_features = tf.concat([local_features_seq, global_features_seq], axis=1)
        
        batch_size = tf.shape(local_input)[0]
        query = tf.tile(self.fusion_query, [batch_size, 1, 1])
        combined_features = self.cross_attention(query, all_features, training=False)
        combined_features = tf.squeeze(combined_features, axis=1)
        
        return {
            'local_features': local_features,
            'global_features': global_features,
            'combined_features': combined_features
        }


def create_fusion_model(model_type='standard', **kwargs):
    """
    Factory function to create fusion models.
    
    Parameters
    ----------
    model_type : str
        Type of fusion model ('standard' or 'attention')
    **kwargs
        Additional arguments for the fusion model
        
    Returns
    -------
    model : tf.keras.Model
        The constructed fusion model
    """
    if model_type == 'standard':
        return ExocometFusionModel(**kwargs)
    elif model_type == 'attention':
        return AttentionFusionModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: 'standard', 'attention'")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Fusion Models...")
    
    # Test inputs
    batch_size = 4
    local_window = 168
    global_window = 500
    
    local_input = tf.random.normal((batch_size, local_window, 1))
    global_input = tf.random.normal((batch_size, global_window, 1))
    test_inputs = [local_input, global_input]
    
    print(f"Local input shape: {local_input.shape}")
    print(f"Global input shape: {global_input.shape}")
    
    # Test Standard Fusion Model
    print("\n1. Testing Standard Fusion Model:")
    std_model = create_fusion_model(
        'standard',
        local_window=local_window,
        global_window=global_window,
        cnn_filters=[16, 32],
        cnn_kernels=[7, 3],
        rnn_hidden=64,
        rnn_type='variability'
    )
    
    std_output = std_model(test_inputs)
    print(f"   Output shape: {std_output.shape}")
    print(f"   Parameters: {std_model.count_params():,}")
    
    # Test feature extraction
    features = std_model.get_feature_representations(test_inputs)
    print(f"   Local features shape: {features['local_features'].shape}")
    print(f"   Global features shape: {features['global_features'].shape}")
    print(f"   Combined features shape: {features['combined_features'].shape}")
    
    # Test Attention Fusion Model
    print("\n2. Testing Attention Fusion Model:")
    att_model = create_fusion_model(
        'attention',
        local_window=local_window,
        global_window=global_window,
        cnn_filters=[16, 32],
        rnn_hidden=64
    )
    
    att_output = att_model(test_inputs)
    print(f"   Output shape: {att_output.shape}")
    print(f"   Parameters: {att_model.count_params():,}")
    
    # Test with different RNN types
    print("\n3. Testing different RNN types:")
    rnn_types = ['variability', 'residual', 'hierarchical']
    
    for rnn_type in rnn_types:
        try:
            model = create_fusion_model(
                'standard',
                local_window=local_window,
                global_window=global_window,
                rnn_type=rnn_type,
                rnn_hidden=32
            )
            output = model(test_inputs)
            print(f"   {rnn_type} RNN: ✓ Output shape {output.shape}, "
                  f"Parameters: {model.count_params():,}")
        except Exception as e:
            print(f"   {rnn_type} RNN: ✗ Error: {e}")
    
    print("\nAll fusion model tests completed!")