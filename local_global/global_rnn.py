"""
Global context RNN for characterizing stellar variability patterns.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import List

class VariabilityRNN(Model):
    """
    RNN model for learning stellar variability patterns from global context.
    Extracts features that help distinguish stellar activity from exocomet transits.
    """
    
    def __init__(self, 
                 hidden_units: int = 128,
                 num_layers: int = 2,
                 dropout_rate: float = 0.3,
                 use_attention: bool = True,
                 feature_dim: int = 32,
                 l2_reg: float = 0.01):
        """
        Parameters
        ----------
        hidden_units : int
            Number of hidden units in LSTM layers
        num_layers : int
            Number of stacked LSTM layers
        dropout_rate : float
            Dropout rate for regularization
        use_attention : bool
            Whether to use attention mechanism
        feature_dim : int
            Output feature dimension
        l2_reg : float
            L2 regularization factor
        """
        super(VariabilityRNN, self).__init__(name='variability_rnn')
        
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.feature_dim = feature_dim
        
        # Build LSTM layers with decreasing size
        self.lstm_layers = []
        for i in range(num_layers):
            units = hidden_units // (2 ** i)  # Decreasing units: 128 -> 64 -> 32
            units = max(units, 32)  # Minimum 32 units
            
            self.lstm_layers.append(
                layers.Bidirectional(
                    layers.LSTM(units, 
                               return_sequences=True,
                               dropout=dropout_rate,
                               recurrent_dropout=dropout_rate,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                               name=f'lstm_{i}'),
                    name=f'bidirectional_lstm_{i}'
                )
            )
        
        # Attention mechanism
        if use_attention:
            self.attention = layers.MultiHeadAttention(
                num_heads=4,
                key_dim=32,
                dropout=dropout_rate,
                name='multihead_attention'
            )
            self.attention_norm = layers.LayerNormalization(name='attention_norm')
        
        # Feature extraction layers
        self.global_pool = layers.GlobalAveragePooling1D(name='global_pool')
        self.feature_dense1 = layers.Dense(
            64, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='feature_dense1'
        )
        self.feature_dropout = layers.Dropout(dropout_rate, name='feature_dropout')
        
        # Output features
        self.output_features = layers.Dense(
            feature_dim, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg), 
            name='variability_features'
        )
    
    def call(self, inputs, training=False):
        """
        Process global context to extract variability features.
        
        Parameters
        ----------
        inputs : tf.Tensor
            Global context window of shape (batch, timesteps, 1)
        training : bool
            Whether in training mode
            
        Returns
        -------
        features : tf.Tensor
            Variability features of shape (batch, feature_dim)
        """
        x = inputs
        
        # Pass through LSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            x = lstm(x, training=training)
            # Add residual connection for deeper networks (if dimensions match)
            if i > 0 and hasattr(residual, 'shape') and x.shape[-1] == residual.shape[-1]:
                x = x + residual
            if i < len(self.lstm_layers) - 1:
                residual = x
        
        # Apply attention if specified
        if self.use_attention:
            # Self-attention
            attended = self.attention(x, x, training=training)
            x = self.attention_norm(x + attended)
        
        # Extract features
        pooled = self.global_pool(x)
        features = self.feature_dense1(pooled)
        features = self.feature_dropout(features, training=training)
        features = self.output_features(features)
        
        return features
    
    def get_attention_weights(self, inputs):
        """
        Extract attention weights for interpretation.
        
        Parameters
        ----------
        inputs : tf.Tensor
            Global context window
            
        Returns
        -------
        attention_weights : tf.Tensor or None
            Attention weights if attention is enabled
        """
        if not self.use_attention:
            return None
            
        x = inputs
        for lstm in self.lstm_layers:
            x = lstm(x, training=False)
        
        # Get attention weights
        attended, weights = self.attention(x, x, return_attention_scores=True)
        
        return weights


class ResidualVariabilityRNN(Model):
    """
    Alternative RNN architecture that predicts expected stellar flux 
    to compute residuals for anomaly detection.
    """
    
    def __init__(self, 
                 hidden_units: int = 128,
                 num_layers: int = 2,
                 dropout_rate: float = 0.3):
        """
        Parameters
        ----------
        hidden_units : int
            Number of hidden units in LSTM layers
        num_layers : int
            Number of stacked LSTM layers
        dropout_rate : float
            Dropout rate for regularization
        """
        super(ResidualVariabilityRNN, self).__init__(name='residual_variability_rnn')
        
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        # Encoder layers
        self.encoder_layers = []
        for i in range(num_layers):
            units = hidden_units // (2 ** i)
            self.encoder_layers.append(
                layers.LSTM(units, 
                           return_sequences=True,
                           dropout=dropout_rate,
                           name=f'encoder_lstm_{i}')
            )
        
        # Decoder layers (reverse order)
        self.decoder_layers = []
        for i in range(num_layers):
            units = hidden_units // (2 ** (num_layers - i - 1))
            self.decoder_layers.append(
                layers.LSTM(units, 
                           return_sequences=True,
                           dropout=dropout_rate,
                           name=f'decoder_lstm_{i}')
            )
        
        # Output layer to predict flux
        self.output_layer = layers.TimeDistributed(
            layers.Dense(1, activation='linear', name='flux_prediction'),
            name='time_distributed_output'
        )
    
    def call(self, inputs, training=False):
        """
        Predict expected flux pattern for residual computation.
        
        Parameters
        ----------
        inputs : tf.Tensor
            Global context window of shape (batch, timesteps, 1)
        training : bool
            Whether in training mode
            
        Returns
        -------
        expected_flux : tf.Tensor
            Predicted flux of shape (batch, timesteps, 1)
        """
        # Encode
        x = inputs
        for encoder in self.encoder_layers:
            x = encoder(x, training=training)
        
        # Decode
        for decoder in self.decoder_layers:
            x = decoder(x, training=training)
        
        # Predict flux
        expected_flux = self.output_layer(x)
        
        return expected_flux
    
    def compute_residuals(self, inputs):
        """
        Compute residuals between input and predicted flux.
        
        Parameters
        ----------
        inputs : tf.Tensor
            Global context window
            
        Returns
        -------
        residuals : tf.Tensor
            Residuals between input and prediction
        """
        predicted = self.call(inputs, training=False)
        residuals = inputs - predicted
        return residuals


class HierarchicalVariabilityRNN(Model):
    """
    Hierarchical RNN that processes global context at multiple timescales.
    """
    
    def __init__(self,
                 base_units: int = 64,
                 scales: List[int] = [1, 2, 4],  # Temporal scales
                 feature_dim: int = 32,
                 dropout_rate: float = 0.3):
        """
        Parameters
        ----------
        base_units : int
            Base number of units for RNNs
        scales : List[int] 
            Temporal downsampling scales to process
        feature_dim : int
            Output feature dimension
        dropout_rate : float
            Dropout rate
        """
        super(HierarchicalVariabilityRNN, self).__init__(name='hierarchical_variability_rnn')
        
        self.scales = scales
        self.rnns = {}
        
        # Create RNN for each scale
        for scale in scales:
            self.rnns[scale] = layers.Bidirectional(
                layers.LSTM(base_units, 
                           return_sequences=True,
                           dropout=dropout_rate,
                           name=f'lstm_scale_{scale}'),
                name=f'bidirectional_lstm_scale_{scale}'
            )
        
        # Feature combination
        self.feature_combine = layers.Dense(
            feature_dim * 2, 
            activation='relu',
            name='feature_combine'
        )
        self.feature_dropout = layers.Dropout(dropout_rate)
        self.output_features = layers.Dense(
            feature_dim,
            activation='relu', 
            name='hierarchical_features'
        )
    
    def call(self, inputs, training=False):
        """
        Process inputs at multiple temporal scales.
        
        Parameters
        ----------
        inputs : tf.Tensor
            Global context window
        training : bool
            Training mode
            
        Returns
        -------
        features : tf.Tensor
            Multi-scale features
        """
        scale_features = []
        
        for scale in self.scales:
            # Downsample input by scale
            if scale > 1:
                downsampled = inputs[:, ::scale, :]
            else:
                downsampled = inputs
                
            # Process with scale-specific RNN
            rnn_output = self.rnns[scale](downsampled, training=training)
            
            # Global pooling for this scale
            pooled = tf.reduce_mean(rnn_output, axis=1)
            scale_features.append(pooled)
        
        # Combine features from all scales
        combined = tf.concat(scale_features, axis=-1)
        features = self.feature_combine(combined)
        features = self.feature_dropout(features, training=training)
        features = self.output_features(features)
        
        return features


def create_global_rnn(rnn_type='variability', **kwargs):
    """
    Factory function to create global RNN model.
    
    Parameters
    ----------
    rnn_type : str
        Type of RNN ('variability', 'residual', or 'hierarchical')
    **kwargs
        Additional arguments for the specific RNN
        
    Returns
    -------
    model : tf.keras.Model
        The constructed RNN model
    """
    # Filter kwargs for each RNN type
    if rnn_type == 'variability':
        valid_params = {'hidden_units', 'num_layers', 'dropout_rate', 'use_attention', 'feature_dim', 'l2_reg'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params and v is not None}
        return VariabilityRNN(**filtered_kwargs)
    elif rnn_type == 'residual':
        valid_params = {'hidden_units', 'num_layers', 'dropout_rate'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params and v is not None}
        return ResidualVariabilityRNN(**filtered_kwargs)
    elif rnn_type == 'hierarchical':
        valid_params = {'base_units', 'scales', 'feature_dim', 'dropout_rate'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params and v is not None}
        return HierarchicalVariabilityRNN(**filtered_kwargs)
    else:
        raise ValueError(f"Unknown RNN type: {rnn_type}. "
                        f"Choose from: 'variability', 'residual', 'hierarchical'")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Global RNN architectures...")
    
    # Test input
    batch_size = 4
    timesteps = 500
    features = 1
    test_input = tf.random.normal((batch_size, timesteps, features))
    
    print(f"Test input shape: {test_input.shape}")
    
    # Test VariabilityRNN
    print("\n1. Testing VariabilityRNN:")
    var_rnn = create_global_rnn('variability', 
                               hidden_units=64,
                               num_layers=2,
                               use_attention=True,
                               feature_dim=32)
    
    var_output = var_rnn(test_input)
    print(f"   Output shape: {var_output.shape}")
    print(f"   Parameters: {var_rnn.count_params():,}")
    
    # Test ResidualVariabilityRNN  
    print("\n2. Testing ResidualVariabilityRNN:")
    res_rnn = create_global_rnn('residual',
                               hidden_units=64,
                               num_layers=2)
    
    res_output = res_rnn(test_input)
    residuals = res_rnn.compute_residuals(test_input)
    print(f"   Predicted flux shape: {res_output.shape}")
    print(f"   Residuals shape: {residuals.shape}")
    print(f"   Parameters: {res_rnn.count_params():,}")
    
    # Test HierarchicalVariabilityRNN
    print("\n3. Testing HierarchicalVariabilityRNN:")
    hier_rnn = create_global_rnn('hierarchical',
                                base_units=32,
                                scales=[1, 2, 4],
                                feature_dim=32)
    
    hier_output = hier_rnn(test_input)
    print(f"   Output shape: {hier_output.shape}")
    print(f"   Parameters: {hier_rnn.count_params():,}")
    
    # Test attention weights extraction
    if hasattr(var_rnn, 'get_attention_weights'):
        print("\n4. Testing attention weights extraction:")
        attention_weights = var_rnn.get_attention_weights(test_input)
        if attention_weights is not None:
            print(f"   Attention weights shape: {attention_weights.shape}")
        else:
            print("   No attention weights (attention disabled)")
    
    print("\nAll tests completed successfully!")