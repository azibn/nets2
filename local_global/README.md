# Local+Global Exocomet Detection Pipeline

This directory contains the implementation of a local+global RNN approach for exocomet detection, designed to work with your existing `nets2` framework.

## Overview

The local+global approach combines:
- **Local CNN**: Analyzes the detailed shape of potential exocomet transits (same as your existing approach)
- **Global RNN**: Learns stellar activity patterns from orbital-scale context to distinguish exocomets from stellar flares

## Key Features

- 🔄 **Backward Compatible**: Your existing CNN pipeline works unchanged
- 🛸 **TESS Orbit Aware**: Automatically splits lightcurves by orbital gaps
- 🧠 **Multiple RNN Types**: Variability, residual, and hierarchical architectures
- 📊 **Rich Evaluation**: Attention visualization, feature analysis, detailed metrics
- ⚡ **Memory Efficient**: On-demand global context matching, batch processing

## File Structure

```
local_global/
├── data_generator.py          # Local+global data generators
├── global_rnn.py             # RNN architectures for stellar context
├── fusion_model.py           # Combined CNN+RNN models
├── train_fusion.py           # Full training pipeline
├── example_usage.py          # Complete usage examples
└── README.md                 # This file
```

## Quick Start

### 1. Create Enhanced Dataset

First, create a `FlareDataSet` with global context enabled:

```python
import sys
sys.path.insert(0, '../stella')
import stella

# Create dataset with global context (NEW)
dataset = stella.FlareDataSet(
    fn_dir="path/to/lightcurves",
    catalog="path/to/catalog.txt",
    cadences=168,
    training=0.8,
    validation=0.1,
    frac_balance=0.73,
    # Enable global context preservation
    save_global_context=True,        
    global_window_size=2000,        # RNN input size
    global_window_days=3.0,         # Days around event
    orbit_gap_threshold=0.5         # TESS orbit splitting
)

# Save for later use
import pickle
with open('dataset_with_global.pkl', 'wb') as f:
    pickle.dump({'dataset': dataset}, f)
```

### 2. Train Local+Global Model

```python
from data_generator import create_local_global_generators
from fusion_model import create_fusion_model

# Create data generators
train_gen, val_gen, test_gen = create_local_global_generators(
    dataset_path="dataset_with_global.pkl",
    local_window=168,
    global_window_size=500,
    batch_size=32
)

# Create fusion model
model = create_fusion_model(
    model_type='standard',
    local_window=168,
    global_window=500,
    rnn_type='variability'
)

# Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=100)
```

### 3. Full Training Pipeline

For production training with callbacks, evaluation, and visualization:

```bash
python train_fusion.py dataset_with_global.pkl \
    --output-dir results \
    --epochs 200 \
    --batch-size 32 \
    --model-type standard \
    --rnn-type variability
```

## Architecture Details

### Enhanced FlareDataSet

Your existing `FlareDataSet` class now supports:

```python
FlareDataSet(
    # All your existing parameters work unchanged
    fn_dir="...", catalog="...", cadences=168, 
    training=0.8, validation=0.1, frac_balance=0.73,
    
    # NEW: Global context parameters (optional)
    save_global_context=False,      # Enable global preservation  
    global_window_size=2000,        # RNN sequence length
    global_window_days=3.0,         # Temporal context span
    orbit_gap_threshold=0.5,        # TESS orbit gap detection
    global_downsampling=4           # Downsample factor
)
```

When `save_global_context=False` (default), everything works exactly as before.

### Global RNN Architectures

#### 1. Variability RNN (Recommended)
```python
rnn = create_global_rnn('variability', 
                       hidden_units=128,
                       use_attention=True,
                       feature_dim=32)
```
- Learns stellar activity patterns
- Bidirectional LSTM with attention
- Outputs variability features for fusion

#### 2. Residual RNN
```python
rnn = create_global_rnn('residual', 
                       hidden_units=128)
```
- Encoder-decoder architecture
- Predicts expected stellar flux
- Detects anomalies via residuals

#### 3. Hierarchical RNN
```python
rnn = create_global_rnn('hierarchical',
                       scales=[1, 2, 4],
                       base_units=64)
```
- Multi-scale temporal processing
- Captures patterns at different timescales

### Fusion Models

#### Standard Fusion
```python
model = create_fusion_model('standard',
                           local_window=168,
                           global_window=500,
                           rnn_type='variability')
```
- Concatenates CNN and RNN features
- MLP classifier for final prediction

#### Attention Fusion
```python
model = create_fusion_model('attention',
                           local_window=168, 
                           global_window=500)
```
- Cross-attention between local and global features
- More sophisticated feature combination

## Usage Examples

### Basic Training
```python
# Run the complete example
python example_usage.py
```

### Advanced Training with Existing CNN
```python
python train_fusion.py dataset_with_global.pkl \
    --use-existing-cnn path/to/your/trained_cnn.h5 \
    --output-dir results \
    --epochs 200
```

### Hyperparameter Exploration
```python
python train_fusion.py dataset_with_global.pkl \
    --model-type attention \
    --rnn-type hierarchical \
    --cnn-filters 32 64 128 \
    --rnn-hidden 256 \
    --fusion-units 128 64 32 \
    --dropout 0.4 \
    --learning-rate 0.0005
```

## Integration with Existing Workflow

### Your Current CNN Workflow (Unchanged)
```python
# This still works exactly as before
dataset = stella.FlareDataSet(fn_dir="...", catalog="...")
cnn = stella.ConvNN(output_dir="cnn-models", ds=dataset)
cnn.train_models(seeds=[42], epochs=200)
```

### New Local+Global Workflow
```python
# Enhanced dataset (superset of original)
dataset = stella.FlareDataSet(..., save_global_context=True)

# Option 1: Use your existing CNN approach
cnn = stella.ConvNN(output_dir="cnn-models", ds=dataset)
cnn.train_models(seeds=[42], epochs=200)

# Option 2: Use new local+global approach  
train_gen, val_gen, test_gen = create_local_global_generators(...)
fusion_model = create_fusion_model(...)
fusion_model.fit(train_gen, validation_data=val_gen)
```

## Performance Analysis

The training pipeline provides comprehensive evaluation:

- **Metrics**: ROC-AUC, PR-AUC, precision, recall, F1-score
- **Visualizations**: ROC curves, precision-recall curves, training history
- **Feature Analysis**: Local vs global feature distributions, PCA projections
- **Attention Weights**: For interpretability (if using attention)

Results are saved in structured format:
```
output_dir/
├── best_model.h5                 # Best model
├── config.json                   # Training configuration  
├── training_history.png          # Training curves
├── evaluation_results.png        # Test set evaluation
├── evaluation_metrics.json       # Detailed metrics
├── test_predictions.txt          # Per-sample predictions
└── logs/                         # TensorBoard logs
```

## Computational Requirements

### Memory Usage
- **Standard Dataset**: Same as your existing approach
- **Enhanced Dataset**: Adds ~2-4x more storage for global contexts
- **Training**: Similar GPU memory requirements

### Training Time
- **Data Loading**: Slightly longer due to global context processing
- **Training**: ~2-3x longer per epoch due to RNN processing
- **Overall**: Comparable total training time with early stopping

### Recommended Resources
- **GPU**: 8GB+ VRAM recommended for batch_size=32
- **RAM**: 16GB+ for large datasets with global context
- **Storage**: ~2-5x dataset size for global context preservation

## Troubleshooting

### Common Issues

1. **"No global context found"** warnings
   - Check `orbit_gap_threshold` - may need adjustment for your data
   - Verify lightcurve files have sufficient length

2. **Memory errors during training**
   - Reduce `batch_size` 
   - Reduce `global_window_size`
   - Use gradient checkpointing: `tf.config.experimental.enable_tensor_float_32_execution(False)`

3. **Poor performance compared to CNN**
   - Try different `rnn_type` ('variability', 'residual', 'hierarchical')
   - Adjust `global_window_days` (try 2.0-5.0 days)
   - Use existing CNN weights: `--use-existing-cnn path/to/model.h5`

4. **Training very slow**
   - Reduce `global_window_size` (try 300-500)
   - Use fewer RNN layers or smaller `rnn_hidden`
   - Enable mixed precision: `tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})`

### Debugging Tips

```python
# Test data generator
train_gen, _, _ = create_local_global_generators("dataset.pkl", batch_size=4)
X, y = train_gen[0]
print(f"Local: {X[0].shape}, Global: {X[1].shape}, Labels: {y.shape}")

# Test model components individually  
from global_rnn import create_global_rnn
rnn = create_global_rnn('variability')
global_features = rnn(X[1])
print(f"Global features: {global_features.shape}")

# Visualize attention weights
if hasattr(model, 'get_attention_weights'):
    weights = model.get_attention_weights(X)
    print(f"Attention weights: {weights.shape if weights is not None else None}")
```

## Citation

If you use this local+global approach in your research, please cite your nets2 paper and mention the local+global extension:

```
This work uses the nets2 framework (Your Citation Here) extended with 
local+global RNN architecture for improved stellar activity discrimination.
```

## Contributing

To extend this framework:

1. **New RNN Architectures**: Add to `global_rnn.py`
2. **New Fusion Methods**: Add to `fusion_model.py`  
3. **New Data Augmentation**: Add to `data_generator.py`
4. **New Evaluation Metrics**: Add to `train_fusion.py`

## Support

For questions about this local+global extension:
1. Check this README and the example files
2. Run `python example_usage.py` for a complete walkthrough
3. Use `python train_fusion.py --help` for training options
4. Review the test outputs in each module (`python -m module_name`)