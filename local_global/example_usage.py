"""
Complete example of using the local+global exocomet detection pipeline.
Shows how to:
1. Create an enhanced FlareDataSet with global context
2. Train a local+global fusion model
3. Compare with standard CNN approach
"""

import os
import sys
import numpy as np
import pickle
from pathlib import Path

# Add paths for imports
sys.path.insert(0, '..')
sys.path.insert(0, '../stella')

import stella
from data_generator import create_local_global_generators
from fusion_model import create_fusion_model
from train_fusion import plot_training_history, evaluate_model

def create_enhanced_dataset_example():
    """
    Example of creating a FlareDataSet with global context enabled.
    """
    print("=" * 60)
    print("STEP 1: Creating Enhanced Dataset with Global Context")
    print("=" * 60)
    
    # Example paths - replace with your actual paths
    lightcurve_dir = "path/to/your/lightcurves"
    catalog_path = "path/to/your/catalog.txt"
    
    print(f"Note: This is an example. Please update paths:")
    print(f"- lightcurve_dir: {lightcurve_dir}")
    print(f"- catalog_path: {catalog_path}")
    
    # Check if example data exists (you would replace this with your actual check)
    if not os.path.exists(lightcurve_dir) or not os.path.exists(catalog_path):
        print("\n⚠️  Example data paths not found.")
        print("To run this example with your data:")
        print("1. Update lightcurve_dir and catalog_path above")
        print("2. Ensure your lightcurves are in .npy format with [time, flux, flux_err]")
        print("3. Ensure your catalog has columns 'TIC' and 'tpeak'")
        return None
    
    try:
        # Create regular dataset (your existing workflow)
        print("\nCreating standard FlareDataSet...")
        standard_dataset = stella.FlareDataSet(
            fn_dir=lightcurve_dir,
            catalog=catalog_path,
            cadences=168,
            training=0.8,
            validation=0.1,
            frac_balance=0.73,
            # Global context disabled (default)
        )
        
        print(f"Standard dataset created:")
        print(f"- Training samples: {len(standard_dataset.train_data)}")
        print(f"- Validation samples: {len(standard_dataset.val_data)}")
        print(f"- Test samples: {len(standard_dataset.test_data)}")
        
        # Create enhanced dataset with global context
        print("\nCreating enhanced FlareDataSet with global context...")
        enhanced_dataset = stella.FlareDataSet(
            fn_dir=lightcurve_dir,
            catalog=catalog_path,
            cadences=168,
            training=0.8,
            validation=0.1,
            frac_balance=0.73,
            # NEW: Enable global context preservation
            save_global_context=True,
            global_window_size=2000,
            global_window_days=3.0,
            orbit_gap_threshold=0.5,
            global_downsampling=4
        )
        
        print(f"Enhanced dataset created:")
        print(f"- Training samples: {len(enhanced_dataset.train_data)}")
        print(f"- Validation samples: {len(enhanced_dataset.val_data)}")
        print(f"- Test samples: {len(enhanced_dataset.test_data)}")
        print(f"- Global orbital segments: {len(enhanced_dataset.global_lightcurves)}")
        
        # Save datasets
        print("\nSaving datasets...")
        with open('standard_dataset.pkl', 'wb') as f:
            pickle.dump({'dataset': standard_dataset}, f)
        
        with open('enhanced_dataset_with_global.pkl', 'wb') as f:
            pickle.dump({'dataset': enhanced_dataset}, f)
        
        print("✅ Datasets saved:")
        print("- standard_dataset.pkl")
        print("- enhanced_dataset_with_global.pkl")
        
        return standard_dataset, enhanced_dataset
        
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        print("Please check your data paths and format.")
        return None

def train_standard_cnn_example(dataset_path="standard_dataset.pkl"):
    """
    Example of training standard CNN (your existing approach).
    """
    print("=" * 60)
    print("STEP 2: Training Standard CNN (Existing Approach)")
    print("=" * 60)
    
    try:
        # Load standard dataset
        with open(dataset_path, 'rb') as f:
            ds_info = pickle.load(f)
            dataset = ds_info['dataset']
        
        # Create CNN (your existing workflow)
        print("Creating standard CNN...")
        cnn = stella.ConvNN(
            output_dir="standard_cnn_models",
            ds=dataset
        )
        
        print("Training CNN for comparison...")
        # Train with single seed for quick comparison
        cnn.train_models(
            seeds=[42],
            epochs=50,  # Reduced for example
            batch_size=32,
            shuffle=True,
            pred_test=True,
            save=True
        )
        
        print("✅ Standard CNN training completed!")
        print("- Models saved in: standard_cnn_models/")
        
        return cnn
        
    except Exception as e:
        print(f"❌ Error training standard CNN: {e}")
        return None

def train_fusion_model_example(dataset_path="enhanced_dataset_with_global.pkl"):
    """
    Example of training the local+global fusion model.
    """
    print("=" * 60)
    print("STEP 3: Training Local+Global Fusion Model")
    print("=" * 60)
    
    try:
        # Create data generators
        print("Creating local+global data generators...")
        train_gen, val_gen, test_gen = create_local_global_generators(
            dataset_path=dataset_path,
            local_window=168,
            global_window_size=500,
            global_window_days=3.0,
            batch_size=16  # Smaller for example
        )
        
        # Test data generator
        print("Testing data generator...")
        X_batch, y_batch = train_gen[0]
        local_batch, global_batch = X_batch
        
        print(f"✅ Data generator working:")
        print(f"- Local batch shape: {local_batch.shape}")
        print(f"- Global batch shape: {global_batch.shape}")
        print(f"- Label batch shape: {y_batch.shape}")
        print(f"- Training batches: {len(train_gen)}")
        print(f"- Validation batches: {len(val_gen)}")
        
        # Create fusion model
        print("\nCreating fusion model...")
        fusion_model = create_fusion_model(
            model_type='standard',
            local_window=168,
            global_window=500,
            cnn_filters=[16, 32],  # Smaller for example
            cnn_kernels=[7, 3],
            rnn_hidden=64,  # Smaller for example
            rnn_type='variability',
            fusion_units=[32, 16],  # Smaller for example
            dropout_rate=0.3,
            l2_reg=0.01
        )
        
        # Compile model
        fusion_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ Fusion model created:")
        print(f"- Total parameters: {fusion_model.count_params():,}")
        
        # Quick training run (reduced epochs for example)
        print("\nStarting training (reduced epochs for example)...")
        history = fusion_model.fit(
            train_gen,
            epochs=10,  # Very short for example
            validation_data=val_gen,
            verbose=1
        )
        
        # Save model
        os.makedirs("fusion_models", exist_ok=True)
        fusion_model.save("fusion_models/example_fusion_model.h5")
        
        print("✅ Fusion model training completed!")
        print("- Model saved: fusion_models/example_fusion_model.h5")
        
        return fusion_model, history
        
    except Exception as e:
        print(f"❌ Error training fusion model: {e}")
        return None, None

def compare_models_example():
    """
    Example of comparing standard CNN vs fusion model performance.
    """
    print("=" * 60)
    print("STEP 4: Comparing Model Performance")
    print("=" * 60)
    
    try:
        # Load test data generators
        _, _, test_gen = create_local_global_generators(
            dataset_path="enhanced_dataset_with_global.pkl",
            batch_size=32
        )
        
        # Load fusion model
        fusion_model = tf.keras.models.load_model("fusion_models/example_fusion_model.h5")
        
        # Get some predictions for comparison
        X_test, y_test = test_gen[0]
        
        # Fusion model predictions
        fusion_preds = fusion_model.predict(X_test)
        
        print("✅ Model comparison:")
        print(f"- Test batch size: {len(y_test)}")
        print(f"- True positive samples: {np.sum(y_test)}")
        print(f"- Fusion model avg prediction: {np.mean(fusion_preds):.4f}")
        print(f"- Fusion model prediction range: [{np.min(fusion_preds):.4f}, {np.max(fusion_preds):.4f}]")
        
        # You could add more detailed comparison metrics here
        
    except Exception as e:
        print(f"❌ Error comparing models: {e}")

def feature_analysis_example():
    """
    Example of analyzing what the fusion model learns.
    """
    print("=" * 60)
    print("STEP 5: Feature Analysis")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        
        # Load model and data
        fusion_model = tf.keras.models.load_model("fusion_models/example_fusion_model.h5")
        _, _, test_gen = create_local_global_generators(
            dataset_path="enhanced_dataset_with_global.pkl",
            batch_size=10
        )
        
        # Get a batch for analysis
        X_test, y_test = test_gen[0]
        
        # Extract feature representations
        features = fusion_model.get_feature_representations(X_test)
        
        print("✅ Feature analysis:")
        print(f"- Local features shape: {features['local_features'].shape}")
        print(f"- Global features shape: {features['global_features'].shape}")
        print(f"- Combined features shape: {features['combined_features'].shape}")
        
        # Simple visualization of feature distributions
        local_feats = features['local_features'].numpy()
        global_feats = features['global_features'].numpy()
        
        print(f"- Local features mean: {np.mean(local_feats, axis=0)[:5]}...")  # First 5
        print(f"- Global features mean: {np.mean(global_feats, axis=0)[:5]}...")  # First 5
        
        # You could add more sophisticated analysis here
        # - t-SNE visualization
        # - Feature importance analysis  
        # - Attention weight visualization
        
    except Exception as e:
        print(f"❌ Error in feature analysis: {e}")

def full_training_example():
    """
    Example of running a full training job using the train_fusion.py script.
    """
    print("=" * 60)
    print("STEP 6: Full Training Example")
    print("=" * 60)
    
    print("To run a full training job, use the train_fusion.py script:")
    print()
    print("Example command:")
    print("python train_fusion.py enhanced_dataset_with_global.pkl \\")
    print("    --output-dir fusion_training_results \\")
    print("    --epochs 200 \\")
    print("    --batch-size 32 \\")
    print("    --model-type standard \\")
    print("    --rnn-type variability \\")
    print("    --cnn-filters 16 64 \\")
    print("    --rnn-hidden 128 \\")
    print("    --learning-rate 0.001 \\")
    print("    --patience 30")
    print()
    print("This will:")
    print("- Train the model with proper callbacks")
    print("- Save the best model")
    print("- Generate training plots")
    print("- Evaluate on test set")
    print("- Save detailed results and metrics")

def main():
    """
    Run the complete example pipeline.
    """
    print("🚀 Local+Global Exocomet Detection Pipeline Example")
    print("This example demonstrates the complete workflow.")
    print()
    
    # Step 1: Create datasets (comment out if already created)
    datasets = create_enhanced_dataset_example()
    if datasets is None:
        print("\n⚠️  Skipping remaining steps due to dataset creation issues.")
        print("Please update the paths in create_enhanced_dataset_example() with your data.")
        return
    
    # Step 2: Train standard CNN for comparison
    # cnn = train_standard_cnn_example()
    
    # Step 3: Train fusion model
    # fusion_model, history = train_fusion_model_example()
    
    # Step 4: Compare models
    # compare_models_example()
    
    # Step 5: Feature analysis
    # feature_analysis_example()
    
    # Step 6: Full training example
    full_training_example()
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED")
    print("=" * 60)
    print("Next steps:")
    print("1. Update data paths in this script")
    print("2. Run the full example with your data")
    print("3. Use train_fusion.py for production training")
    print("4. Compare results with your existing CNN approach")
    print("5. Analyze feature representations and attention weights")

if __name__ == "__main__":
    # Ensure we have required imports
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not found. Please install: pip install tensorflow")
        sys.exit(1)
    
    try:
        from astropy.table import Table
        print(f"Astropy available ✅")
    except ImportError:
        print("❌ Astropy not found. Please install: pip install astropy")
        sys.exit(1)
    
    try:
        import matplotlib.pyplot as plt
        print(f"Matplotlib available ✅")
    except ImportError:
        print("❌ Matplotlib not found. Please install: pip install matplotlib")
        sys.exit(1)
    
    print()
    main()