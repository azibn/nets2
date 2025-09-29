"""
Integration test for the complete local+global pipeline.
Tests all components working together without requiring real data.
"""

import os
import sys
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add paths
sys.path.insert(0, '..')
sys.path.insert(0, '../stella')

def create_synthetic_data(temp_dir, n_targets=5, n_cadences=6000):
    """Create synthetic lightcurve data for testing."""
    print("Creating synthetic test data...")
    
    # Create lightcurve directory
    lc_dir = os.path.join(temp_dir, 'lightcurves')
    os.makedirs(lc_dir, exist_ok=True)
    
    # Create catalog
    catalog_path = os.path.join(temp_dir, 'catalog.txt')
    
    catalog_data = []
    catalog_data.append("TIC tpeak")
    
    for i in range(n_targets):
        tic_id = 100000 + i
        
        # Create synthetic time series (TESS-like with orbital gaps)
        time = np.linspace(0, 30, n_cadences)  # 30 days
        
        # Add orbital gaps every ~13.7 days (TESS-like)
        gap_starts = [13.7, 27.4]
        for gap_start in gap_starts:
            gap_mask = (time >= gap_start) & (time <= gap_start + 0.7)
            time = time[~gap_mask]
        
        # Create flux with stellar variability + some transients
        flux = 1.0 + 0.01 * np.sin(2 * np.pi * time / 5.2)  # Rotation
        flux += 0.005 * np.random.normal(size=len(time))  # Noise
        
        # Add some synthetic exocomet-like events
        n_events = np.random.randint(2, 6)  # 2-5 events per target (ensure more samples)
        event_times = []
        
        for j in range(n_events):
            # Random event time
            event_time = np.random.uniform(2, 28)
            event_times.append(event_time)
            
            # Find closest time index
            closest_idx = np.argmin(np.abs(time - event_time))
            
            # Create asymmetric transit-like dip
            transit_duration = 0.3  # ~7 hours
            transit_depth = 0.002 + 0.003 * np.random.random()  # 0.2-0.5%
            
            for k in range(len(time)):
                dt = abs(time[k] - event_time)
                if dt < transit_duration:
                    # Asymmetric profile (exocomet-like)
                    if time[k] <= event_time:  # Ingress
                        flux[k] -= transit_depth * (1 - dt/transit_duration)**2
                    else:  # Egress (longer)
                        flux[k] -= transit_depth * (1 - dt/(transit_duration*1.5))**1.5
        
        # Create flux errors
        flux_err = 0.001 * np.ones_like(flux)
        
        # Save lightcurve
        lc_data = np.array([time, flux, flux_err], dtype=object)
        lc_path = os.path.join(lc_dir, f'{tic_id}_sector01.npy')
        np.save(lc_path, lc_data)
        
        # Add to catalog
        for event_time in event_times:
            catalog_data.append(f"{tic_id} {event_time:.6f}")
    
    # Write catalog
    with open(catalog_path, 'w') as f:
        f.write('\n'.join(catalog_data))
    
    print(f"✅ Created {n_targets} synthetic lightcurves and catalog")
    return lc_dir, catalog_path

def test_enhanced_dataset():
    """Test the enhanced FlareDataSet with global context."""
    print("\n" + "="*50)
    print("Testing Enhanced FlareDataSet")
    print("="*50)
    
    try:
        import stella
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create synthetic data
            lc_dir, catalog_path = create_synthetic_data(temp_dir)
            
            print("Testing standard FlareDataSet (existing functionality)...")
            standard_ds = stella.FlareDataSet(
                fn_dir=lc_dir,
                catalog=catalog_path,
                cadences=100,  # Small for testing
                training=0.7,
                validation=0.2,
                frac_balance=0.8,
                # Default: save_global_context=False
            )
            
            print(f"Standard dataset: {len(standard_ds.train_data)} train, "
                  f"{len(standard_ds.val_data)} val, {len(standard_ds.test_data)} test")
            
            print("Testing enhanced FlareDataSet (with global context)...")
            enhanced_ds = stella.FlareDataSet(
                fn_dir=lc_dir,
                catalog=catalog_path,
                cadences=100,
                training=0.7,
                validation=0.2, 
                frac_balance=0.8,
                # NEW: Enable global context
                save_global_context=True,
                global_window_size=500,
                global_window_days=2.0,
                orbit_gap_threshold=0.5
            )
            
            print(f"Enhanced dataset: {len(enhanced_ds.train_data)} train, "
                  f"{len(enhanced_ds.val_data)} val, {len(enhanced_ds.test_data)} test")
            print(f"Global orbital segments: {len(enhanced_ds.global_lightcurves)}")
            
            # Verify global context structure
            if len(enhanced_ds.global_lightcurves) > 0:
                sample_global = enhanced_ds.global_lightcurves[0]
                print(f"Sample global segment: TIC {sample_global['tic_id']}, "
                      f"{len(sample_global['time'])} cadences, "
                      f"{len(sample_global['tpeaks'])} transits")
            
            print("✅ Enhanced FlareDataSet test passed!")
            return enhanced_ds
            
    except Exception as e:
        print(f"❌ Enhanced FlareDataSet test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_data_generator(enhanced_ds):
    """Test the local+global data generator."""
    print("\n" + "="*50)
    print("Testing Data Generator")
    print("="*50)
    
    try:
        # Save dataset temporarily
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            import pickle
            pickle.dump({'dataset': enhanced_ds}, f)
            temp_dataset_path = f.name
        
        try:
            from data_generator import create_local_global_generators
            
            print("Creating data generators...")
            train_gen, val_gen, test_gen = create_local_global_generators(
                dataset_path=temp_dataset_path,
                local_window=100,
                global_window_size=200,
                global_window_days=2.0,
                batch_size=4
            )
            
            print(f"Generators created: {len(train_gen)} train, {len(val_gen)} val, {len(test_gen)} test batches")
            
            # Test data generation
            print("Testing batch generation...")
            X_batch, y_batch = train_gen[0]
            local_batch, global_batch = X_batch
            
            print(f"Batch shapes: local {local_batch.shape}, global {global_batch.shape}, labels {y_batch.shape}")
            print(f"Local data range: [{np.min(local_batch):.4f}, {np.max(local_batch):.4f}]")
            print(f"Global data range: [{np.min(global_batch):.4f}, {np.max(global_batch):.4f}]")
            print(f"Labels: {np.unique(y_batch, return_counts=True)}")
            
            print("✅ Data generator test passed!")
            return train_gen, val_gen, test_gen
            
        finally:
            # Clean up temp file
            os.unlink(temp_dataset_path)
            
    except Exception as e:
        print(f"❌ Data generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_global_rnn():
    """Test the global RNN architectures."""
    print("\n" + "="*50) 
    print("Testing Global RNN")
    print("="*50)
    
    try:
        import tensorflow as tf
        from global_rnn import create_global_rnn
        
        # Test input
        batch_size, timesteps, features = 4, 200, 1
        test_input = tf.random.normal((batch_size, timesteps, features))
        
        rnn_types = ['variability', 'residual', 'hierarchical']
        
        for rnn_type in rnn_types:
            print(f"Testing {rnn_type} RNN...")
            
            rnn = create_global_rnn(rnn_type, 
                                   hidden_units=32,  # Small for testing
                                   num_layers=2 if rnn_type != 'hierarchical' else None,
                                   base_units=16 if rnn_type == 'hierarchical' else None)
            
            output = rnn(test_input)
            print(f"  {rnn_type} RNN output shape: {output.shape}")
            print(f"  Parameters: {rnn.count_params():,}")
        
        print("✅ Global RNN test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Global RNN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fusion_model():
    """Test the fusion model architectures."""
    print("\n" + "="*50)
    print("Testing Fusion Model")  
    print("="*50)
    
    try:
        import tensorflow as tf
        from fusion_model import create_fusion_model
        
        # Test inputs
        batch_size = 4
        local_window = 100
        global_window = 200
        
        local_input = tf.random.normal((batch_size, local_window, 1))
        global_input = tf.random.normal((batch_size, global_window, 1))
        test_inputs = [local_input, global_input]
        
        fusion_types = ['standard', 'attention']
        
        for fusion_type in fusion_types:
            print(f"Testing {fusion_type} fusion model...")
            
            model = create_fusion_model(
                fusion_type,
                local_window=local_window,
                global_window=global_window,
                cnn_filters=[8, 16],  # Small for testing
                rnn_hidden=32,
                rnn_type='variability',
                fusion_units=[16, 8]
            )
            
            output = model(test_inputs)
            print(f"  {fusion_type} fusion output shape: {output.shape}")
            
            # Count parameters after the model has been called
            try:
                print(f"  Parameters: {model.count_params():,}")
            except ValueError as e:
                print(f"  Parameters: Could not count - {str(e)[:50]}...")
            
            # Test feature extraction
            if hasattr(model, 'get_feature_representations'):
                features = model.get_feature_representations(test_inputs)
                print(f"  Local features: {features['local_features'].shape}")
                print(f"  Global features: {features['global_features'].shape}")
        
        print("✅ Fusion model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Fusion model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_integration():
    """Test end-to-end training integration."""
    print("\n" + "="*50)
    print("Testing Training Integration")
    print("="*50)
    
    try:
        import tensorflow as tf
        
        # Create small synthetic dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data
            lc_dir, catalog_path = create_synthetic_data(temp_dir, n_targets=10, n_cadences=1000)
            
            # Create enhanced dataset  
            import stella
            enhanced_ds = stella.FlareDataSet(
                fn_dir=lc_dir,
                catalog=catalog_path,
                cadences=50,  # Very small for testing
                training=0.8,
                validation=0.1,
                frac_balance=0.9,
                save_global_context=True,
                global_window_size=100,
                global_window_days=1.0
            )
            
            # Save dataset
            import pickle
            dataset_path = os.path.join(temp_dir, 'test_dataset.pkl')
            with open(dataset_path, 'wb') as f:
                pickle.dump({'dataset': enhanced_ds}, f)
            
            # Create generators
            from data_generator import create_local_global_generators
            train_gen, val_gen, test_gen = create_local_global_generators(
                dataset_path=dataset_path,
                local_window=50,
                global_window_size=100,
                batch_size=2
            )
            
            # Create model
            from fusion_model import create_fusion_model
            model = create_fusion_model(
                'standard',
                local_window=50,
                global_window=100,
                cnn_filters=[4, 8],  # Very small for testing
                rnn_hidden=16,
                fusion_units=[8, 4]
            )
            
            # Compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.01),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Model parameter count - will be available after compile or first call
            try:
                print(f"Model created with {model.count_params():,} parameters")
            except ValueError:
                print("Model created (parameter count will be available after first batch)")
            print(f"Training on {len(train_gen)} batches, validating on {len(val_gen)} batches")
            
            # Quick training test (just 2 epochs)
            history = model.fit(
                train_gen,
                epochs=2,
                validation_data=val_gen,
                verbose=1
            )
            
            print(f"Training completed!")
            print(f"Final training loss: {history.history['loss'][-1]:.4f}")
            print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
            
            # Test prediction
            X_test, y_test = test_gen[0]
            predictions = model.predict(X_test)
            print(f"Test predictions: {predictions.flatten()}")
            
            print("✅ Training integration test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Training integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("🧪 Running Local+Global Pipeline Integration Tests")
    print("=" * 60)
    
    # Check dependencies
    missing_deps = []
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
    except ImportError:
        missing_deps.append("tensorflow")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import stella
        print("✅ stella module")
    except ImportError:
        missing_deps.append("stella (check sys.path)")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        return False
    
    # Run tests
    tests = [
        ("Enhanced Dataset", test_enhanced_dataset),
        ("Global RNN", test_global_rnn),  
        ("Fusion Model", test_fusion_model),
        ("Training Integration", test_training_integration)
    ]
    
    results = {}
    enhanced_ds = None
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running {test_name} Test")
        print(f"{'='*60}")
        
        if test_name == "Enhanced Dataset":
            enhanced_ds = test_func()
            results[test_name] = enhanced_ds is not None
        elif test_name == "Data Generator" and enhanced_ds is not None:
            gen_result = test_data_generator(enhanced_ds)
            results[test_name] = gen_result[0] is not None
        else:
            results[test_name] = test_func()
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed!")
        print("The local+global pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python example_usage.py' with your data")
        print("2. Use 'python train_fusion.py' for full training")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)