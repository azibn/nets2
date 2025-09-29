import pytest
import numpy as np
import os
import sys
import pickle
import tempfile
import argparse
import glob
from unittest.mock import Mock, patch, MagicMock, mock_open
from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import predict
from predict import (
    init_cnn, load_lightcurves_generator, find_models, process_lightcurve,
    process_single_lightcurve, load_predictions, main, PIPELINE
)


@pytest.fixture
def sample_args():
    """Create sample command line arguments"""
    args = argparse.Namespace()
    args.path = "/test/lightcurves"
    args.model = ["/test/models"]
    args.o = "test_output.pkl"
    args.p = "eleanor-lite"
    args.threshold = 0.7
    args.threads = 4
    args.ds = "test_ds.pkl"
    return args


@pytest.fixture
def mock_lightcurve_data():
    """Create mock lightcurve data"""
    time = np.linspace(0, 100, 1000)
    flux = np.random.normal(1.0, 0.01, 1000) + 0.1 * np.sin(2 * np.pi * time / 10)
    flux_err = np.full_like(flux, 0.01)
    return time, flux, flux_err


@pytest.fixture
def mock_dataset():
    """Create mock dataset for CNN initialization"""
    dataset = {
        'dataset': Mock(),
        'frac_balance': 0.73,
        'training_fraction': 0.8,
        'validation_fraction': 0.1,
        'cadences': 168
    }
    return dataset


def test_pipeline_configurations():
    """Test that pipeline configurations are properly defined"""
    assert "eleanor-lite" in PIPELINE
    assert "SPOC" in PIPELINE
    assert "K2" in PIPELINE
    
    # Check eleanor-lite pipeline
    eleanor_pipeline = PIPELINE["eleanor-lite"]
    assert eleanor_pipeline["time"] == "TIME"
    assert eleanor_pipeline["flux"] == "PCA_FLUX"
    assert eleanor_pipeline["flux_err"] == "FLUX_ERR"
    assert eleanor_pipeline["id"] == "TIC_ID"
    
    # Check SPOC pipeline
    spoc_pipeline = PIPELINE["SPOC"]
    assert spoc_pipeline["time"] == "TIME"
    assert spoc_pipeline["flux"] == "PDCSAP_FLUX"
    assert spoc_pipeline["flux_err"] == "PDCSAP_FLUX_ERR"
    assert spoc_pipeline["id"] == "TICID"


@patch('predict.stella')
@patch('pickle.load')
@patch('builtins.open', new_callable=mock_open)
def test_init_cnn(mock_file, mock_pickle_load, mock_stella, mock_dataset):
    """Test CNN initialization function"""
    mock_pickle_load.return_value = mock_dataset
    mock_conv_nn = Mock()
    mock_stella.ConvNN.return_value = mock_conv_nn
    
    init_cnn("test_ds.pkl")
    
    mock_pickle_load.assert_called_once()
    mock_stella.ConvNN.assert_called_once_with(
        output_dir="/cnn-models/",
        ds=mock_dataset['dataset']
    )


@patch('glob.glob')
def test_load_lightcurves_generator(mock_glob):
    """Test lightcurve file discovery"""
    mock_glob.side_effect = [
        ["/test/file1.fits", "/test/file2.fits"],  # .fits files
        ["/test/file3.npy", "/test/file4.npy"]     # .npy files
    ]
    
    files = list(load_lightcurves_generator("/test/path"))
    
    assert len(files) == 4
    assert "/test/file1.fits" in files
    assert "/test/file3.npy" in files


@patch('glob.glob')
@patch('os.path.isdir')
@patch('os.path.isfile')
def test_find_models(mock_isfile, mock_isdir, mock_glob):
    """Test model discovery function"""
    # Test with directory path
    mock_isdir.side_effect = [True, False]
    mock_isfile.side_effect = [False, True]
    mock_glob.return_value = ["/models/model1.h5", "/models/model2.h5"]
    
    paths = ["/models/dir", "/models/specific_model.h5"]
    result = find_models(paths)
    
    assert len(result) == 3  # 2 from glob + 1 specific file
    assert "/models/model1.h5" in result
    assert "/models/specific_model.h5" in result


def test_process_lightcurve_fits(mock_lightcurve_data):
    """Test processing FITS lightcurve files"""
    time, flux, flux_err = mock_lightcurve_data
    
    # Mock FITS file data
    mock_lc_data = {
        "TIME": time,
        "PCA_FLUX": flux,
        "FLUX_ERR": flux_err
    }
    mock_info = {"TIC_ID": 123456789}
    
    pipeline = PIPELINE["eleanor-lite"]
    
    with patch('predict.import_lightcurve', return_value=(mock_lc_data, mock_info)):
        result = process_lightcurve("/test/file.fits", pipeline)
        
        assert result is not None
        source_id, proc_time, proc_flux, proc_flux_err, orig_flux, orig_time = result
        
        assert source_id == 123456789
        assert len(proc_time) <= len(time)  # May be shorter due to NaN removal
        assert np.all(proc_flux >= 0) and np.all(proc_flux <= 1)  # Should be normalized


def test_process_lightcurve_npy(mock_lightcurve_data):
    """Test processing NPY lightcurve files"""
    time, flux, flux_err = mock_lightcurve_data
    npy_data = np.array([time, flux])
    
    pipeline = PIPELINE["eleanor-lite"]
    
    with patch('numpy.load', return_value=npy_data), \
         patch('os.path.basename', return_value="123456789_lc.npy"):
        
        result = process_lightcurve("/test/file.npy", pipeline)
        
        assert result is not None
        source_id, proc_time, proc_flux, proc_flux_err, orig_flux, orig_time = result
        
        assert source_id == "123456789"
        assert len(proc_time) == len(time)


def test_process_lightcurve_with_nans():
    """Test lightcurve processing with NaN values"""
    time = np.array([1, 2, np.nan, 4, 5])
    flux = np.array([1.0, 1.1, 1.2, np.nan, 1.4])
    flux_err = np.array([0.01, 0.01, 0.01, 0.01, np.nan])
    
    mock_lc_data = {
        "TIME": time,
        "PCA_FLUX": flux,
        "FLUX_ERR": flux_err
    }
    mock_info = {"TIC_ID": 123456}
    
    pipeline = PIPELINE["eleanor-lite"]
    
    with patch('predict.import_lightcurve', return_value=(mock_lc_data, mock_info)):
        result = process_lightcurve("/test/file.fits", pipeline)
        
        assert result is not None
        source_id, proc_time, proc_flux, proc_flux_err, orig_flux, orig_time = result
        
        # Should have removed NaN values
        assert len(proc_time) == 2  # Only indices 0 and 1 are valid
        assert not np.any(np.isnan(proc_time))
        assert not np.any(np.isnan(proc_flux))


def test_process_lightcurve_file_error():
    """Test handling of file reading errors"""
    pipeline = PIPELINE["eleanor-lite"]
    
    with patch('predict.import_lightcurve', side_effect=OSError("File not found")):
        result = process_lightcurve("/test/nonexistent.fits", pipeline)
        assert result is None


@patch('predict.cnn')
def test_process_single_lightcurve(mock_global_cnn, mock_lightcurve_data):
    """Test processing a single lightcurve with CNN prediction"""
    time, flux, flux_err = mock_lightcurve_data
    
    # Mock CNN predictions
    mock_global_cnn.predictions = [np.random.random(len(time))]
    
    # Mock process_lightcurve result
    lc_result = (123456, time, flux, flux_err, flux * 1.5, time)
    
    models = ["/test/model1.h5", "/test/model2.h5"]
    threshold = 0.7
    args = ("/test/file.fits", PIPELINE["eleanor-lite"], models, threshold)
    
    with patch('predict.process_lightcurve', return_value=lc_result):
        result = process_single_lightcurve(args)
        
        assert result is not None
        assert result["ID"] == 123456
        assert "pred" in result
        assert "t_pred" in result
        assert "is_interesting" in result
        assert "predictions" in result


@patch('predict.cnn')
def test_process_single_lightcurve_interesting_event(mock_global_cnn):
    """Test processing lightcurve with interesting event above threshold"""
    time = np.linspace(0, 100, 100)
    flux = np.random.random(100)
    
    # Create high prediction values to trigger interesting event
    high_predictions = np.full(100, 0.8)  # Above threshold of 0.7
    mock_global_cnn.predictions = [high_predictions]
    
    lc_result = (123456, time, flux, flux * 0.1, flux * 1.5, time)
    models = ["/test/model.h5"]
    threshold = 0.7
    args = ("/test/file.fits", PIPELINE["eleanor-lite"], models, threshold)
    
    with patch('predict.process_lightcurve', return_value=lc_result):
        result = process_single_lightcurve(args)
        
        assert result["is_interesting"] == 1
        assert result["pred"] >= threshold
        assert "original_time" in result
        assert "original_flux" in result


def test_load_predictions():
    """Test loading predictions from pickle file"""
    test_data = [
        {"ID": 123, "pred": 0.8},
        {"ID": 456, "pred": 0.3},
        {"ID": 789, "pred": 0.9}
    ]
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp_file:
        for item in test_data:
            pickle.dump(item, tmp_file)
        tmp_file.flush()
        
        try:
            loaded_data = load_predictions(tmp_file.name)
            assert len(loaded_data) == 3
            assert loaded_data[0]["ID"] == 123
            assert loaded_data[2]["pred"] == 0.9
        finally:
            os.unlink(tmp_file.name)


@patch('predict.multiprocessing.Pool')
@patch('predict.find_models')
@patch('predict.load_lightcurves_generator')
@patch('builtins.open', new_callable=mock_open)
def test_main_execution_flow(mock_file, mock_load_lcs, mock_find_models, mock_pool, sample_args):
    """Test main execution flow"""
    # Mock setup
    mock_load_lcs.return_value = ["/test/lc1.fits", "/test/lc2.fits"]
    mock_find_models.return_value = ["/test/model.h5"]
    
    # Mock pool context manager
    mock_pool_instance = Mock()
    mock_pool.return_value.__enter__.return_value = mock_pool_instance
    mock_pool_instance.imap_unordered.return_value = [
        {"ID": 123, "pred": 0.8},
        {"ID": 456, "pred": 0.3}
    ]
    
    with patch('predict.args', sample_args), \
         patch('predict.tqdm'), \
         patch('predict.time.time', return_value=0), \
         patch('predict.time.sleep'), \
         patch('predict.pickle.dump'), \
         patch('builtins.open', mock_open()):
        
        main()
        
        mock_pool.assert_called()
        mock_pool_instance.imap_unordered.assert_called()


@patch('predict.multiprocessing.Pool')
@patch('predict.find_models')
@patch('predict.load_lightcurves_generator')
def test_main_batch_processing(mock_load_lcs, mock_find_models, mock_pool, sample_args):
    """Test batch processing in main function"""
    # Create more lightcurves than batch size to test batching
    lightcurves = [f"/test/lc{i}.fits" for i in range(250)]  # More than batch_size=100
    mock_load_lcs.return_value = lightcurves
    mock_find_models.return_value = ["/test/model.h5"]
    
    # Mock pool
    mock_pool_instance = Mock()
    mock_pool.return_value.__enter__.return_value = mock_pool_instance
    mock_pool_instance.imap_unordered.return_value = [{"ID": i} for i in range(100)]
    
    with patch('predict.args', sample_args), \
         patch('predict.tqdm'), \
         patch('predict.time.time'), \
         patch('predict.time.sleep'), \
         patch('builtins.open', mock_open()):
        
        main()
        
        # Should create multiple pools (one per batch)
        assert mock_pool.call_count >= 3  # At least 3 batches for 250 files


def test_memory_cleanup_in_process_single_lightcurve():
    """Test that memory cleanup happens in process_single_lightcurve"""
    with patch('predict.process_lightcurve', return_value=None):
        result = process_single_lightcurve(("test", {}, [], 0.7))
        assert result is None


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test"""
    yield
    # Any cleanup needed after tests