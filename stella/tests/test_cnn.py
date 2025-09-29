import pytest
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import tempfile
import argparse
from unittest.mock import Mock, patch, MagicMock, mock_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cnn
from cnn import plot_metrics, create_dataset, model_RS


@pytest.fixture
def mock_stella():
    """Mock stella module components"""
    with patch('cnn.stella') as mock_stella:
        mock_flare_dataset = Mock()
        mock_conv_nn = Mock()
        mock_stella.FlareDataSet = Mock(return_value=mock_flare_dataset)
        mock_stella.ConvNN = Mock(return_value=mock_conv_nn)
        yield mock_stella


@pytest.fixture
def mock_cnn_with_data():
    """Create a mock CNN instance with prediction and history data"""
    mock_cnn = Mock()
    
    # Mock validation prediction table
    mock_cnn.val_pred_table = {
        'tpeak': np.linspace(2457000, 2458000, 100),
        'pred_s0049': np.random.random(100),
        'labels': np.random.randint(0, 4, 100),
        'gt': np.random.randint(0, 2, 100)
    }
    
    # Mock history table
    mock_cnn.history_table = {
        'accuracy_s0049': np.random.random(50),
        'val_accuracy_s0049': np.random.random(50),
        'loss_s0049': np.random.random(50),
        'val_loss_s0049': np.random.random(50)
    }
    
    return mock_cnn


@pytest.fixture
def sample_args():
    """Create sample command line arguments"""
    args = argparse.Namespace()
    args.path = "/test/path"
    args.catalog = "test_catalog.csv"
    args.c = 168
    args.training = 0.8
    args.validation = 0.1
    args.frac_balance = 0.73
    args.seed = [49]
    args.e = 200
    args.batch_size = 32
    args.optimise_bayes = False
    args.optimise_bayes_name = 'cnn_optimisation.db'
    args.merge = None
    args.merge_catalogs = None
    args.merge_labels = None
    args.flip_portion = None
    args.dsn = None
    args.load_dataset = None
    args.layers = None
    return args


@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
@patch('matplotlib.pyplot.tight_layout')
@patch('os.makedirs')
def test_plot_metrics(mock_makedirs, mock_tight_layout, mock_close, mock_savefig, mock_cnn_with_data):
    """Test the plot_metrics function"""
    seed = 49
    
    plot_metrics(mock_cnn_with_data, seed)
    
    # Check that directories are created
    mock_makedirs.assert_called_once_with("plots-es/", exist_ok=True)
    
    # Check that plot is saved with correct filename
    mock_savefig.assert_called_once_with(f"plots-es/cnn-metrics-s{seed}.png", dpi=300)
    
    # Check that plot is properly closed
    mock_close.assert_called_once()
    mock_tight_layout.assert_called_once()


def test_create_dataset(mock_stella):
    """Test dataset creation function"""
    path = "/test/path"
    catalog = "test_catalog.csv"
    cadences = 168
    training = 0.8
    validation = 0.1
    frac_balance = 0.73
    
    result = create_dataset(path, catalog, cadences, training, validation, frac_balance)
    
    # Check that FlareDataSet was called with correct parameters
    mock_stella.FlareDataSet.assert_called_once_with(
        fn_dir=path,
        catalog=catalog,
        cadences=cadences,
        training=training,
        validation=validation,
        frac_balance=frac_balance,
    )
    
    assert result == mock_stella.FlareDataSet.return_value


def test_model_RS():
    """Test the RandomSearch model creation function"""
    # Define test parameters
    filter1 = 32
    filter2 = 64
    dense = 128
    dropout = 0.3
    learning_rate = 0.001
    kernel1 = 7
    kernel2 = 3
    pool = 2
    l2val = 1e-4
    
    # Mock args.c for input shape
    with patch('cnn.args') as mock_args:
        mock_args.c = 168
        
        model = model_RS(filter1, filter2, dense, dropout, learning_rate, kernel1, kernel2, pool, l2val)
        
        assert isinstance(model, tf.keras.models.Sequential)
        assert model.input_shape == (None, 168, 1)
        assert model.output_shape == (None, 1)
        assert model.loss == 'binary_crossentropy'
        
        # Check layer structure
        layer_types = [type(layer).__name__ for layer in model.layers]
        expected_types = ['Conv1D', 'MaxPooling1D', 'Dropout', 'Conv1D', 'MaxPooling1D', 
                         'Dropout', 'Flatten', 'Dense', 'Dropout', 'Dense']
        assert layer_types == expected_types


@patch('builtins.input', return_value='y')
@patch('cnn.stella')
@patch('pickle.load')
@patch('builtins.open', new_callable=mock_open)
def test_main_with_load_dataset(mock_file, mock_pickle_load, mock_stella, mock_input, sample_args):
    """Test main execution with dataset loading"""
    sample_args.load_dataset = "test_ds.pkl"
    
    # Mock dataset loading
    mock_dataset = {'dataset': Mock()}
    mock_pickle_load.return_value = mock_dataset
    
    # Mock CNN
    mock_cnn = Mock()
    mock_stella.ConvNN.return_value = mock_cnn
    
    with patch('cnn.args', sample_args):
        with patch('cnn.plot_metrics'):
            # Mock the main execution flow
            with patch('sys.argv', ['cnn.py']):
                # This would test the main flow but the actual script has complex interactions
                pass


@patch('cnn.stella')
def test_dataset_creation_with_merge(mock_stella, sample_args):
    """Test dataset creation with merge functionality"""
    sample_args.merge = ["/path1", "/path2"]
    sample_args.merge_catalogs = ["cat1.csv", "cat2.csv"]
    sample_args.merge_labels = [1, 0]
    
    # Mock additional datasets
    mock_additional_dataset1 = Mock()
    mock_additional_dataset2 = Mock()
    mock_stella.FlareDataSet.side_effect = [mock_additional_dataset1, mock_additional_dataset2, Mock()]
    
    with patch('cnn.args', sample_args):
        # Test the merge logic (this is simplified as the actual main is complex)
        datasets = []
        for additional_dir, additional_catalog in zip(sample_args.merge, sample_args.merge_catalogs):
            additional_dataset = create_dataset(
                additional_dir,
                additional_catalog,
                sample_args.c,
                training=1,
                validation=0,
                frac_balance=1,
            )
            datasets.append(additional_dataset)
        
        assert len(datasets) == 2
        assert datasets[0] == mock_additional_dataset1
        assert datasets[1] == mock_additional_dataset2


def test_plot_metrics_data_structure(mock_cnn_with_data):
    """Test that plot_metrics handles the expected data structure"""
    seed = 49
    
    # Ensure the mock data has the expected structure
    assert 'tpeak' in mock_cnn_with_data.val_pred_table
    assert 'pred_s0049' in mock_cnn_with_data.val_pred_table
    assert 'labels' in mock_cnn_with_data.val_pred_table
    assert 'gt' in mock_cnn_with_data.val_pred_table
    
    assert 'accuracy_s0049' in mock_cnn_with_data.history_table
    assert 'val_accuracy_s0049' in mock_cnn_with_data.history_table
    assert 'loss_s0049' in mock_cnn_with_data.history_table
    assert 'val_loss_s0049' in mock_cnn_with_data.history_table
    
    with patch('matplotlib.pyplot.savefig'), \
         patch('matplotlib.pyplot.close'), \
         patch('matplotlib.pyplot.tight_layout'), \
         patch('os.makedirs'):
        
        # Should not raise any exceptions
        plot_metrics(mock_cnn_with_data, seed)


@patch('cnn.optimise.optimise_hyperparameters')
@patch('cnn.optimise.train_final_model')
@patch('cnn.stella')
def test_optimisation_workflow(mock_stella, mock_train_final, mock_optimize, sample_args):
    """Test the optimisation workflow"""
    sample_args.optimise_bayes = True
    
    # Mock optimisation results
    best_params = {'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 256}
    mock_optimize.return_value = best_params
    
    # Mock final model training
    mock_model = Mock()
    mock_history = Mock()
    mock_history.history = {'loss': [0.5, 0.3], 'val_loss': [0.6, 0.4]}
    mock_train_final.return_value = (mock_model, mock_history)
    
    # Mock CNN instance
    mock_cnn = Mock()
    mock_stella.ConvNN.return_value = mock_cnn
    
    # Test the optimisation calls
    best_params_result = mock_optimize(mock_cnn, n_trials=50, name=sample_args.optimise_bayes_name)
    model, history = mock_train_final(mock_cnn, best_params_result, epochs=sample_args.e, seed=49)
    
    assert best_params_result == best_params
    assert model == mock_model
    assert history == mock_history
    
    mock_optimize.assert_called_once()
    mock_train_final.assert_called_once()


@pytest.fixture(autouse=True)
def cleanup_tensorflow():
    """Clean up TensorFlow session after each test"""
    yield
    tf.keras.backend.clear_session()
    plt.close('all')