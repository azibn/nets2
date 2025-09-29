import pytest
import numpy as np
import tensorflow as tf
import optuna
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optimise import create_model_with_params, objective, optimise_hyperparameters, train_final_model


@pytest.fixture
def mock_cnn_instance():
    """Create a mock CNN instance for testing"""
    mock_cnn = Mock()
    mock_cnn.cadences = 168
    
    # Mock dataset
    mock_ds = Mock()
    mock_ds.train_data = np.random.random((100, 168, 1))
    mock_ds.val_data = np.random.random((20, 168, 1))
    mock_ds.train_labels = np.random.randint(0, 2, 100)
    mock_ds.val_labels = np.random.randint(0, 2, 20)
    
    mock_cnn.ds = mock_ds
    return mock_cnn


@pytest.fixture
def test_params():
    """Test parameters for model creation"""
    return {
        'dropout': 0.3,
        'l2_lambda': 1e-4,
        'learning_rate': 1e-3,
        'batch_size': 256
    }


def test_create_model_with_params(mock_cnn_instance, test_params):
    """Test model creation with parameters"""
    model = create_model_with_params(mock_cnn_instance, test_params)
    
    assert isinstance(model, tf.keras.models.Sequential)
    assert model.input_shape == (None, 168, 1)
    assert model.output_shape == (None, 1)
    assert model.loss == 'binary_crossentropy'
    
    # Check that the model has the expected layers
    layer_types = [type(layer).__name__ for layer in model.layers]
    expected_types = ['Conv1D', 'MaxPooling1D', 'Dropout', 'Conv1D', 'MaxPooling1D', 
                     'Dropout', 'Flatten', 'Dense', 'Dropout', 'Dense']
    assert layer_types == expected_types


def test_create_model_with_params_invalid_cadences(test_params):
    """Test model creation with invalid cadences"""
    mock_cnn = Mock()
    mock_cnn.cadences = 0
    
    with pytest.raises(ValueError):
        create_model_with_params(mock_cnn, test_params)


@patch('tensorflow.keras.models.Sequential.fit')
def test_objective(mock_fit, mock_cnn_instance):
    """Test the objective function for optimization"""
    # Mock the fit method to return a history-like object
    mock_history = Mock()
    mock_history.history = {'val_auc': [0.5, 0.6, 0.7, 0.8]}
    mock_fit.return_value = mock_history
    
    # Create a mock trial
    mock_trial = Mock()
    mock_trial.suggest_float.side_effect = [0.3, 1e-4, 1e-3]  # dropout, l2_lambda, learning_rate
    mock_trial.suggest_int.return_value = 256  # batch_size
    
    result = objective(mock_trial, mock_cnn_instance)
    
    assert result == 0.8  # Last value from val_auc
    mock_fit.assert_called_once()


def test_objective_parameter_ranges(mock_cnn_instance):
    """Test that objective function suggests parameters within expected ranges"""
    mock_trial = Mock()
    mock_trial.suggest_float.side_effect = [0.3, 1e-4, 1e-3]
    mock_trial.suggest_int.return_value = 256
    
    with patch('tensorflow.keras.models.Sequential.fit') as mock_fit:
        mock_history = Mock()
        mock_history.history = {'val_auc': [0.8]}
        mock_fit.return_value = mock_history
        
        objective(mock_trial, mock_cnn_instance)
        
        # Check that parameters were suggested with correct ranges
        calls = mock_trial.suggest_float.call_args_list
        assert calls[0][0] == ("dropout", 0.1, 0.5)
        assert calls[1][0] == ("l2_lambda", 1e-6, 1e-2)
        assert calls[1][1] == {"log": True}
        assert calls[2][0] == ("learning_rate", 1e-4, 1e-2)
        assert calls[2][1] == {"log": True}
        
        mock_trial.suggest_int.assert_called_with("batch_size", 128, 1024, step=256)


@patch('optuna.create_study')
def test_optimise_hyperparameters(mock_create_study, mock_cnn_instance):
    """Test hyperparameter optimisation"""
    # Mock study
    mock_study = Mock()
    mock_trial = Mock()
    mock_trial.value = 0.85
    mock_trial.params = {'dropout': 0.2, 'l2_lambda': 1e-5, 'learning_rate': 5e-4, 'batch_size': 512}
    mock_study.best_trial = mock_trial
    mock_study.best_params = mock_trial.params
    mock_create_study.return_value = mock_study
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_name = os.path.join(temp_dir, 'test_optimization.db')
        
        result = optimise_hyperparameters(
            mock_cnn_instance, 
            n_trials=5, 
            name=db_name, 
            show_progress_bar=False
        )
        
        assert result == mock_trial.params
        mock_create_study.assert_called_once()
        mock_study.optimize.assert_called_once()


@patch('tensorflow.keras.models.Sequential.fit')
@patch('tensorflow.random.set_seed')
@patch('tensorflow.keras.backend.clear_session')
def test_train_final_model(mock_clear_session, mock_set_seed, mock_fit, mock_cnn_instance, test_params):
    """Test training final model with best parameters"""
    # Mock the fit method
    mock_history = Mock()
    mock_history.history = {'loss': [0.5, 0.3, 0.2], 'val_loss': [0.6, 0.4, 0.3]}
    mock_fit.return_value = mock_history
    
    model, history = train_final_model(
        mock_cnn_instance, 
        test_params, 
        epochs=10, 
        seed=42
    )
    
    assert isinstance(model, tf.keras.models.Sequential)
    assert history == mock_history
    
    mock_clear_session.assert_called_once()
    mock_set_seed.assert_called_once_with(42)
    mock_fit.assert_called_once()
    
    # Check that early stopping callback was added
    fit_call_args = mock_fit.call_args
    callbacks = fit_call_args[1]['callbacks']
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], tf.keras.callbacks.EarlyStopping)


def test_train_final_model_with_different_epochs(mock_cnn_instance, test_params):
    """Test training final model with different epoch values"""
    with patch('tensorflow.keras.models.Sequential.fit') as mock_fit:
        mock_history = Mock()
        mock_fit.return_value = mock_history
        
        train_final_model(mock_cnn_instance, test_params, epochs=50, seed=123)
        
        fit_call_args = mock_fit.call_args[1]
        assert fit_call_args['epochs'] == 50
        assert fit_call_args['batch_size'] == test_params['batch_size']


def test_parameters_validation(mock_cnn_instance):
    """Test parameter validation in functions"""
    # Test with invalid parameters
    invalid_params = {
        'dropout': -0.1,  # Invalid negative dropout
        'l2_lambda': 1e-4,
        'learning_rate': 1e-3,
        'batch_size': 256
    }
    
    # Model creation should still work (TensorFlow will handle invalid values)
    model = create_model_with_params(mock_cnn_instance, invalid_params)
    assert isinstance(model, tf.keras.models.Sequential)


@pytest.fixture(autouse=True)
def cleanup_tensorflow():
    """Clean up TensorFlow session after each test"""
    yield
    tf.keras.backend.clear_session()