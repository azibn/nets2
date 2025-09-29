"""
Training script for the local+global fusion model.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from datetime import datetime
import matplotlib.pyplot as plt
from astropy.table import Table
import json

from data_generator import create_local_global_generators
from fusion_model import create_fusion_model

def create_callbacks(output_dir, patience=30, monitor='val_loss'):
    """Create training callbacks."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    callbacks = []
    
    # Model checkpoint
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(output_dir, 'best_model.h5'),
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        mode='min' if 'loss' in monitor else 'max'
    )
    callbacks.append(checkpoint)
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        verbose=1,
        mode='min' if 'loss' in monitor else 'max'
    )
    callbacks.append(early_stop)
    
    # Learning rate reduction
    lr_reduce = ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=patience//2,
        min_lr=1e-7,
        verbose=1,
        mode='min' if 'loss' in monitor else 'max'
    )
    callbacks.append(lr_reduce)
    
    # TensorBoard
    log_dir = os.path.join(output_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False
    )
    callbacks.append(tensorboard)
    
    return callbacks

def plot_training_history(history, output_dir):
    """Plot and save training history."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training & validation loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot training & validation accuracy
    if 'accuracy' in history.history:
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot precision if available
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot recall if available
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_generator, output_dir):
    """Evaluate model on test set and save results."""
    
    print("Evaluating model on test set...")
    
    # Get predictions for entire test set
    test_predictions = []
    test_labels = []
    test_local_features = []
    test_global_features = []
    
    for i in range(len(test_generator)):
        X_batch, y_batch = test_generator[i]
        
        # Get predictions
        batch_preds = model.predict(X_batch, verbose=0)
        test_predictions.extend(batch_preds.flatten())
        test_labels.extend(y_batch)
        
        # Get feature representations for analysis
        features = model.get_feature_representations(X_batch)
        test_local_features.extend(features['local_features'].numpy())
        test_global_features.extend(features['global_features'].numpy())
    
    test_predictions = np.array(test_predictions)
    test_labels = np.array(test_labels)
    test_local_features = np.array(test_local_features)
    test_global_features = np.array(test_global_features)
    
    # Calculate metrics
    from sklearn.metrics import (
        roc_auc_score, roc_curve, precision_recall_curve,
        average_precision_score, classification_report, confusion_matrix
    )
    
    # ROC curve
    fpr, tpr, _ = roc_curve(test_labels, test_predictions)
    roc_auc = roc_auc_score(test_labels, test_predictions)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(test_labels, test_predictions)
    pr_auc = average_precision_score(test_labels, test_predictions)
    
    # Classification report (using 0.5 threshold)
    test_pred_binary = (test_predictions > 0.5).astype(int)
    class_report = classification_report(test_labels, test_pred_binary, output_dict=True)
    conf_matrix = confusion_matrix(test_labels, test_pred_binary)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # ROC curve
    axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate') 
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Precision-Recall curve
    axes[0, 1].plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Prediction histogram
    axes[1, 0].hist(test_predictions[test_labels == 0], bins=50, alpha=0.5, 
                   label='Non-exocomet', density=True)
    axes[1, 0].hist(test_predictions[test_labels == 1], bins=50, alpha=0.5, 
                   label='Exocomet', density=True)
    axes[1, 0].set_xlabel('Prediction Probability')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Prediction Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Feature space visualization (2D projection)
    from sklearn.decomposition import PCA
    combined_features = np.concatenate([test_local_features, test_global_features], axis=1)
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(combined_features)
    
    scatter = axes[1, 1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=test_labels, cmap='viridis', alpha=0.6, s=10)
    axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[1, 1].set_title('Feature Space (PCA)')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save evaluation metrics
    metrics = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'n_test_samples': len(test_labels),
        'n_positive': int(np.sum(test_labels)),
        'n_negative': int(len(test_labels) - np.sum(test_labels))
    }
    
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    results_table = Table([
        test_generator.window_ids,
        test_generator.window_times,
        test_labels,
        test_predictions,
        test_pred_binary
    ], names=['tic_id', 'time', 'true_label', 'prediction', 'predicted_label'])
    
    results_table.write(os.path.join(output_dir, 'test_predictions.txt'), 
                       format='ascii', overwrite=True)
    
    print(f"Test Results:")
    print(f"- ROC AUC: {roc_auc:.4f}")
    print(f"- PR AUC: {pr_auc:.4f}")
    print(f"- Accuracy: {class_report['accuracy']:.4f}")
    print(f"- Precision: {class_report['1']['precision']:.4f}")
    print(f"- Recall: {class_report['1']['recall']:.4f}")
    print(f"- F1-Score: {class_report['1']['f1-score']:.4f}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train local+global fusion model for exocomet detection")
    
    # Dataset arguments
    parser.add_argument("dataset_path", help="Path to dataset with global context")
    parser.add_argument("-o", "--output-dir", default="fusion_training_output", 
                       help="Output directory for models and results")
    
    # Model architecture arguments
    parser.add_argument("--local-window", type=int, default=168,
                       help="Size of local window")
    parser.add_argument("--global-window-size", type=int, default=500,
                       help="Size of global context after downsampling")
    parser.add_argument("--global-window-days", type=float, default=3.0,
                       help="Days around event for global context")
    parser.add_argument("--model-type", choices=['standard', 'attention'], default='standard',
                       help="Type of fusion model")
    parser.add_argument("--rnn-type", choices=['variability', 'residual', 'hierarchical'], 
                       default='variability', help="Type of global RNN")
    parser.add_argument("--cnn-filters", nargs='+', type=int, default=[16, 64],
                       help="CNN filter sizes")
    parser.add_argument("--cnn-kernels", nargs='+', type=int, default=[7, 3],
                       help="CNN kernel sizes")
    parser.add_argument("--rnn-hidden", type=int, default=128,
                       help="RNN hidden units")
    parser.add_argument("--fusion-units", nargs='+', type=int, default=[64, 32],
                       help="Fusion network layer sizes")
    parser.add_argument("--use-existing-cnn", type=str, default=None,
                       help="Path to existing CNN model to use")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=200,
                       help="Maximum number of epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Initial learning rate")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate")
    parser.add_argument("--l2-reg", type=float, default=0.01,
                       help="L2 regularization factor")
    parser.add_argument("--patience", type=int, default=30,
                       help="Early stopping patience")
    parser.add_argument("--monitor", default="val_loss",
                       help="Metric to monitor for early stopping")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU to use (-1 for CPU)")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Configure GPU
    if args.gpu >= 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and len(gpus) > args.gpu:
            tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
            tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
        else:
            print(f"Warning: GPU {args.gpu} not available, using CPU")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Creating data generators...")
    train_gen, val_gen, test_gen = create_local_global_generators(
        dataset_path=args.dataset_path,
        local_window=args.local_window,
        global_window_size=args.global_window_size,
        global_window_days=args.global_window_days,
        batch_size=args.batch_size
    )
    
    print("Creating fusion model...")
    model = create_fusion_model(
        model_type=args.model_type,
        local_window=args.local_window,
        global_window=args.global_window_size,
        cnn_filters=args.cnn_filters,
        cnn_kernels=args.cnn_kernels,
        rnn_hidden=args.rnn_hidden,
        rnn_type=args.rnn_type,
        fusion_units=args.fusion_units,
        dropout_rate=args.dropout,
        l2_reg=args.l2_reg,
        use_existing_cnn=args.use_existing_cnn is not None,
        existing_cnn_path=args.use_existing_cnn
    )
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.F1Score(name='f1_score', average='macro')
        ]
    )
    
    # Print model summary
    print(f"\nModel Summary:")
    print(f"Model type: {args.model_type}")
    print(f"RNN type: {args.rnn_type}")
    print(f"Total parameters: {model.count_params():,}")
    
    # Create callbacks
    callbacks = create_callbacks(args.output_dir, args.patience, args.monitor)
    
    print(f"\nStarting training...")
    print(f"- Training batches: {len(train_gen)}")
    print(f"- Validation batches: {len(val_gen)}")
    print(f"- Max epochs: {args.epochs}")
    print(f"- Early stopping patience: {args.patience}")
    
    # Train model
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, args.output_dir)
    
    # Load best model for evaluation
    best_model_path = os.path.join(args.output_dir, 'best_model.h5')
    if os.path.exists(best_model_path):
        print("Loading best model for evaluation...")
        model = tf.keras.models.load_model(best_model_path)
    
    # Evaluate model
    metrics = evaluate_model(model, test_gen, args.output_dir)
    
    print(f"\nTraining completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Best model saved as: {best_model_path}")

if __name__ == "__main__":
    main()