import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import argparse
import logging
from tqdm import tqdm
import json
from datetime import datetime
import random

from dataset import TumorAggressivenessDataset, TumorAggressivenessDataLoader
from model import TumorClassificationModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate various classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def train_epoch(model, train_loader, criterion, optimizer, device, fusion_strategy):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, labels) in enumerate(pbar):
        # Move data to device
        if fusion_strategy == 'early':
            data = data.to(device)
        else:
            data = [d.to(device) for d in data]
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Collect predictions
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
        
        total_loss += loss.item()
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities[:, 1].detach().cpu().numpy())  # Probability of positive class
        
        # Update progress bar
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_predictions, all_probabilities)
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics

def validate_epoch(model, val_loader, criterion, device, fusion_strategy):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch_idx, (data, labels) in enumerate(pbar):
            # Move data to device
            if fusion_strategy == 'early':
                data = data.to(device)
            else:
                data = [d.to(device) for d in data]
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # Collect predictions
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_predictions, all_probabilities)
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics

def test_model(model, test_loader, device, fusion_strategy):
    """Test the model and return detailed results."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_sample_info = []
    
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Testing"):
            # Move data to device
            if fusion_strategy == 'early':
                data = data.to(device)
            else:
                data = [d.to(device) for d in data]
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].detach().cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_predictions, all_probabilities)
    
    return metrics, {
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }

def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    return epoch, metrics

def main():
    parser = argparse.ArgumentParser(description='Train tumor aggressiveness classification model')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='Data', help='Root directory for data')
    parser.add_argument('--target_size', type=int, nargs=3, default=[128, 128, 128], help='Target size for resampling (D H W)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--fusion_strategy', type=str, default='early', 
                       choices=['early', 'intermediate', 'single', 'ensemble'], 
                       help='Fusion strategy for CT sequences')
    parser.add_argument('--ct_types', type=str, nargs='+', default=['A', 'D', 'N', 'V'], 
                       help='CT sequence types to use')
    parser.add_argument('--model_depth', type=int, default=18, help='ResNet depth')
    parser.add_argument('--fusion_method', type=str, default='concat', 
                       choices=['concat', 'attention', 'weighted_sum'], 
                       help='Fusion method for intermediate fusion')
    parser.add_argument('--ensemble_method', type=str, default='voting', 
                       choices=['voting', 'weighted_voting', 'stacking'], 
                       help='Ensemble method')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.fusion_strategy}_fusion_{timestamp}"
    
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    data_loader = TumorAggressivenessDataLoader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=tuple(args.target_size),
        fusion_strategy=args.fusion_strategy,
        ct_types=args.ct_types,
        normalize=True,
        apply_voi_mask=True
    )
    
    train_loader = data_loader.create_train_loader()
    internal_val_loader = data_loader.create_internal_val_loader()
    external_test_loader = data_loader.create_external_test_loader()
    
    logger.info(f"Train samples: {len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else 'Unknown'}")
    logger.info(f"Internal validation samples: {len(internal_val_loader.dataset) if hasattr(internal_val_loader.dataset, '__len__') else 'Unknown'}")
    logger.info(f"External test samples: {len(external_test_loader.dataset) if hasattr(external_test_loader.dataset, '__len__') else 'Unknown'}")
    
    # Create model
    logger.info("Creating model...")
    model = TumorClassificationModel(
        fusion_strategy=args.fusion_strategy,
        num_sequences=len(args.ct_types),
        num_classes=2,
        spatial_dims=3,
        model_depth=args.model_depth,
        fusion_method=args.fusion_method,
        ensemble_method=args.ensemble_method,
        dropout_rate=args.dropout_rate
    )
    
    model = model.to(device)
    model_info = model.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metrics = None
    if args.resume:
        start_epoch, best_metrics = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
    
    # Training loop
    logger.info("Starting training...")
    best_auc = 0
    patience_counter = 0
    patience = 20
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, args.fusion_strategy)
        
        # Validate
        val_metrics = validate_epoch(model, internal_val_loader, criterion, device, args.fusion_strategy)
        
        # Log metrics
        for metric, value in train_metrics.items():
            writer.add_scalar(f'train/{metric}', value, epoch)
        for metric, value in val_metrics.items():
            writer.add_scalar(f'val/{metric}', value, epoch)
        
        # Update learning rate
        scheduler.step(val_metrics['auc'])
        
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)
        
        # Print metrics
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, "
                   f"Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, "
                   f"Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_metrics = val_metrics
            patience_counter = 0
            
            # Save best checkpoint
            best_checkpoint_path = os.path.join(output_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, best_checkpoint_path)
            logger.info(f"New best model saved with AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping after {patience} epochs without improvement")
            break
    
    # Load best model for testing
    logger.info("Loading best model for testing...")
    load_checkpoint(model, optimizer, os.path.join(output_dir, 'best_model.pth'))
    
    # Test on internal validation set
    logger.info("Testing on internal validation set...")
    internal_test_metrics, internal_results = test_model(
        model, internal_val_loader, device, args.fusion_strategy
    )
    
    # Test on external test set
    logger.info("Testing on external test set...")
    external_test_metrics, external_results = test_model(
        model, external_test_loader, device, args.fusion_strategy
    )
    
    # Save test results
    test_results = {
        'internal_test': internal_test_metrics,
        'external_test': external_test_metrics,
        'internal_predictions': internal_results,
        'external_predictions': external_results
    }
    
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    # Print final results
    logger.info("Final Results:")
    logger.info(f"Internal Test - AUC: {internal_test_metrics['auc']:.4f}, "
               f"Accuracy: {internal_test_metrics['accuracy']:.4f}")
    logger.info(f"External Test - AUC: {external_test_metrics['auc']:.4f}, "
               f"Accuracy: {external_test_metrics['accuracy']:.4f}")
    
    # Close tensorboard writer
    writer.close()
    
    logger.info(f"Training completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 