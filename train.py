import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
import argparse
import logging
from tqdm import tqdm
import json
from datetime import datetime
import random

# --- Imports from your project files ---
# Assumes your model is in 'model.py' and your dataset is in 'dataset.py'
from model import MultiSequenceResNet
from model_2d import MultiSequenceResNet2DRes18, MultiSequenceResNet2DEfficientNet, MultiSequenceViT2D
# from monai_dataset_v2 import MONAITumorDataLoader
from monai_dataset_smart import MONAITumorDataLoader
from config import Config

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """A class to encapsulate the training and validation loop."""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._setup_output_dir()
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'tensorboard'))

        self.model = self._init_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=10, verbose=True)
        
        # Correctly inverted weights for class imbalance
        weights = torch.tensor([101.0/561.0, 460.0/561.0], device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        # Load data
        self.train_loader, self.val_loader = self._init_dataloaders()
        
        self.start_epoch = 0
        self.best_auc = 0.0
        self.patience_counter = 0

    def _setup_output_dir(self):
        """Creates the output directory for the experiment."""
        if self.config.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config.experiment_name = f"{self.config.fusion_method}_fusion_{timestamp}"
        
        self.output_dir = os.path.join(self.config.output_dir, self.config.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(self.config), f, indent=2)

    def _init_model(self):
        """Initializes the appropriate model (2D or 3D) based on the config."""
        if self.config.use_2d_slices:
            logger.info("Initializing 2D Model...")
            model = MultiSequenceViT2D(
                ct_types=self.config.ct_types,
                num_classes=2,
                fusion_method=self.config.fusion_method, # Only concat is supported for now
                dropout_rate=self.config.dropout_rate
            ).to(self.device)
        else:
            logger.info("Initializing 3D Model...")
            model = MultiSequenceResNet(
                ct_types=self.config.ct_types,
                num_classes=2,
                model_depth=self.config.model_depth,
                fusion_method=self.config.fusion_method,
                dropout_rate=self.config.dropout_rate,
                pretrained=self.config.pretrained,
                pretrained_path=self.config.local_pretrained_path
            ).to(self.device)
            
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")
        return model

    def _init_dataloaders(self):
        """Initializes the training and validation data loaders."""
        # Create a dictionary of arguments for the DataLoader factory.
        # This is a clean and robust way to manage parameters.
        loader_params = {
            "data_root": self.config.data_root,
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "replace_rate": 1.0,
            # Pass dataset-specific args
            "ct_types": self.config.ct_types,
            "target_size": tuple(self.config.target_size),
            "apply_voi_mask": self.config.apply_voi_mask,
            "use_2d_slices": self.config.use_2d_slices,
            "dilate_roi_size": self.config.dilate_roi_size,
        }
        
        loader_factory = MONAITumorDataLoader(**loader_params)
        return loader_factory.get_train_loader(), loader_factory.get_val_loader()

    def _run_epoch(self, epoch, is_train):
        """Runs a single epoch of training or validation."""
        self.model.train(is_train)
        loader = self.train_loader if is_train else self.val_loader
        phase = "Training" if is_train else "Validation"
        
        total_loss = 0
        all_labels, all_preds, all_probs = [], [], []

        pbar = tqdm(loader, desc=f"{phase} Epoch {epoch+1}/{self.config.epochs}")
        for batch_data in pbar:
            inputs = {k: v.to(self.device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}
            labels = inputs.pop('label')

            with torch.set_grad_enabled(is_train):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            total_loss += loss.item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        return self._calculate_metrics(all_labels, all_preds, all_probs, total_loss / len(loader))

    def _calculate_metrics(self, y_true, y_pred, y_prob, avg_loss):
        """Calculates and returns a dictionary of metrics."""
        pred_counts = np.bincount(y_pred, minlength=2)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)[0],
            'recall': precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)[1],
            'f1': precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)[2],
            'auc': roc_auc_score(y_true, y_prob),
            'pred_counts': pred_counts
        }

    def train(self):
        """The main training loop."""
        logger.info("Starting training...")
        for epoch in range(self.start_epoch, self.config.epochs):
            train_metrics = self._run_epoch(epoch, is_train=True)
            val_metrics = self._run_epoch(epoch, is_train=False)

            self._log_metrics(epoch, train_metrics, val_metrics)
            self.scheduler.step(val_metrics['auc'])

            if self._check_for_improvement(epoch, val_metrics):
                break
        
        self.writer.close()
        logger.info(f"Training completed. Best validation AUC: {self.best_auc:.4f}")

    def _log_metrics(self, epoch, train_metrics, val_metrics):
        """Logs metrics to console and TensorBoard."""
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        
        val_pred_counts = val_metrics['pred_counts']
        val_pred_dist_str = f"Pred Dist [0, 1]: [{val_pred_counts[0]}, {val_pred_counts[1]}]"
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['accuracy']:.4f} | {val_pred_dist_str}")
        
        for metric, value in train_metrics.items():
            if metric != 'pred_counts':
                self.writer.add_scalar(f'train/{metric}', value, epoch)
        for metric, value in val_metrics.items():
            if metric != 'pred_counts':
                self.writer.add_scalar(f'val/{metric}', value, epoch)
        self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

    def _check_for_improvement(self, epoch, val_metrics):
        """Handles checkpointing and early stopping."""
        if val_metrics['auc'] > self.best_auc:
            self.best_auc = val_metrics['auc']
            self.patience_counter = 0
            self._save_checkpoint(epoch, 'best_model.pth')
            logger.info(f"New best model saved with AUC: {self.best_auc:.4f}")
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.config.patience:
            logger.info(f"Early stopping after {self.config.patience} epochs without improvement.")
            return True
        return False

    def _save_checkpoint(self, epoch, filename):
        """Saves a model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_auc': self.best_auc
        }
        path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")


def main():
    parser = argparse.ArgumentParser(description='Train a 3D Fusion Model for Tumor Classification')
    
    # Paths and experiment
    parser.add_argument('--data_root', type=str, default='Data', help='Root directory for data')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None, help='Name for the experiment')

    # Data parameters
    parser.add_argument('--ct_types', type=str, nargs='+', default=['A'], help='CT sequence types to use')
    parser.add_argument('--target_size', type=int, nargs=3, default=[96, 96, 96], help='Target spatial size (D H W)')
    parser.add_argument('--apply_voi_mask', action='store_true', help='Apply the VOI mask to the images')
    
    # Model parameters
    parser.add_argument('--model_depth', type=int, default=18, help='ResNet depth (18, 34, 50, etc.)')
    parser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'attention'], help='Method for fusing features')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate in the classifier')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Arguments for pre-trained weights
    parser.add_argument('--pretrained', action='store_true', help='Use pre-trained weights from MedicalNet')
    parser.add_argument('--local_pretrained_path', type=str, 
                        default='pre_trained/tencent_pretrain/resnet_18_23dataset.pth', 
                        help='Path to local pre-trained weights file')
    parser.add_argument('--use_2d_slices', action='store_true', 
                        help='If set, uses a 2D model on the largest mask slice instead of the full 3D volume.')
    
    parser.add_argument('--dilate_roi_size', type=int, default=0, 
                        help='Kernel size for mask dilation (e.g., 3 or 5). 0 means no dilation')


    args = parser.parse_args()

    if args.use_2d_slices:
        # 2D ResNet models expect at least 224x224 for ImageNet pre-training to be effective
        logger.info("2D mode enabled. Overriding target_size to (224, 224).")
        args.target_size = [224, 224]

    set_seed(args.seed)
    
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
