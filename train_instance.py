#!/usr/bin/env python3
"""
Training script for instance segmentation on CLDHits dataset using PointTransformerV3.

This script trains a PointTransformerV3 model with discriminative loss for particle
instance segmentation. Each calorimeter hit is embedded in a feature space
where hits from the same particle cluster together.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/eventssl-vol')

from dataset_cld import CLDHitsDatasetPTv3, cld_collate_fn_ptv3
from model_instance import PointTransformerV3_Instance, PointTransformerV3_Instance_Small
from discriminative_loss import DiscriminativeLoss


class InstanceSegmentationTrainer:
    """Trainer for instance segmentation with discriminative loss using PTv3."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

        # Create tensorboard writer
        self.writer = SummaryWriter(log_dir=self.output_dir / 'logs')

        # Initialize datasets
        self._init_datasets()

        # Initialize model
        self._init_model()

        # Initialize loss and optimizer
        self._init_training()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

    def _init_datasets(self):
        """Initialize train and validation datasets."""
        print("\n" + "="*80)
        print("INITIALIZING DATASETS")
        print("="*80)

        data_config = self.config['dataset']

        self.train_dataset = CLDHitsDatasetPTv3(
            root=data_config['root'],
            split='train',
            voxel_size=data_config['voxel_size'],
            num_points=data_config['num_points'],
            train_fraction=data_config.get('train_fraction', 0.8),
            nfiles=data_config.get('nfiles', -1),
            num_workers=data_config.get('load_workers', 8)
        )

        self.val_dataset = CLDHitsDatasetPTv3(
            root=data_config['root'],
            split='val',
            voxel_size=data_config['voxel_size'],
            num_points=data_config['num_points'],
            train_fraction=data_config.get('train_fraction', 0.8),
            nfiles=data_config.get('nfiles', -1),
            num_workers=data_config.get('load_workers', 8)
        )

        # Create dataloaders
        num_workers = self.config['training']['num_workers']
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=cld_collate_fn_ptv3,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=cld_collate_fn_ptv3,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )

        print(f"\nDataloaders created:")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")

    def _init_model(self):
        """Initialize model."""
        print("\n" + "="*80)
        print("INITIALIZING MODEL")
        print("="*80)

        model_config = self.config['model']
        model_type = model_config.get('type', 'default')

        if model_type == 'small':
            self.model = PointTransformerV3_Instance_Small(
                embedding_dim=model_config['embedding_dim'],
                in_channels=model_config.get('in_channels', 4),
                enable_flash=model_config.get('enable_flash', True),
                enable_rpe=model_config.get('enable_rpe', False),
            ).to(self.device)
            print(f"\nModel: PointTransformerV3_Instance_Small")
        else:
            self.model = PointTransformerV3_Instance(
                embedding_dim=model_config['embedding_dim'],
                in_channels=model_config.get('in_channels', 4),
                enc_depths=tuple(model_config.get('enc_depths', [2, 2, 2, 6, 2])),
                enc_channels=tuple(model_config.get('enc_channels', [32, 64, 128, 256, 512])),
                enc_num_head=tuple(model_config.get('enc_num_head', [2, 4, 8, 16, 32])),
                enc_patch_size=tuple(model_config.get('enc_patch_size', [1024, 1024, 1024, 1024, 1024])),
                dec_depths=tuple(model_config.get('dec_depths', [2, 2, 2, 2])),
                dec_channels=tuple(model_config.get('dec_channels', [64, 64, 128, 256])),
                dec_num_head=tuple(model_config.get('dec_num_head', [4, 4, 8, 16])),
                dec_patch_size=tuple(model_config.get('dec_patch_size', [1024, 1024, 1024, 1024])),
                drop_path=model_config.get('drop_path', 0.3),
                enable_flash=model_config.get('enable_flash', True),
                enable_rpe=model_config.get('enable_rpe', False),
            ).to(self.device)
            print(f"\nModel: PointTransformerV3_Instance")

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"  Embedding dim: {model_config['embedding_dim']}")
        print(f"  Total parameters: {num_params:,}")
        print(f"  Trainable parameters: {num_trainable:,}")

    def _init_training(self):
        """Initialize loss function and optimizer."""
        print("\n" + "="*80)
        print("INITIALIZING TRAINING")
        print("="*80)

        loss_config = self.config['loss']
        train_config = self.config['training']

        # Loss function
        self.criterion = DiscriminativeLoss(
            delta_v=loss_config['delta_v'],
            delta_d=loss_config['delta_d'],
            alpha=loss_config['alpha'],
            beta=loss_config['beta'],
            gamma=loss_config['gamma']
        )

        print(f"\nLoss: DiscriminativeLoss")
        print(f"  delta_v: {loss_config['delta_v']}")
        print(f"  delta_d: {loss_config['delta_d']}")
        print(f"  alpha: {loss_config['alpha']}")
        print(f"  beta: {loss_config['beta']}")
        print(f"  gamma: {loss_config['gamma']}")

        # Optimizer
        optimizer_type = train_config.get('optimizer', 'adamw').lower()
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=train_config['lr'],
                weight_decay=train_config.get('weight_decay', 1e-4)
            )
        elif optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=train_config['lr'],
                weight_decay=train_config.get('weight_decay', 1e-4)
            )
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=train_config['lr'],
                momentum=train_config.get('momentum', 0.9),
                weight_decay=train_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        print(f"\nOptimizer: {optimizer_type.upper()}")
        print(f"  Learning rate: {train_config['lr']}")
        print(f"  Weight decay: {train_config.get('weight_decay', 1e-4)}")

        # Learning rate scheduler
        if train_config.get('scheduler') == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['epochs'],
                eta_min=train_config.get('min_lr', 1e-6)
            )
            print(f"  Scheduler: CosineAnnealing")
        elif train_config.get('scheduler') == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_config.get('step_size', 10),
                gamma=train_config.get('gamma', 0.1)
            )
            print(f"  Scheduler: StepLR")
        elif train_config.get('scheduler') == 'onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=train_config['lr'],
                epochs=train_config['epochs'],
                steps_per_epoch=len(self.train_loader),
                pct_start=0.3
            )
            print(f"  Scheduler: OneCycleLR")
        else:
            self.scheduler = None
            print(f"  Scheduler: None")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {
            'variance': 0.0,
            'distance': 0.0,
            'regularization': 0.0,
            'num_instances': 0.0
        }

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            coord = batch['coord'].to(self.device)
            feat = batch['feat'].to(self.device)
            offset = batch['offset'].to(self.device)
            labels = batch['labels'].to(self.device)
            grid_size = batch['grid_size']

            # Create data dict for PTv3
            data_dict = {
                'coord': coord,
                'feat': feat,
                'offset': offset,
                'grid_size': grid_size
            }

            # Forward pass
            try:
                embeddings = self.model(data_dict)
            except Exception as e:
                print(f"\n\nERROR at batch {batch_idx}!")
                print(f"  Num points: {len(coord)}")
                print(f"  Coord min: {coord.min(dim=0).values.tolist()}")
                print(f"  Coord max: {coord.max(dim=0).values.tolist()}")
                raise e

            # Compute loss
            loss, loss_dict = self.criterion(embeddings, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )

            self.optimizer.step()

            # Step scheduler for OneCycleLR
            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            # Update metrics
            epoch_loss += loss.item()
            for key in epoch_metrics:
                if key in loss_dict:
                    epoch_metrics[key] += loss_dict[key]

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'inst': f'{loss_dict.get("num_instances", 0):.0f}'
            })

            # TensorBoard logging
            if batch_idx % self.config['training']['log_interval'] == 0:
                self.writer.add_scalar('batch/loss', loss.item(), self.global_step)
                for key, value in loss_dict.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'batch/{key}', value, self.global_step)

        # Average metrics
        num_batches = len(self.train_loader)
        epoch_loss /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_loss, epoch_metrics

    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        val_loss = 0.0
        val_metrics = {
            'variance': 0.0,
            'distance': 0.0,
            'regularization': 0.0,
            'num_instances': 0.0
        }

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for batch in pbar:
                # Move to device
                coord = batch['coord'].to(self.device)
                feat = batch['feat'].to(self.device)
                offset = batch['offset'].to(self.device)
                labels = batch['labels'].to(self.device)
                grid_size = batch['grid_size']

                # Create data dict for PTv3
                data_dict = {
                    'coord': coord,
                    'feat': feat,
                    'offset': offset,
                    'grid_size': grid_size
                }

                # Forward pass
                embeddings = self.model(data_dict)

                # Compute loss
                loss, loss_dict = self.criterion(embeddings, labels)

                # Update metrics
                val_loss += loss.item()
                for key in val_metrics:
                    if key in loss_dict:
                        val_metrics[key] += loss_dict[key]

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Average metrics
        num_batches = len(self.val_loader)
        val_loss /= num_batches
        for key in val_metrics:
            val_metrics[key] /= num_batches

        return val_loss, val_metrics

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"  Saved best checkpoint to {best_path}")

        # Save periodic checkpoint
        if (self.current_epoch + 1) % self.config['training']['save_interval'] == 0:
            epoch_path = self.output_dir / f'checkpoint_epoch_{self.current_epoch:03d}.pth'
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"  Resumed from epoch {self.current_epoch}")
        print(f"  Best loss: {self.best_loss:.4f}")

    def train(self):
        """Main training loop."""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)

        num_epochs = self.config['training']['epochs']
        start_epoch = self.current_epoch

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)

            # Train
            train_loss, train_metrics = self.train_epoch()

            # Validate
            val_loss, val_metrics = self.validate()

            # Update learning rate (for non-OneCycleLR schedulers)
            if self.scheduler is not None and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"    - Variance: {train_metrics['variance']:.4f}")
            print(f"    - Distance: {train_metrics['distance']:.4f}")
            print(f"    - Regularization: {train_metrics['regularization']:.4f}")
            print(f"    - Avg Instances: {train_metrics['num_instances']:.1f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"    - Variance: {val_metrics['variance']:.4f}")
            print(f"    - Distance: {val_metrics['distance']:.4f}")
            print(f"    - Regularization: {val_metrics['regularization']:.4f}")
            print(f"    - Avg Instances: {val_metrics['num_instances']:.1f}")
            print(f"  Learning Rate: {current_lr:.6f}")

            # TensorBoard logging
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/lr', current_lr, epoch)
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'epoch/train_{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'epoch/val_{key}', value, epoch)

            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            self.save_checkpoint(is_best=is_best)

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Best validation loss: {self.best_loss:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train PTv3 instance segmentation on CLDHits')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override output dir if specified
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir

    # Create trainer
    trainer = InstanceSegmentationTrainer(config)

    # Resume from checkpoint if specified
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
