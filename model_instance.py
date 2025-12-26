"""
PointTransformerV3 for Instance Segmentation

Outputs embeddings instead of class logits for clustering-based instance segmentation.
Based on discriminative loss approach (same as SPVNAS adaptation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model import PointTransformerV3, Point
except ImportError:
    from model import PointTransformerV3, Point


class PointTransformerV3_Instance(nn.Module):
    """
    PointTransformerV3 adapted for instance segmentation.

    Outputs L2-normalized embeddings for each point, which can be clustered
    using DBSCAN or similar algorithms to obtain instance predictions.
    """

    def __init__(
        self,
        embedding_dim=128,
        in_channels=4,  # x, y, z, energy
        # PTv3 architecture params (smaller defaults for efficiency)
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        drop_path=0.3,
        enable_flash=True,
        enable_rpe=False,
        **kwargs
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Initialize PTv3 backbone
        self.backbone = PointTransformerV3(
            in_channels=in_channels,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            drop_path=drop_path,
            enable_flash=enable_flash,
            enable_rpe=enable_rpe,
            cls_mode=False,  # We need decoder for dense prediction
            **kwargs
        )

        # Final decoder output channels
        final_channels = dec_channels[0]  # 64 by default

        # Embedding head: maps decoder features to embedding space
        self.embedding_head = nn.Sequential(
            nn.Linear(final_channels, final_channels),
            nn.BatchNorm1d(final_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(final_channels, embedding_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize embedding head weights."""
        for m in self.embedding_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, data_dict):
        """
        Forward pass.

        Args:
            data_dict: Dictionary containing:
                - 'feat': (N, C) features [x, y, z, energy, ...]
                - 'coord': (N, 3) coordinates
                - 'grid_size': float, voxel size for grid coordinates
                - 'offset' or 'batch': batch indexing

        Returns:
            embeddings: (N, embedding_dim) L2-normalized embeddings
        """
        # Forward through PTv3 backbone
        point = self.backbone(data_dict)

        # Get per-point features
        features = point.feat  # (N, final_channels)

        # Map to embedding space
        embeddings = self.embedding_head(features)

        # L2 normalize for better clustering
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def get_num_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Smaller variant for faster experimentation
class PointTransformerV3_Instance_Small(PointTransformerV3_Instance):
    """Smaller PTv3 instance segmentation model."""

    def __init__(self, embedding_dim=64, in_channels=4, **kwargs):
        super().__init__(
            embedding_dim=embedding_dim,
            in_channels=in_channels,
            enc_depths=(1, 1, 1, 3, 1),
            enc_channels=(32, 64, 128, 256, 256),
            enc_num_head=(2, 4, 8, 16, 16),
            enc_patch_size=(512, 512, 512, 512, 512),
            dec_depths=(1, 1, 1, 1),
            dec_channels=(32, 32, 64, 128),
            dec_num_head=(2, 2, 4, 8),
            dec_patch_size=(512, 512, 512, 512),
            drop_path=0.1,
            **kwargs
        )


if __name__ == "__main__":
    # Quick test
    print("Testing PointTransformerV3_Instance...")

    model = PointTransformerV3_Instance(
        embedding_dim=32,
        in_channels=4,
        enable_flash=False  # Disable flash attention for CPU testing
    )

    print(f"Model parameters: {model.get_num_params():,}")

    # Create dummy input
    N = 1000
    data_dict = {
        'feat': torch.randn(N, 4),
        'coord': torch.randn(N, 3) * 100,
        'grid_size': 1.0,
        'batch': torch.zeros(N, dtype=torch.long)
    }

    # Forward pass
    embeddings = model(data_dict)
    print(f"Output shape: {embeddings.shape}")
    print(f"Embeddings normalized: {torch.allclose(embeddings.norm(dim=1), torch.ones(N))}")
