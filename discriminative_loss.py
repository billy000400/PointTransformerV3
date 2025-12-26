"""
Discriminative Loss for Instance Segmentation.

Based on "Semantic Instance Segmentation with a Discriminative Loss Function"
https://arxiv.org/abs/1708.02551

Copied from /eventssl-vol/spvnas/core/losses/discriminative_loss.py
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add

__all__ = ['DiscriminativeLoss']


class DiscriminativeLoss(nn.Module):
    """
    Discriminative Loss for Instance Segmentation (Vectorized Implementation).

    This loss function is designed for instance segmentation where the number of
    instances varies per sample. It encourages the network to learn embeddings where:

    1. Variance term (pull): Points belonging to the same instance are pulled together
    2. Distance term (push): Different instance clusters are pushed apart
    3. Regularization term: Instance cluster centers stay bounded near the origin

    This implementation uses torch_scatter for fully vectorized operations,
    avoiding Python for-loops for significant speedup (10-50x faster).
    """

    def __init__(self, delta_v=0.5, delta_d=1.5, alpha=1.0, beta=1.0, gamma=0.001):
        """
        Initialize discriminative loss.

        Args:
            delta_v: Margin for variance term (intra-cluster). Points within this
                    distance from cluster center incur no loss.
            delta_d: Margin for distance term (inter-cluster). Clusters farther than
                    2*delta_d apart incur no loss.
            alpha: Weight for variance term (pull force)
            beta: Weight for distance term (push force)
            gamma: Weight for regularization term
        """
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, embeddings, instance_labels):
        """
        Compute discriminative loss (vectorized).

        Args:
            embeddings: (N, D) - D-dimensional embeddings for N points
            instance_labels: (N,) - Instance ID for each point (e.g., gen_idx)
                           Points with label -1 are ignored

        Returns:
            loss: Scalar tensor containing the total loss
            loss_dict: Dictionary with individual loss components for logging
        """
        # Filter out ignore labels (e.g., -1 for unlabeled hits)
        valid_mask = instance_labels >= 0
        embeddings = embeddings[valid_mask]
        instance_labels = instance_labels[valid_mask]

        # Handle edge cases
        if len(embeddings) == 0:
            zero_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            return zero_loss, {'total': 0.0, 'variance': 0.0, 'distance': 0.0,
                              'regularization': 0.0, 'num_instances': 0}

        # Remap labels to contiguous indices [0, 1, 2, ..., C-1]
        # This is required for scatter operations
        unique_instances, remapped_labels = torch.unique(instance_labels, return_inverse=True)
        num_instances = len(unique_instances)

        if num_instances == 0:
            zero_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            return zero_loss, {'total': 0.0, 'variance': 0.0, 'distance': 0.0,
                              'regularization': 0.0, 'num_instances': 0}

        # ============================================================
        # 1. Compute cluster centers (vectorized using scatter_mean)
        # ============================================================
        # cluster_means: (C, D) where C = num_instances
        cluster_means = scatter_mean(embeddings, remapped_labels, dim=0)

        # ============================================================
        # 2. Variance term (pull force) - vectorized
        # ============================================================
        # Get the cluster center for each point
        point_centers = cluster_means[remapped_labels]  # (N, D)

        # Distance from each point to its cluster center
        distances = torch.norm(embeddings - point_centers, dim=1)  # (N,)

        # Hinge loss: only penalize if distance > delta_v
        hinged_distances = torch.clamp(distances - self.delta_v, min=0.0)  # (N,)
        hinged_sq = hinged_distances.pow(2)  # (N,)

        # Sum squared hinged distances per cluster
        cluster_var_sum = scatter_add(hinged_sq, remapped_labels, dim=0)  # (C,)

        # Count points per cluster
        ones = torch.ones_like(remapped_labels, dtype=torch.float32)
        cluster_counts = scatter_add(ones, remapped_labels, dim=0)  # (C,)

        # Average variance per cluster, then average across clusters
        cluster_var = cluster_var_sum / cluster_counts.clamp(min=1)  # (C,)
        var_loss = cluster_var.mean()

        # ============================================================
        # 3. Distance term (push force) - vectorized
        # ============================================================
        if num_instances > 1:
            # Compute pairwise distances between all cluster centers
            # Using broadcasting: (C, 1, D) - (1, C, D) -> (C, C, D)
            center_diffs = cluster_means.unsqueeze(1) - cluster_means.unsqueeze(0)  # (C, C, D)
            center_distances = torch.norm(center_diffs, dim=2)  # (C, C)

            # Hinge loss: only penalize if centers are too close (< 2*delta_d)
            margin = 2 * self.delta_d
            hinged_center_distances = torch.clamp(margin - center_distances, min=0.0)  # (C, C)
            hinged_center_sq = hinged_center_distances.pow(2)  # (C, C)

            # Only count upper triangle (exclude diagonal and lower triangle)
            # Use triu with diagonal=1 to get strictly upper triangular part
            upper_mask = torch.triu(torch.ones_like(hinged_center_sq, dtype=torch.bool), diagonal=1)
            upper_hinged_sq = hinged_center_sq[upper_mask]

            # Average over all pairs
            num_pairs = num_instances * (num_instances - 1) // 2
            dist_loss = upper_hinged_sq.sum() / max(num_pairs, 1)
        else:
            dist_loss = torch.tensor(0.0, device=embeddings.device)

        # ============================================================
        # 4. Regularization term - keep cluster centers bounded
        # ============================================================
        reg_loss = torch.norm(cluster_means, dim=1).mean()

        # ============================================================
        # 5. Combined weighted loss
        # ============================================================
        total_loss = self.alpha * var_loss + self.beta * dist_loss + self.gamma * reg_loss

        # ============================================================
        # 6. NaN safeguard - detect and handle numerical issues
        # ============================================================
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\n[WARNING] NaN/Inf detected in loss!")
            print(f"  var_loss: {var_loss.item()}, dist_loss: {dist_loss.item() if num_instances > 1 else 0.0}, reg_loss: {reg_loss.item()}")
            print(f"  num_instances: {num_instances}, num_points: {len(embeddings)}")
            print(f"  embeddings has NaN: {torch.isnan(embeddings).any().item()}")
            print(f"  cluster_means has NaN: {torch.isnan(cluster_means).any().item()}")
            # Return zero loss to skip this batch without crashing
            zero_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            return zero_loss, {'total': 0.0, 'variance': 0.0, 'distance': 0.0,
                              'regularization': 0.0, 'num_instances': num_instances, 'nan_detected': True}

        # Return loss components for logging
        loss_dict = {
            'total': total_loss.item(),
            'variance': var_loss.item(),
            'distance': dist_loss.item() if num_instances > 1 else 0.0,
            'regularization': reg_loss.item(),
            'num_instances': num_instances
        }

        return total_loss, loss_dict
