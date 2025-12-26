"""
CLDHits Dataset Adapter for PointTransformerV3

Loads calorimeter hit data from parquet files and formats it for PTv3.
"""

import sys
sys.path.append('/eventssl-vol')

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import pickle


def _load_single_parquet(args):
    """Load a single parquet file and extract all events. Used for parallel loading."""
    import awkward as ak
    import numpy as np

    file_path, file_idx = args
    events = []

    try:
        data = ak.from_parquet(file_path)
        num_events = len(data["genparticle_to_calo_hit_matrix"])

        for event_i in range(num_events):
            genparticle_to_calo_hit_matrix = data["genparticle_to_calo_hit_matrix"][event_i]
            calo_hit_features_raw = data["calo_hit_features"][event_i]

            gen_idx = genparticle_to_calo_hit_matrix["gen_idx"].to_numpy()
            hit_idx = genparticle_to_calo_hit_matrix["hit_idx"].to_numpy()
            weights = genparticle_to_calo_hit_matrix["weight"].to_numpy()

            calo_hit_features = np.column_stack((
                calo_hit_features_raw["position.x"].to_numpy(),
                calo_hit_features_raw["position.y"].to_numpy(),
                calo_hit_features_raw["position.z"].to_numpy(),
                calo_hit_features_raw["energy"].to_numpy(),
            ))

            hit_labels = _get_hit_labels_fast(hit_idx, gen_idx, weights)

            events.append({
                'hit_labels': hit_labels,
                'calo_hit_features': calo_hit_features,
            })
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return file_idx, []

    return file_idx, events


def _get_hit_labels_fast(hit_idx, gen_idx, weights):
    """Vectorized version of get_hit_labels."""
    if len(hit_idx) == 0:
        return np.array([], dtype=np.int64)

    max_hit = np.max(hit_idx) + 1
    hit_labels = np.full(max_hit, -1, dtype=np.int64)
    hit_weights = np.full(max_hit, -np.inf, dtype=np.float32)

    sort_idx = np.argsort(-weights)
    hit_idx_sorted = hit_idx[sort_idx]
    gen_idx_sorted = gen_idx[sort_idx]
    weights_sorted = weights[sort_idx]

    for h, g, w in zip(hit_idx_sorted, gen_idx_sorted, weights_sorted):
        if w > hit_weights[h]:
            hit_weights[h] = w
            hit_labels[h] = g

    return hit_labels


class CLDHitsDatasetPTv3(Dataset):
    """
    CLDHits dataset adapter for PointTransformerV3.

    Outputs data in the format expected by PTv3:
    - feat: (N, 4) features [x, y, z, energy]
    - coord: (N, 3) coordinates
    - grid_size: voxel size for grid coordinates
    - labels: (N,) instance labels
    """

    def __init__(self, root, split='train', voxel_size=50.0, num_points=100000,
                 train_fraction=0.8, nfiles=-1, num_workers=8, cache_dir=None, **kwargs):
        """
        Initialize CLDHits dataset for PTv3.

        Args:
            root: Path to folder containing parquet files
            split: 'train' or 'val'
            voxel_size: Grid size for PTv3 (in mm)
            num_points: Maximum points per event
            train_fraction: Fraction of data for training
            nfiles: Number of files to load (-1 for all)
            num_workers: Parallel workers for loading
            cache_dir: Cache directory
        """
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.split = split
        self.root = Path(root)

        print(f"Initializing CLDHits dataset (PTv3) for {split} split...")

        # Setup cache
        if cache_dir is None:
            cache_dir = self.root / '.cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Get parquet files
        parquet_files = sorted(self.root.glob("*.parquet"))
        split_index = int(len(parquet_files) * train_fraction)
        if split == 'train':
            parquet_files = parquet_files[:split_index]
        else:
            parquet_files = parquet_files[split_index:]

        if nfiles > 0:
            parquet_files = parquet_files[:nfiles]

        # Cache handling
        cache_key = self._get_cache_key(parquet_files, split)
        cache_path = self.cache_dir / f"cld_hits_ptv3_{split}_{cache_key}.pkl"

        if cache_path.exists():
            print(f"Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.data_list = pickle.load(f)
            print(f"Loaded {len(self.data_list)} events from cache")
        else:
            self.data_list = self._load_parallel(parquet_files, num_workers)
            print(f"Saving to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(self.data_list, f)

        self._compute_statistics()

    def _get_cache_key(self, parquet_files, split):
        """Generate cache key."""
        file_str = f"{split}:" + ",".join([f.name for f in parquet_files])
        return hashlib.md5(file_str.encode()).hexdigest()[:12]

    def _load_parallel(self, parquet_files, num_workers):
        """Load parquet files in parallel."""
        print(f"Loading {len(parquet_files)} files with {num_workers} workers...")

        args_list = [(str(f), i) for i, f in enumerate(parquet_files)]
        all_events = [None] * len(parquet_files)
        completed = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_load_single_parquet, args): args[1]
                      for args in args_list}

            for future in as_completed(futures):
                file_idx, events = future.result()
                all_events[file_idx] = events
                completed += 1
                if completed % 10 == 0 or completed == len(parquet_files):
                    print(f"  Loaded {completed}/{len(parquet_files)} files...")

        data_list = []
        for events in all_events:
            if events:
                data_list.extend(events)

        print(f"Loaded {len(data_list)} events total")
        return data_list

    def _compute_statistics(self):
        """Compute dataset statistics."""
        num_hits = [len(event['calo_hit_features']) for event in self.data_list]
        num_instances = [len(np.unique(event['hit_labels'][event['hit_labels'] >= 0]))
                        for event in self.data_list]

        print(f"\nDataset Statistics ({self.split}):")
        print(f"  Total events: {len(self.data_list)}")
        print(f"  Hits per event: mean={np.mean(num_hits):.1f}, "
              f"median={np.median(num_hits):.1f}, "
              f"min={np.min(num_hits)}, max={np.max(num_hits)}")
        print(f"  Instances per event: mean={np.mean(num_instances):.1f}, "
              f"median={np.median(num_instances):.1f}, "
              f"min={np.min(num_instances)}, max={np.max(num_instances)}\n")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Get a single event in PTv3 format.

        Returns:
            dict with:
                - feat: (N, 4) tensor [x, y, z, energy]
                - coord: (N, 3) tensor [x, y, z]
                - grid_size: float
                - labels: (N,) instance labels
        """
        event = self.data_list[idx]

        # Get features: [x, y, z, energy]
        hit_features = event['calo_hit_features'].astype(np.float32)
        instance_labels = event['hit_labels'].astype(np.int64)

        # Coordinates (keep original scale for PTv3)
        coords = hit_features[:, :3].copy()

        # Downsample if too many points
        if len(coords) > self.num_points:
            sample_idx = np.random.choice(len(coords), self.num_points, replace=False)
            coords = coords[sample_idx]
            hit_features = hit_features[sample_idx]
            instance_labels = instance_labels[sample_idx]

        # Convert to tensors
        coord = torch.from_numpy(coords).float()
        feat = torch.from_numpy(hit_features).float()
        labels = torch.from_numpy(instance_labels).long()

        return {
            'coord': coord,
            'feat': feat,
            'labels': labels,
            'grid_size': self.voxel_size,
            'idx': idx
        }


def cld_collate_fn_ptv3(batch):
    """
    Collate function for PTv3 format.

    PTv3 expects:
    - feat: (N_total, C) concatenated features
    - coord: (N_total, 3) concatenated coordinates
    - grid_size: float
    - offset: (B,) cumulative point counts per batch item
    - labels: (N_total,) with offset instance IDs

    Args:
        batch: List of items from CLDHitsDatasetPTv3.__getitem__

    Returns:
        dict ready for PTv3 forward pass
    """
    coords = []
    feats = []
    labels = []
    offset = []
    indices = []

    label_offset = 0
    cumulative_points = 0

    for item in batch:
        coord = item['coord']
        feat = item['feat']
        lbl = item['labels'].clone()

        n_points = coord.shape[0]

        # Offset labels to make unique across batch
        valid_mask = lbl >= 0
        if valid_mask.any():
            lbl[valid_mask] += label_offset
            label_offset = lbl[valid_mask].max().item() + 1

        coords.append(coord)
        feats.append(feat)
        labels.append(lbl)
        indices.append(item['idx'])

        cumulative_points += n_points
        offset.append(cumulative_points)

    # Concatenate
    coord_batch = torch.cat(coords, dim=0)
    feat_batch = torch.cat(feats, dim=0)
    labels_batch = torch.cat(labels, dim=0)
    offset_batch = torch.tensor(offset, dtype=torch.long)

    # Get grid_size from first item (should be same for all)
    grid_size = batch[0]['grid_size']

    return {
        'coord': coord_batch,
        'feat': feat_batch,
        'grid_size': grid_size,
        'offset': offset_batch,
        'labels': labels_batch,
        'indices': indices
    }


if __name__ == "__main__":
    # Test the dataset
    print("Testing CLDHitsDatasetPTv3...")

    dataset = CLDHitsDatasetPTv3(
        root="/eventssl-vol/particlemind/data/p8_ee_tt_ecm365/parquet_full",
        split='val',
        voxel_size=50.0,
        nfiles=2
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test single item
    item = dataset[0]
    print(f"\nSingle item:")
    print(f"  coord shape: {item['coord'].shape}")
    print(f"  feat shape: {item['feat'].shape}")
    print(f"  labels shape: {item['labels'].shape}")
    print(f"  grid_size: {item['grid_size']}")

    # Test collate
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=cld_collate_fn_ptv3)
    batch = next(iter(loader))

    print(f"\nBatch:")
    print(f"  coord shape: {batch['coord'].shape}")
    print(f"  feat shape: {batch['feat'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  offset: {batch['offset']}")
    print(f"  grid_size: {batch['grid_size']}")
