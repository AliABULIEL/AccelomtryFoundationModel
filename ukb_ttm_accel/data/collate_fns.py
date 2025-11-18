# FILE: ukb_ttm_accel/data/collate_fns.py

"""
Custom collate functions for PyTorch DataLoader.

Handles batching of accelerometry windows with clinical metadata.
"""

import torch
from typing import List, Dict, Any


def accel_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for accelerometry dataset.

    Batches multiple samples into tensors, handling:
    - Variable-length metadata (padding with zeros)
    - Creating attention masks for missing data
    - Stacking signals, labels, and other features

    Args:
        batch: List of sample dictionaries from UKBAccelDataset.__getitem__

    Returns:
        Batched dictionary with keys:
        - signals: FloatTensor[batch_size, window_length, num_channels]
        - labels: LongTensor[batch_size] or FloatTensor[batch_size]
        - participant_ids: LongTensor[batch_size]
        - metadata: FloatTensor[batch_size, num_metadata_features]
        - metadata_mask: BoolTensor[batch_size, num_metadata_features]
                        (True = valid, False = padded/missing)
        - time_of_day: FloatTensor[batch_size]

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=accel_collate_fn)
        >>> batch = next(iter(loader))
        >>> print(batch['signals'].shape)
        torch.Size([32, 500, 3])
    """
    # Extract components from batch
    signals = []
    labels = []
    participant_ids = []
    metadata_list = []
    time_of_day_list = []

    for sample in batch:
        signals.append(sample['signal'])
        labels.append(sample['label'])
        participant_ids.append(sample['participant_id'])
        metadata_list.append(sample['metadata'])
        time_of_day_list.append(sample['time_of_day'])

    # Stack signals: [batch_size, window_length, num_channels]
    signals_batch = torch.stack(signals, dim=0)

    # Stack labels: [batch_size]
    labels_batch = torch.stack(labels, dim=0)

    # Stack participant IDs: [batch_size]
    participant_ids_batch = torch.stack(participant_ids, dim=0)

    # Stack time of day: [batch_size]
    time_of_day_batch = torch.stack(time_of_day_list, dim=0)

    # Handle metadata (may have different lengths across samples)
    # Determine max metadata length in batch
    max_metadata_len = max(len(m) for m in metadata_list)

    if max_metadata_len > 0:
        # Pad metadata to same length
        metadata_padded = []
        metadata_mask = []

        for metadata in metadata_list:
            # Pad with zeros
            pad_length = max_metadata_len - len(metadata)

            if pad_length > 0:
                padded = torch.cat([
                    metadata,
                    torch.zeros(pad_length, dtype=metadata.dtype)
                ])
                mask = torch.cat([
                    torch.ones(len(metadata), dtype=torch.bool),
                    torch.zeros(pad_length, dtype=torch.bool)
                ])
            else:
                padded = metadata
                mask = torch.ones(len(metadata), dtype=torch.bool)

            metadata_padded.append(padded)
            metadata_mask.append(mask)

        metadata_batch = torch.stack(metadata_padded, dim=0)
        metadata_mask_batch = torch.stack(metadata_mask, dim=0)
    else:
        # No metadata
        metadata_batch = torch.zeros((len(batch), 0), dtype=torch.float32)
        metadata_mask_batch = torch.zeros((len(batch), 0), dtype=torch.bool)

    return {
        'signals': signals_batch,
        'labels': labels_batch,
        'participant_ids': participant_ids_batch,
        'metadata': metadata_batch,
        'metadata_mask': metadata_mask_batch,
        'time_of_day': time_of_day_batch,
    }


def ssl_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for self-supervised learning.

    Similar to accel_collate_fn, but optimized for SSL where labels may not be needed.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary (labels are included but may be ignored in SSL training)
    """
    # For SSL, we can reuse the same collate function
    # The SSL training loop will simply not use the 'labels' key
    return accel_collate_fn(batch)


def create_custom_collate_fn(include_labels: bool = True) -> callable:
    """
    Factory function to create custom collate functions.

    Args:
        include_labels: Whether to include labels in batch

    Returns:
        Custom collate function

    Example:
        >>> collate_fn = create_custom_collate_fn(include_labels=False)
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    """
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        result = accel_collate_fn(batch)

        if not include_labels:
            result.pop('labels', None)

        return result

    return collate_fn
