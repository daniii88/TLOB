import numpy as np
import torch


def engine_load(path: str, len_smooth: int, h: int, seq_size: int):
    """Load engine-export dataset for TLOB/MLPLOB training.

    Expected npy shape:
      [num_rows, num_features + 4]
    with last 4 columns as horizon labels in order: h10, h20, h50, h100.
    """
    dataset = np.load(path)
    if dataset.ndim != 2 or dataset.shape[1] < 5:
        raise ValueError(
            f"invalid ENGINE dataset `{path}`: expected 2D array with >=5 columns, got {dataset.shape}"
        )

    horizon_to_col_from_end = {10: 4, 20: 3, 50: 2, 100: 1}
    if h not in horizon_to_col_from_end:
        raise ValueError(f"unsupported horizon `{h}` for ENGINE dataset; expected 10|20|50|100")

    label_col_from_end = horizon_to_col_from_end[h]
    labels = dataset[seq_size - len_smooth :, -label_col_from_end]
    labels = labels[np.isfinite(labels)]
    labels = torch.from_numpy(labels).long()
    features = torch.from_numpy(dataset[:, :-4]).float()
    return features, labels
