from __future__ import annotations

import copy

import pandas as pd

from ..data.dataset import TabularDataset


def combine_datasets(dataset_list: list[TabularDataset], strict_metafeature=True) -> TabularDataset:
    """Combine a list of TabularDataset into a single TabularDataset.

    Args:
        dataset_list (list[TabularDataset]): List of TabularDataset to combine.

    Returns:
        TabularDataset: Combined TabularDataset.
    """
    # ===== Sanity check =====
    task_type = dataset_list[0].task_type
    metafeature_dict = dataset_list[0].metafeature_dict
    for dataset in dataset_list:
        # === All datasets must have the same task type ===
        if dataset.task_type != task_type:
            raise ValueError("All datasets must have the same task type.")
        # === All datasets must have the same metafeature_dict ===
        if strict_metafeature and dataset.metafeature_dict != metafeature_dict:
            raise ValueError("All datasets must have the same metafeature_dict.")

    # ===== Build a large TabularDataset with all provided datasets =====
    # DataFrame accepts duplicate indices
    data_df_combined = pd.concat([dataset.data_df for dataset in dataset_list], axis=0)
    dataset_combined = copy.deepcopy(dataset_list[0])
    dataset_combined.dataset_name = f"{dataset_combined.dataset_name}_combined"
    dataset_combined.data_df = data_df_combined

    return dataset_combined
