import random
from functools import partial

import numpy as np
import torch

from basicsr.data.prefetch_dataloader import PrefetchDataLoader
from basicsr.data.single_h5_image_dataset import SingleH5ImageDataset
from basicsr.utils import get_root_logger
from basicsr.utils.dist_util import get_dist_info


def create_dataset(dataset_opt):
    dataset_type = dataset_opt["type"]
    if dataset_type != "SingleH5ImageDataset":
        raise ValueError(f"Unsupported dataset: {dataset_type}")
    dataset = SingleH5ImageDataset(dataset_opt)
    get_root_logger().info(
        f'Dataset {dataset.__class__.__name__} - {dataset_opt["name"]} is created.'
    )
    return dataset


def create_dataloader(
    dataset,
    dataset_opt,
    num_gpu=1,
    dist=False,
    sampler=None,
    seed=None,
):
    phase = dataset_opt["phase"]
    rank, _ = get_dist_info()
    if phase == "train":
        multiplier = 1 if dist or num_gpu == 0 else num_gpu
        batch_size = dataset_opt["batch_size_per_gpu"] * multiplier
        num_workers = dataset_opt["num_worker_per_gpu"] * multiplier
        args = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": sampler is None,
            "num_workers": num_workers,
            "sampler": sampler,
            "drop_last": True,
            "worker_init_fn": (
                partial(
                    worker_init_fn,
                    num_workers=num_workers,
                    rank=rank,
                    seed=seed,
                )
                if seed is not None
                else None
            ),
        }
    elif phase in {"val", "test"}:
        args = {
            "dataset": dataset,
            "batch_size": 1,
            "shuffle": False,
            "num_workers": dataset_opt.get("num_worker_per_gpu", 0),
        }
    else:
        raise ValueError(f"Unsupported dataset phase: {phase}")

    args["pin_memory"] = dataset_opt.get("pin_memory", False)
    if dataset_opt.get("prefetch_mode") == "cpu":
        queue_size = dataset_opt.get("num_prefetch_queue", 1)
        get_root_logger().info(
            f"Use cpu prefetch dataloader: num_prefetch_queue = {queue_size}"
        )
        return PrefetchDataLoader(num_prefetch_queue=queue_size, **args)
    return torch.utils.data.DataLoader(**args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


__all__ = ["create_dataset", "create_dataloader", "SingleH5ImageDataset"]
