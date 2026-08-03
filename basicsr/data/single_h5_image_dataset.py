import random
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class SingleH5ImageDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        roots = opt["dataroot"]
        roots = roots if isinstance(roots, list) else [roots]
        self.paths = []
        for root in roots:
            root_path = Path(root).expanduser()
            if root_path.is_file() and root_path.suffix.lower() == ".h5":
                self.paths.append(root_path)
            elif root_path.is_dir():
                self.paths.extend(sorted(root_path.glob("*.h5")))
            else:
                raise FileNotFoundError(f"H5 data path does not exist: {root_path}")
        self.paths = sorted(self.paths)
        if not self.paths:
            raise FileNotFoundError(f"No .h5 files found under: {roots}")

        self.crop_size = opt.get("crop_size")
        self.use_flip = opt.get("use_flip", False)
        self.use_rot = opt.get("use_rot", False)
        self.norm_voxel = opt.get("norm_voxel", True)

    @staticmethod
    def _to_chw(array, expected_channels=None):
        tensor = torch.from_numpy(array).float()
        if tensor.ndim == 2:
            return tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(f"Expected a 2D or 3D array, got shape {tuple(tensor.shape)}")
        if expected_channels and tensor.shape[0] == expected_channels:
            return tensor
        if expected_channels and tensor.shape[-1] == expected_channels:
            return tensor.permute(2, 0, 1).contiguous()
        if tensor.shape[0] <= 32 and tensor.shape[1] > 32 and tensor.shape[2] > 32:
            return tensor
        if tensor.shape[-1] <= 32:
            return tensor.permute(2, 0, 1).contiguous()
        raise ValueError(f"Cannot infer channel order for shape {tuple(tensor.shape)}")

    @staticmethod
    def _scale_image(tensor):
        return tensor / 255.0 if tensor.max() > 1.0 else tensor

    def _augment(self, tensors):
        height, width = tensors[0].shape[-2:]
        if any(tensor.shape[-2:] != (height, width) for tensor in tensors):
            raise ValueError("image, sharp, and voxel must have identical spatial sizes")

        if self.crop_size is not None:
            size = int(self.crop_size)
            if height < size or width < size:
                raise ValueError(
                    f"Crop size {size} exceeds sample size {(height, width)}"
                )
            top = random.randint(0, height - size)
            left = random.randint(0, width - size)
            tensors = [
                tensor[:, top : top + size, left : left + size]
                for tensor in tensors
            ]

        if self.use_flip:
            dims = []
            if random.random() < 0.5:
                dims.append(-1)
            if random.random() < 0.5:
                dims.append(-2)
            if dims:
                tensors = [torch.flip(tensor, dims=dims) for tensor in tensors]

        if self.use_rot:
            turns = random.randint(0, 3)
            if turns:
                tensors = [
                    torch.rot90(tensor, turns, dims=(-2, -1)) for tensor in tensors
                ]
        return tensors

    def __getitem__(self, index):
        path = self.paths[index]
        with h5py.File(path, "r") as handle:
            missing = {"image", "sharp", "voxel"} - set(handle.keys())
            if missing:
                raise KeyError(f"{path} is missing H5 keys: {sorted(missing)}")
            lq = self._to_chw(handle["image"][:], expected_channels=3)
            gt = self._to_chw(handle["sharp"][:], expected_channels=3)
            event = self._to_chw(handle["voxel"][:], expected_channels=6)

        lq = self._scale_image(lq)
        gt = self._scale_image(gt)
        if self.norm_voxel:
            max_abs = event.abs().max()
            if max_abs > 0:
                event = event / max_abs
        lq, gt, event = self._augment([lq, gt, event])
        return {
            "LQ": lq,
            "GT": gt,
            "Event": event,
            "LQ_path": str(path),
            "seq": path.stem,
        }

    def __len__(self):
        return len(self.paths)
