import cv2
import numpy as np
import torch

from basicsr.metrics.metric_util import reorder_image, to_y_channel


def _prepare(img, input_order):
    if isinstance(img, torch.Tensor):
        if img.ndim == 4:
            img = img.squeeze(0)
        img = img.detach().cpu().numpy().transpose(1, 2, 0)
        input_order = "HWC"
    return reorder_image(img, input_order=input_order).astype(np.float64)


def calculate_psnr(
    img1,
    img2,
    crop_border,
    input_order="HWC",
    test_y_channel=False,
):
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes differ: {img1.shape} and {img2.shape}")
    img1 = _prepare(img1, input_order)
    img2 = _prepare(img2, input_order)
    if crop_border:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    data_range = 1.0 if max(img1.max(), img2.max()) <= 1.0 else 255.0
    return 20.0 * np.log10(data_range / np.sqrt(mse))


def _ssim_channel(img1, img2, data_range):
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    sigma1 = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1**2
    sigma2 = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2**2
    sigma12 = (
        cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1 * mu2
    )
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2)
    return (numerator / denominator).mean()


def calculate_ssim(
    img1,
    img2,
    crop_border,
    input_order="HWC",
    test_y_channel=False,
):
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes differ: {img1.shape} and {img2.shape}")
    img1 = _prepare(img1, input_order)
    img2 = _prepare(img2, input_order)
    if crop_border:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    data_range = 1.0 if max(img1.max(), img2.max()) <= 1.0 else 255.0
    if img1.shape[2] == 1:
        return _ssim_channel(img1[..., 0], img2[..., 0], data_range)
    return float(
        np.mean(
            [
                _ssim_channel(img1[..., channel], img2[..., channel], data_range)
                for channel in range(img1.shape[2])
            ]
        )
    )
