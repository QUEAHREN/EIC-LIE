import logging
from os import path as osp

import torch

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (
    get_env_info,
    get_root_logger,
    get_time_str,
    make_exp_dirs,
)
from basicsr.utils.options import dict2str


def main():
    opt = parse_options(is_train=False)
    if not opt["path"].get("pretrain_network_g"):
        raise ValueError(
            "A checkpoint is required. Pass --weights /path/to/net_g_best.pth."
        )
    torch.backends.cudnn.benchmark = True
    make_exp_dirs(opt)
    log_file = osp.join(
        opt["path"]["log"], f'test_{opt["name"]}_{get_time_str()}.log'
    )
    logger = get_root_logger(
        logger_name="basicsr",
        log_level=logging.INFO,
        log_file=log_file,
    )
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    model = create_model(opt)

    for _, dataset_opt in sorted(opt["datasets"].items()):
        dataset = create_dataset(dataset_opt)
        loader = create_dataloader(
            dataset,
            dataset_opt,
            num_gpu=opt["num_gpu"],
            dist=opt["dist"],
            sampler=None,
            seed=opt["manual_seed"],
        )
        logger.info(f'Testing {dataset_opt["name"]} ({len(dataset)} images).')
        model.validation(
            loader,
            current_iter=opt["name"],
            tb_logger=None,
            save_img=opt["val"]["save_img"],
            rgb2bgr=opt["val"].get("rgb2bgr", False),
            use_image=opt["val"].get("use_image", True),
        )


if __name__ == "__main__":
    main()
