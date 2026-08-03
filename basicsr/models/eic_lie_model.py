import importlib
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info

loss_module = importlib.import_module("basicsr.models.losses")
metric_module = importlib.import_module("basicsr.metrics")


class EICLIEModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.net_g = self.model_to_device(define_network(opt["network_g"]))
        load_path = opt["path"].get("pretrain_network_g")
        if load_path:
            self.load_network(
                self.net_g,
                load_path,
                opt["path"].get("strict_load_g", True),
                param_key=opt["path"].get("param_key", "params"),
            )
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]
        self.best_metric_value = float("-inf")
        self.best_metric_iter = -1
        pixel_opt = deepcopy(train_opt["pixel_opt"])
        loss_type = pixel_opt.pop("type")
        self.cri_pix = getattr(loss_module, loss_type)(**pixel_opt).to(self.device)
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        optim_opt = deepcopy(self.opt["train"]["optim_g"])
        optim_type = optim_opt.pop("type")
        if optim_type != "AdamW":
            raise ValueError(f"EIC-LIE release configs require AdamW, got {optim_type}")

        regular, low_lr, high_lr = [], [], []
        for name, parameter in self.net_g.named_parameters():
            if not parameter.requires_grad:
                get_root_logger().warning(f"Parameter {name} is frozen.")
            elif name.startswith("module.offsets") or name.startswith("module.dcns"):
                low_lr.append(parameter)
            elif name.endswith("alpha"):
                high_lr.append(parameter)
            else:
                regular.append(parameter)

        base_lr = optim_opt["lr"]
        groups = [
            {"params": regular},
            {"params": low_lr, "lr": base_lr * 0.1},
            {"params": high_lr, "lr": base_lr},
        ]
        self.optimizer_g = torch.optim.AdamW(groups, **optim_opt)
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data["LQ"].to(self.device)
        self.voxel = data["Event"].to(self.device)
        self.gt = data["GT"].to(self.device) if "GT" in data else None
        self.seq = data.get("seq")
        self.lq_path = data.get("LQ_path")

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        prediction = self.net_g(torch.cat([self.lq, self.voxel], dim=1))
        predictions = prediction if isinstance(prediction, list) else [prediction]
        self.output = predictions[-1]
        pixel_loss = sum(self.cri_pix(item, self.gt) for item in predictions)
        pixel_loss.backward()
        if self.opt["train"].get("use_grad_clip", False):
            max_norm = self.opt["train"].get("clip_grad_norm", 0.01)
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(
            OrderedDict([("l_pix", pixel_loss)])
        )

    def test(self):
        self.net_g.eval()
        batch_size = self.lq.size(0)
        mini_batch = self.opt["val"].get("max_minibatch", batch_size)
        outputs = []
        with torch.no_grad():
            for start in range(0, batch_size, mini_batch):
                end = min(start + mini_batch, batch_size)
                lq = self.lq[start:end]
                event = self.voxel[start:end]
                height, width = lq.shape[-2:]
                pad_h = (32 - height % 32) % 32
                pad_w = (32 - width % 32) % 32
                if pad_h or pad_w:
                    lq = F.pad(lq, (0, pad_w, 0, pad_h), mode="reflect")
                    event = F.pad(event, (0, pad_w, 0, pad_h), mode="reflect")
                output = self.net_g(torch.cat([lq, event], dim=1))
                if isinstance(output, list):
                    output = output[-1]
                outputs.append(output[:, :, :height, :width])
        self.output = torch.cat(outputs, dim=0)
        if self.is_train:
            self.net_g.train()

    def dist_validation(
        self,
        dataloader,
        current_iter,
        tb_logger,
        save_img,
        rgb2bgr,
        use_image,
    ):
        rank, _ = get_dist_info()
        if rank == 0:
            return self.nondist_validation(
                dataloader,
                current_iter,
                tb_logger,
                save_img,
                rgb2bgr,
                use_image,
            )
        return 0.0

    @staticmethod
    def _unwrap_name(value, fallback):
        while isinstance(value, (list, tuple)) and value:
            value = value[0]
        if value:
            return osp.splitext(osp.basename(str(value)))[0]
        return fallback

    def nondist_validation(
        self,
        dataloader,
        current_iter,
        tb_logger,
        save_img,
        rgb2bgr,
        use_image,
    ):
        dataset_name = dataloader.dataset.opt["name"]
        metrics_opt = self.opt["val"].get("metrics")
        metric_results = (
            {name: 0.0 for name in metrics_opt} if metrics_opt is not None else None
        )
        progress = tqdm(total=len(dataloader), unit="image")

        for index, data in enumerate(dataloader):
            self.feed_data(data)
            image_name = self._unwrap_name(
                data.get("seq"),
                self._unwrap_name(data.get("LQ_path"), f"{index:08d}"),
            )
            self.test()
            visuals = self.get_current_visuals()
            result_image = tensor2img([visuals["result"]], rgb2bgr=rgb2bgr)
            gt_image = (
                tensor2img([visuals["gt"]], rgb2bgr=rgb2bgr)
                if "gt" in visuals
                else None
            )

            if save_img:
                suffix = f"_{current_iter}" if self.is_train else ""
                save_path = osp.join(
                    self.opt["path"]["visualization"],
                    dataset_name,
                    f"{image_name}{suffix}.png",
                )
                imwrite(result_image, save_path)

            if metric_results is not None:
                if gt_image is None:
                    raise ValueError("Ground truth is required to calculate metrics.")
                for name, options in metrics_opt.items():
                    metric_options = deepcopy(options)
                    metric_type = metric_options.pop("type")
                    if use_image:
                        score = getattr(metric_module, metric_type)(
                            result_image, gt_image, **metric_options
                        )
                    else:
                        score = getattr(metric_module, metric_type)(
                            visuals["result"], visuals["gt"], **metric_options
                        )
                    metric_results[name] += score

            progress.update(1)
            progress.set_description(f"Test {image_name}")
            del self.lq, self.voxel, self.output
            if self.gt is not None:
                del self.gt
            torch.cuda.empty_cache()

        progress.close()
        if metric_results is None:
            return 0.0
        self.metric_results = {
            name: value / len(dataloader) for name, value in metric_results.items()
        }
        self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        key_metric = self.opt["val"].get("key_metric", "psnr")
        return self.metric_results.get(key_metric, next(iter(self.metric_results.values())))

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        values = "".join(
            f"\t # {name}: {value:.4f}"
            for name, value in self.metric_results.items()
        )
        logger = get_root_logger()
        logger.info(f"Validation {dataset_name},{values}")

        if self.is_train and self.opt["val"].get("save_best", False):
            key = self.opt["val"].get("key_metric", "psnr")
            value = self.metric_results[key]
            if value > self.best_metric_value:
                self.best_metric_value = value
                self.best_metric_iter = current_iter
                self.save_network(self.net_g, "net_g", "best")
                logger.info(
                    f"New best {key}: {value:.4f} at iter {current_iter}; "
                    "saved net_g_best.pth."
                )
        if tb_logger:
            for name, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{name}", value, current_iter)

    def get_current_visuals(self):
        visuals = OrderedDict(
            lq=self.lq.detach().cpu(),
            result=self.output.detach().cpu(),
        )
        if self.gt is not None:
            visuals["gt"] = self.gt.detach().cpu()
        return visuals

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)
