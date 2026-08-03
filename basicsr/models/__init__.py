from basicsr.models.eic_lie_model import EICLIEModel
from basicsr.utils import get_root_logger


def create_model(opt):
    model_type = opt["model_type"]
    if model_type != "EICLIEModel":
        raise ValueError(f"Unsupported model: {model_type}")
    model = EICLIEModel(opt)
    get_root_logger().info(f"Model [{model.__class__.__name__}] is created.")
    return model


__all__ = ["create_model", "EICLIEModel"]
