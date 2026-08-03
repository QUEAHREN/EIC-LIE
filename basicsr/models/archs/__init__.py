from copy import deepcopy

from basicsr.models.archs.EIC_LIE_arch import EICLIE


def define_network(opt):
    options = deepcopy(opt)
    network_type = options.pop("type")
    if network_type != "EIC-LIE":
        raise ValueError(f"Unsupported network: {network_type}")
    return EICLIE(**options)


__all__ = ["define_network", "EICLIE"]
