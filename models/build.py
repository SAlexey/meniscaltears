from hydra.utils import call, instantiate


def detr(backbone, transformer, model):
    backbone = call(backbone)
    transformer = call(transformer)
    return instantiate(model, backbone=backbone, transformer=transformer)
