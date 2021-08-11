import torch
from torch import nn
from model.network.backbone import Backbone
from model.layers import ShapeSpec
from model.network import ResnetBackbone
from model.network import CenternetDeconv
from model.network import CenternetHead
from model.network import CenterNet


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = ResnetBackbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_upsample_layers(cfg, ):
    upsample = CenternetDeconv(cfg)
    return upsample


def build_head(cfg, ):
    head = CenternetHead(cfg)
    return head


def build_model(cfg):

    cfg.build_backbone = build_backbone
    cfg.build_upsample_layers = build_upsample_layers
    cfg.build_head = build_head
    model = CenterNet(cfg)
    return model


def load_model(cfg, num_classes=15, path=None, freeze=True):
    model = build_model(cfg)
    cfg.MODEL.CENTERNET.NUM_CLASSES = num_classes
    if not path:
        model.load_state_dict(torch.load('model/weights/resnet50_centernet.pth')['model'])
    model.head.cls_head.out_conv = nn.Conv2d(64, num_classes, 1, 1)
    if path:
        model.load_state_dict(torch.load(path))
    model = model.cuda()
    if freeze:
        for param in model.backbone.parameters():
            param.requires_grad=False
        for param in model.upsample.parameters():
            param.requires_grad=False
    return model, cfg

