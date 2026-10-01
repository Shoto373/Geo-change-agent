from models.backbone import PretrainedSiameseBackbone
from models.joint_fusion import BITJointFusion
from models.caption_decoder import CaptionDecoder
from models.multitask_model import MultiTaskChangeModel

# Псевдоним для совместимости
SiameseEncoder = PretrainedSiameseBackbone

__all__ = [
    "PretrainedSiameseBackbone",
    "SiameseEncoder",
    "BITJointFusion",
    "CaptionDecoder",
    "MultiTaskChangeModel"
]
