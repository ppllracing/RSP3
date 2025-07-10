import math

import torch
import torch.nn.functional as F

from torch import nn
from ...tools.config import Configuration


class SegmentationHead(nn.Module):
    def __init__(self, cfg: Configuration):
        super(SegmentationHead, self).__init__()
        # 提取基本数据
        self.cfg = cfg
        # self.device = cfg.device
        self.dtype = cfg.dtype_model_torch
        self.model_params = cfg.segmentation_head_params

        self.segmentation_head = nn.Sequential(
            nn.Upsample(size=self.cfg.map_bev['final_dim'][:2], mode='bilinear', align_corners=False),
            nn.Conv2d(self.model_params['inp_channel'], self.model_params['mid_channel'], kernel_size=5, padding=2, bias=False, dtype=self.dtype),
            nn.BatchNorm2d(self.model_params['mid_channel'], affine=True, dtype=self.dtype),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(self.model_params['mid_channel'], self.model_params['num_class'], kernel_size=1, padding=0, dtype=self.dtype)
        )

    def forward(self, bev_feature):
        # 进行语义分割
        bev_feature = bev_feature.permute(0, 2, 1)  # [B, N, F] -> [B, C, H*W]
        B, C, _ = bev_feature.shape
        bev_f = bev_feature.reshape(B, C, *self.cfg.map_bev['map_down_sample'])  # [B, C, H, W]
        segmentation = self.segmentation_head(bev_f)  # [B, num_class, fH, fH]

        # 将语义分割结果进行one-hot操作
        with torch.no_grad():
            segmentation_argmax = torch.argmax(segmentation, dim=1)  # [B, fH, fH]
            segmentation_onehot = F.one_hot(segmentation_argmax, num_classes=self.model_params['num_class']).permute(0, 3, 1, 2)  # [B, num_class, fH, fH]

        oups = {
            'segmentation': segmentation,
            'segmentation_onehot': segmentation_onehot
        }
        return oups
