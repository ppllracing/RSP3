import math
import torch
import torch.nn.functional as F
from torch import nn
from ...tools.config import Configuration

from .convolutions import UpsamplingConcat

class HeuristicHead(nn.Module):
    def __init__(self, cfg: Configuration):
        super(HeuristicHead, self).__init__()

        # 提取基本数据
        self.cfg = cfg
        # self.device = cfg.device
        self.dtype = cfg.dtype_model_torch
        self.model_params = cfg.heuristic_head_params

        # 定义网络层
        inp_channel = self.model_params['inp_channel']
        oup_channel = self.model_params['oup_channel']
        self.feature_extract = nn.Sequential(
            nn.Upsample(scale_factor=5, mode='bilinear', align_corners=True),  # [16, 16] -> [80, 80]
            nn.Conv2d(inp_channel, oup_channel, kernel_size=5, padding=2, bias=False),  # [80, 80] -> [80, 80]
            nn.ReLU(inplace=True),
            nn.Upsample(size=cfg.map_bev['final_dim'][:2], mode='bilinear', align_corners=True),  # [80, 80] -> [64, 72]
            nn.Conv2d(oup_channel, oup_channel, kernel_size=5, padding=2, bias=False),  # [64, 72] -> [64, 72]
            nn.Sigmoid()
        )

    def forward(self, inp_feature):
        assert len(inp_feature.shape) == 3, "inp_feature.shape should be [B, C, F]"
        assert inp_feature.shape[-1] == 256, "inp_feature.shape[-1] should be 256"

        B, C, _ = inp_feature.shape
        inp_feature_remap = inp_feature.reshape(B, C, 16, 16)
        heuristic_fig = self.feature_extract(inp_feature_remap)  # [B, 1, 64, 72]
        heuristic_fig = heuristic_fig.squeeze(1)  # [B, 64, 72]

        oups = {
            'heuristic_fig': heuristic_fig
        }
        return oups