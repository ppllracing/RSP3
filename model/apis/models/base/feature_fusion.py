import math
import torch
from torch import nn
import torch.nn.functional as F

from apis.tools.config import Configuration


class FeatureFusion(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        # 提取基本数据
        self.cfg = cfg
        # self.device = cfg.device
        self.dtype = cfg.dtype_model_torch
        self.model_params = cfg.feature_fusion_params

        self.pp_query = nn.Embedding(3, self.model_params['inp_channel'])
        self.fuse = nn.Sequential(
            nn.Conv2d(self.model_params['inp_channel'] * 2, self.model_params['mid_channel'], kernel_size=5, padding=2, bias=False, dtype=self.dtype),
            nn.BatchNorm2d(self.model_params['mid_channel'], affine=True, dtype=self.dtype),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.model_params['mid_channel'], self.model_params['oup_channel'], kernel_size=5, padding=2, bias=False, dtype=self.dtype),
            nn.BatchNorm2d(self.model_params['oup_channel'], affine=True, dtype=self.dtype),
            nn.ReLU(inplace=True)
        )

    def forward(self, bev_feature, aim_parking_plot_id):
        pp_feature = self.pp_query(aim_parking_plot_id.long())  # [B, 1, F]
        pp_feature = pp_feature.expand(-1, bev_feature.shape[1], -1)  # [B, 1, F] -> [B, N, F]

        # 拼接特征，并恢复[C, H, W]的形式
        feature = torch.cat([bev_feature, pp_feature], dim=-1)  # [B, N, F * 2]
        feature = feature.permute(0, 2, 1).reshape(  # [B, N, F * 2] -> [B, F * 2, N] -> [B, F * 2, H, W]
            feature.shape[0], feature.shape[2], 
            *self.cfg.map_bev['map_down_sample']
        )

        # 通过卷积进行融合
        fuse_feature = self.fuse(feature)  # [B, F * 2, H, W]
        fuse_feature = fuse_feature.flatten(2).permute(0, 2, 1) + bev_feature  # [B, F * 2, H, W] -> [B, H * W, F * 2] -> [B, H * W, F]

        return fuse_feature
