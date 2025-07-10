import torch
import torch.nn as nn
import numpy as np

from efficientnet_pytorch import EfficientNet
from .convolutions import UpsamplingConcat, DeepLabHead


class CamEncoder(nn.Module):
    def __init__(self, cfg, D):
        super(CamEncoder, self).__init__()
        # 提取基本数据
        self.cfg = cfg
        # self.device = cfg.device
        self.dtype = cfg.dtype_model_torch
        self.model_params = self.cfg.bev_model_params
        self.D = D

        assert self.model_params['backbone'].split('-')[1] in ['b4'], 'Only EfficientNet-B4 is supported for map BEV encoder'
        self.backbone = EfficientNet.from_pretrained(self.model_params['backbone']).to(self.dtype)
        self.reduction_channel = [0, 24, 32, 56, 160, 448]
        self.cut_model = [0, 2, 6, 10, 22, 31]
        self.index = np.log2(self.model_params['bev_down_sample']).astype(int) + 1  # 2^index = downsample，为了满足上采样和cat之后维度对齐，这里需要再往后取一个
        self.delete_unused_layers()

        self.feature_dlh = DeepLabHead(
            self.reduction_channel[self.index],  # 相比于LSS，这里加入了一个空洞卷积，主要是ASPP
            self.reduction_channel[self.index],
            self.model_params['mid_channel']
        ).to(self.dtype)
        self.feature_uc = UpsamplingConcat(
            self.reduction_channel[self.index] + self.reduction_channel[self.index - 1],
            self.model_params['oup_channel']
        ).to(self.dtype)
        self.depth_dlh = DeepLabHead(
            self.reduction_channel[self.index],  # 相比于LSS，这里加入了一个空洞卷积，主要是ASPP
            self.reduction_channel[self.index],
            self.model_params['mid_channel']
        ).to(self.dtype)
        self.depth_uc = UpsamplingConcat(
            self.reduction_channel[self.index] + self.reduction_channel[self.index - 1],
            self.D
        ).to(self.dtype)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if idx > self.cut_model[self.index]:
                indices_to_delete.append(idx)
        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_feature_depth(self, x):
        assert x.shape[-3:] == torch.Size([3, 256, 256]), 'Input should be in the shape of [B*N, 3, 256, 256]'

        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = list()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))  # [b*n, 3, image_crop, image_crop] -> [B*N, 48, 128, 128]
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints.append(prev_x)  # 当prev_x和x第3个维度不一样的时候，保存当前的prev_x
            elif idx == len(self.backbone._blocks) - 1:
                endpoints.append(x)  # 补充最后一个x
            prev_x = x

        # [B*N, 3, 256, 256]
        # [B*N, 24, 128, 128]
        # [B*N, 32, 64, 64]
        # [B*N, 56, 32, 32]
        # [B*N, 160, 16, 16]
        # [B*N, 448, 8, 8]

        input_1 = endpoints[-1]
        input_2 = endpoints[-2]

        feature = self.feature_uc(
            self.feature_dlh(input_1), 
            input_2
        )

        depth = self.depth_uc(
            self.depth_dlh(input_1),
            input_2
        )

        return feature, depth

    def forward(self, x):
        x = x.to(self.dtype)
        feature, depth = self.get_feature_depth(x)  # [B*N, C, dsH, dsW], [B*N, D, dsH, dsW]
        # 沿着channel方向做softmax，得到深度的概率分布
        feature = depth.softmax(dim=1).unsqueeze(1) * feature.unsqueeze(2)
        return feature, depth
