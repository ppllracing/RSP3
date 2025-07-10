import math
import torch
import torch.nn.functional as F
from torch import nn
from ...tools.config import Configuration

class OrientationHead(nn.Module):
    def __init__(self, cfg: Configuration):
        super(OrientationHead, self).__init__()

        # 提取基本数据
        self.cfg = cfg
        # self.device = cfg.device
        self.dtype = cfg.dtype_model_torch
        self.model_params = cfg.orientation_head_params

        self.fH, self.fW = self.cfg.map_bev['final_dim'][:2]
        self.token_num = {
            'x': self.fH + 2,  # 将【列】的每个格子映射为一个token，其中0为pad的，1为end, [1, self.token_num]为有效的token
            'y': self.fW + 2  # 将【行】的每个格子映射为一个token，其中0为pad的，1为end，[1, self.token_num]为有效的token
        }
        self.pad_value = self.cfg.pad_value_for_path_point_token

        self.token_feature = nn.ModuleDict({
            'x': nn.Embedding(self.token_num['x'], int(self.model_params['feature_dim'] / 2), dtype=self.dtype),  # [0, self.token_num['x'] - 1]
            'y': nn.Embedding(self.token_num['y'], int(self.model_params['feature_dim'] / 2), dtype=self.dtype),  # [0, self.token_num['y'] - 1]
        })

        self.ego_oritation_query = nn.Parameter(
            torch.randn(
                1,  # B
                1,
                self.model_params['feature_dim'],
                dtype=self.dtype,
                requires_grad=True
            )
        )
        self.tf_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.model_params['feature_dim'], 
                nhead=self.model_params['tf_nhead'],
                dropout=self.model_params['tf_dropout'],
                bias=False,
                batch_first=True,
                dtype=self.dtype
            ), 
            num_layers=self.model_params['tf_num_layer']
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.model_params['feature_dim'], int(self.model_params['feature_dim'] / 2), bias=False, dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(int(self.model_params['feature_dim'] / 2), 1, bias=False, dtype=self.dtype)
        )

    def forward(self, bev_feature):
        # 交叉注意力
        ego_oritation_feature = self.ego_oritation_query.repeat(bev_feature.shape[0], 1, 1)
        oritation_feature = self.tf_decoder(
            tgt=ego_oritation_feature,
            memory=bev_feature
        )
        oritation = self.mlp(oritation_feature)
        oritation = oritation.clamp(-math.pi, math.pi)

        oups = {
            'start_orientation': oritation
        }
        return oups