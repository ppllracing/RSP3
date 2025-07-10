import math
import torch
import torch.nn.functional as F
from torch import nn
from ...tools.config import Configuration

from .convolutions import UpsamplingConcat

class RiskAssessmentHead(nn.Module):
    def __init__(self, cfg: Configuration):
        super(RiskAssessmentHead, self).__init__()

        # 提取基本数据
        self.cfg = cfg
        # self.device = cfg.device
        self.dtype = cfg.dtype_model_torch
        self.model_params = cfg.risk_assessment_head_params

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

        # 定义网络层
        self.pos_embed = nn.Parameter(
            torch.randn(
                1,  # B
                self.cfg.max_num_for_path,  # path_points
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
            nn.Linear(self.model_params['feature_dim'], self.model_params['risk_discretization_dim'], bias=False, dtype=self.dtype),
            nn.ReLU(inplace=True)
        )

    def create_mask(self, tgt):
        tgt_mask = (torch.triu(torch.ones((tgt.shape[1], tgt.shape[1]), device=tgt.device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        tgt_mask = tgt_mask.to(self.dtype)
        tgt_mask = tgt_mask.bool()
        tgt_padding_mask = (tgt[..., 0] == self.pad_value)
        return tgt_mask, tgt_padding_mask

    def forward(self, bev_feature, path_points_token_feature):
        # # 从token中提取特征
        # path_point_token = path_point_token.long()
        # path_point_token_feature = torch.cat([
        #     self.token_feature['x'](path_point_token[..., 0]),
        #     self.token_feature['y'](path_point_token[..., 1])
        # ], dim=-1)
        # path_point_token_mask, path_point_token_padding_mask = self.create_mask(path_point_token)
        ppt_feature = path_points_token_feature + self.pos_embed

        # 交叉注意力
        risk_degree_feature = self.tf_decoder(
            tgt=ppt_feature,
            memory=bev_feature,
            # tgt_mask=path_point_token_mask,
            # tgt_key_padding_mask=path_point_token_padding_mask
        )
        risk_degree_feature = self.mlp(risk_degree_feature)
        with torch.no_grad():
            risk_degree = risk_degree_feature.softmax(dim=-1).argmax(dim=-1) / (risk_degree_feature.shape[-1] - 1)

        oups = {
            'risk_degree': risk_degree,
            'risk_degree_feature': risk_degree_feature
        }
        return oups