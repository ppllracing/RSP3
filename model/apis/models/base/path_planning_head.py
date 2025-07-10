import math
import torch
import torch.nn.functional as F
from torch import nn

from ...tools.config import Configuration
from ...tools.util import get_xy_id_from_token, get_effective_length_from_path_point_token

class PathPlanningHead(nn.Module):
    def __init__(self, cfg: Configuration):
        super(PathPlanningHead, self).__init__()
        # 提取基本数据
        self.cfg = cfg
        # self.device = cfg.device
        self.dtype = cfg.dtype_model_torch
        self.model_params = cfg.path_planning_head_params

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
        # self.pos_embed_pp = nn.Parameter(
        #     torch.randn(
        #         1,  # B
        #         self.cfg.max_num_for_path,  # path_points
        #         self.model_params['feature_dim'],
        #         dtype=self.dtype,
        #         requires_grad=True
        #     )
        # )
        self.register_buffer('pos_embed_pp', self._build_sincos_pos_embed_pp())
        self.register_buffer('pos_embed_bev', self._build_sincos_pos_embed_bev())
        self.cros_attn = nn.TransformerDecoder(
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
        self.ln_embed = nn.LayerNorm(self.model_params['feature_dim'], dtype=self.dtype)
        # 预计算并缓存future mask
        self._build_future_mask(self.cfg.max_num_for_path)

        # 多层感知机作为解码器
        mlp_common = nn.Sequential(
            nn.Linear(self.model_params['feature_dim'], self.model_params['feature_dim'], dtype=self.dtype),
            nn.LayerNorm(self.model_params['feature_dim']),
            nn.ReLU(inplace=True),
        )
        if self.model_params['method'] == 'distribution':
            self.mlps = nn.ModuleDict({
                'path_point_decode_x': nn.Sequential(
                    nn.Linear(self.model_params['feature_dim'], self.model_params['feature_dim'], dtype=self.dtype),
                    nn.LayerNorm(self.model_params['feature_dim']),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.model_params['feature_dim'], self.token_num['x'], dtype=self.dtype)
                ),
                'path_point_decode_y': nn.Sequential(
                    nn.Linear(self.model_params['feature_dim'], self.model_params['feature_dim'], dtype=self.dtype),
                    nn.LayerNorm(self.model_params['feature_dim']),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.model_params['feature_dim'], self.token_num['y'], dtype=self.dtype)
                )
            })
        elif self.model_params['method'] == 'pathpoint':
            self.mlps = nn.ModuleDict({
                'path_point_decode_x': nn.Sequential(
                    mlp_common,
                    nn.Linear(self.model_params['feature_dim'], 1, dtype=self.dtype),
                    nn.Sigmoid()
                ),
                'path_point_decode_y': nn.Sequential(
                    mlp_common,
                    nn.Linear(self.model_params['feature_dim'], 1, dtype=self.dtype),
                    nn.Sigmoid()
                )
            })
        else:
            raise NotImplementedError('method not implemented')

        # 特征缓存初始化
        self.feature_cache = None

    def _build_future_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        self.register_buffer('future_mask', mask, persistent=False)

    def _init_feature_cache(self):
        self.feature_cache = None

    def _build_sincos_pos_embed_pp(self):
        """
        Returns: [1, max_num_for_path, dim] sine-cosine positional embedding
        """
        max_num_for_path = self.cfg.max_num_for_path
        d_model = self.model_params['feature_dim']

        # 确保维度是偶数
        assert d_model % 2 == 0, "Feature dimension must be even for sinusoidal positional encoding"

        # 创建时间步索引
        positions = torch.arange(max_num_for_path, dtype=self.dtype)

        # 计算位置编码
        pos_embed = self.positional_encoding_1d(d_model, positions, self.dtype)
        return pos_embed.unsqueeze(0)  # [1, max_num_for_path, d_model]

    def positional_encoding_1d(self, d_model, positions, dtype):
        """
        为1D位置生成正弦-余弦位置编码
        参数:
            d_model: 输出维度
            step_idx: 时间步索引张量 (任意形状)
            dtype: 数据类型
        返回:
            位置编码 [max_num_for_path, d_model]
        """
        # 预计算频率向量 (避免重复计算)
        half_dim = d_model // 2
        exponents = torch.arange(half_dim, dtype=dtype) / half_dim
        inv_freq = 1.0 / (10000 ** exponents)  # [d_model//2]
        
        # 计算位置-频率矩阵 [L, d_model//2]
        pos_angles = positions[:, None] * inv_freq[None, :]  # [L, d_model//2]
        
        # 创建位置编码矩阵
        pos_embed = torch.zeros((positions.shape[0], d_model), dtype=dtype)
        
        # 交替填充sin和cos值
        pos_embed[:, 0::2] = torch.sin(pos_angles)  # 偶数索引: sin
        pos_embed[:, 1::2] = torch.cos(pos_angles)  # 奇数索引: cos
        return pos_embed  # [max_num_for_path, d_model]

    def _build_sincos_pos_embed_bev(self):
        """
        Returns: [1, H*W, dim] sine-cosine positional embedding
        """
        H, W = self.cfg.map_bev['map_down_sample']
        d_model = self.model_params['feature_dim']
        
        # 确保维度是偶数
        assert d_model % 2 == 0, "Feature dimension must be even for sinusoidal positional encoding"
        
        # 创建网格坐标
        grid_y = torch.arange(H, dtype=self.dtype)
        grid_x = torch.arange(W, dtype=self.dtype)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')  # [H, W]
        
        # 计算位置编码
        pos_embed = self.positional_encoding_2d(d_model, grid_y, grid_x, self.dtype)
        return pos_embed.unsqueeze(0)  # [1, H*W, d_model]

    def positional_encoding_2d(self, d_model, y, x, dtype):
        """
        为2D位置生成正弦-余弦位置编码
        参数:
            d_model: 输出维度
            y: y坐标张量 (任意形状)
            x: x坐标张量 (与y相同形状)
            dtype: 数据类型
        返回:
            位置编码 [H*W, d_model]
        """
        # 展平坐标
        y = y.flatten()  # [H*W]
        x = x.flatten()  # [H*W]
        
        # 计算通道索引
        half_dim = d_model // 2
        dim_t = torch.arange(half_dim, dtype=dtype)  # [d_model//2]
        
        # 标准Transformer频率计算
        inv_freq = 1.0 / (10000 ** (2 * dim_t / half_dim))
        
        # 计算x和y的位置编码 (使用不同的频率分配)
        enc_x = x[:, None] * inv_freq[None, :]  # [H*W, d_model//2]
        enc_y = y[:, None] * inv_freq[None, :]  # [H*W, d_model//2]
        
        # 组合位置编码 (交替通道)
        pos_embed = torch.zeros((x.shape[0], d_model), dtype=dtype)
        pos_embed[:, 0::4] = torch.sin(enc_x[:, :half_dim//2])  # x的sin(前半)
        pos_embed[:, 1::4] = torch.cos(enc_x[:, :half_dim//2])  # x的cos(前半)
        pos_embed[:, 2::4] = torch.sin(enc_y[:, half_dim//2:])  # y的sin(后半)
        pos_embed[:, 3::4] = torch.cos(enc_y[:, half_dim//2:])  # y的cos(后半)
        
        return pos_embed  # [H*W, d_model]

    def create_mask(self, tgt):
        tgt_padding_mask = (tgt[..., 0] == self.pad_value)
        return self.future_mask[:tgt.size(1), :tgt.size(1)], tgt_padding_mask

    def get_path_points_token_feature(self, path_points_token):
        path_points_token_ = path_points_token.round().long()
        path_points_token_feature = torch.cat([
            self.token_feature['x'](path_points_token_[..., 0]),
            self.token_feature['y'](path_points_token_[..., 1])
        ], dim=-1)
        return path_points_token_feature

    def attention(self, bev_feature, tgt):
        assert (tgt[..., 0] < self.token_num['x']).all() and (tgt[..., 1] < self.token_num['y']).all(), 'tgt out of range'

        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        # 从token中提取特征
        tgt_feature = self.get_path_points_token_feature(tgt.long()) + self.pos_embed_pp[:, :tgt.size(1), :]
        bev_feature = bev_feature + self.pos_embed_bev

        # 交叉注意力
        path_point_feature = self.cros_attn(
            tgt=self.ln_embed(tgt_feature),
            memory=self.ln_embed(bev_feature),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        return path_point_feature

    def decode(self, path_point_feature):
        if self.model_params['method'] == 'distribution':
            # 从特征中降维生成概率分布
            path_point_decode = [
                self.mlps['path_point_decode_x'](path_point_feature),
                self.mlps['path_point_decode_y'](path_point_feature)
            ]
            # 生成token
            with torch.no_grad():
                path_point_token = torch.stack([
                    path_point_prob.softmax(dim=-1).argmax(dim=-1) for path_point_prob in path_point_decode
                ], dim=-1)

            # 整理输出
            oups = {
                'path_point_token': path_point_token,
                'path_point_probability': path_point_decode,
                'path_point_feature': path_point_feature
            }
        elif self.model_params['method'] == 'pathpoint':
            # 直接生成路径点
            path_point_decode = [
                self.mlps['path_point_decode_x'](path_point_feature) * (self.token_num['x'] - 1),
                self.mlps['path_point_decode_y'](path_point_feature) * (self.token_num['y'] - 1)
            ]
            oups = {
                'path_point_token': torch.cat(path_point_decode, dim=-1),
                'path_point_feature': path_point_feature
            }

        return oups

    def forward_for_train(self, bev_feature, tgt):
        feature = self.attention(bev_feature, tgt)
        oups = self.decode(feature)
        return oups

    def forward_for_test(self, bev_feature, path_point_token_start):
        # 初始化缓存
        self._init_feature_cache()

        # 预处理
        bev_feature_with_pos = bev_feature + self.pos_embed_bev

        if self.model_params['method'] == 'distribution':
            path_point_probability = [[], []]
        path_point_feature = []
        path_point_token = []
        for i in range(self.cfg.max_num_for_path):
            # 只处理当前时间步的点
            if i == 0:
                current_token = path_point_token_start
            else:
                current_token = path_point_token_next
            
            # 仅计算当前点的特征
            current_feature = self.get_path_points_token_feature(current_token) + self.pos_embed_pp[:, i:i+1, :]

            # 更新缓存
            self.feature_cache = torch.cat([
                self.feature_cache, 
                current_feature
            ], dim=1) if self.feature_cache is not None else current_feature

            # 仅通过解码器获取当前输出
            path_point_feature_current = self.cros_attn(
                tgt=self.ln_embed(self.feature_cache),
                memory=self.ln_embed(bev_feature_with_pos),
                tgt_mask=self.future_mask[:self.feature_cache.size(1), :self.feature_cache.size(1)],
                tgt_key_padding_mask=None
            )[:, -1:, :]  # 只取最后时间步

            # 获取解码后的输出
            oups = self.decode(path_point_feature_current)
            path_point_token_next = oups['path_point_token']
            path_point_feature_next = oups['path_point_feature']
            if self.model_params['method'] == 'distribution':
                path_point_probability_next = oups['path_point_probability']

            # 记录
            path_point_token.append(path_point_token_next)
            if self.model_params['method'] == 'distribution':
                path_point_probability[0].append(path_point_probability_next[0])
                path_point_probability[1].append(path_point_probability_next[1])
            path_point_feature.append(path_point_feature_next)

        # 数据规整
        path_point_token = torch.cat(path_point_token, dim=1)
        if self.model_params['method'] == 'distribution':
            path_point_probability = [torch.cat(f, dim=1) for f in path_point_probability]
        path_point_feature = torch.cat(path_point_feature, dim=1)

        oups = {
            'path_point_token': path_point_token,
            'path_point_feature': path_point_feature
        }
        if self.model_params['method'] == 'distribution':
            oups['path_point_probability'] = path_point_probability

        return oups

    def forward(self, bev_feature, planning_input):
        if self.training:
            oups = self.forward_for_train(bev_feature, planning_input)
        else:
            oups = self.forward_for_test(bev_feature, planning_input)

        # 从token生成有效长度和xy的坐标
        with torch.no_grad():
            path_point_token = oups['path_point_token']
            path_point = torch.stack([
                get_xy_id_from_token(path_point_token_b, flag_cat=True) for path_point_token_b in path_point_token
            ], dim=0)
            effective_length = get_effective_length_from_path_point_token(path_point_token.round(), self.cfg.end_value_for_path_point_token, self.cfg.pad_value_for_path_point_token)
            start_point = path_point[:, 0:1]
            end_point = torch.stack([
                p[[[max(l-1, 0)]]] for p, l in zip(path_point, effective_length)
            ], dim=0)

        # 整理输出
        oups = {
            **oups,
            'path_point': path_point,
            'effective_length': effective_length,
            'start_point': start_point,
            'end_point': end_point
        }
        return oups