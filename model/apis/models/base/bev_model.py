import torch

from torch import nn
from .cam_encoder import CamEncoder
from ...tools.config import Configuration
from ...tools.geometry import VoxelsSumming, calculate_birds_eye_view_parameters


class BevModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        # 提取基本数据
        self.cfg = cfg
        # self.device = cfg.device
        self.dtype = cfg.dtype_model_torch
        self.model_params = self.cfg.bev_model_params

        bev_res = torch.tensor([*self.cfg.map_bev['resolution'][:2], self.cfg.map_bev['z_range_local'][1] - self.cfg.map_bev['z_range_local'][0]], dtype=self.dtype)
        bev_start_pos = torch.tensor([self.cfg.map_bev['x_range_local'][0], self.cfg.map_bev['y_range_local'][0], self.cfg.map_bev['z_range_local'][0]], dtype=self.dtype)
        bev_dim = torch.tensor([*self.cfg.map_bev['final_dim'][:2], 1.0]).to(torch.int)

        frustum = self.create_frustum()  # 视锥
        self.bev_res = nn.Parameter(bev_res, requires_grad=False)
        self.bev_start_pos = nn.Parameter(bev_start_pos, requires_grad=False)
        self.bev_dim = nn.Parameter(bev_dim, requires_grad=False)
        self.frustum = nn.Parameter(frustum, requires_grad=False)

        self.depth_channel, _, _, _ = self.frustum.shape
        self.cam_encoder = CamEncoder(self.cfg, self.depth_channel)

    def create_frustum(self):
        h, w = self.cfg.collect['image_crop']
        down_sample_h, down_sample_w = h // self.model_params['bev_down_sample'], w // self.model_params['bev_down_sample']

        depth_grid = torch.arange(*self.cfg.map_bev['d_range'], dtype=self.dtype)  # 在垂直相机平面方向上取离散的点
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, down_sample_h, down_sample_w)  # 根据每个点都复制成维度[down_sample_h, down_sample_w]
        depth_slice = depth_grid.shape[0]

        x_grid = torch.linspace(0, w - 1, down_sample_w, dtype=self.dtype)
        x_grid = x_grid.view(1, 1, down_sample_w).expand(depth_slice, down_sample_h, down_sample_w)
        y_grid = torch.linspace(0, h - 1, down_sample_h, dtype=self.dtype)
        y_grid = y_grid.view(1, down_sample_h, 1).expand(depth_slice, down_sample_h, down_sample_w)

        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)  # 形成视锥

        return frustum

    def get_geometry(self, post_trans, post_rots, intrinsics, extrinsics):
        extrinsics = torch.inverse(extrinsics)  # 对外参矩阵求逆
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]  # 从逆矩阵中提取旋转矩阵和平移矩阵
        B, N, _ = translation.shape  # 获取batch和相机数量

        # 消除crop对视锥的影响（这么理解也许是对的）
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],  # x_grid和y_grid乘上depth_grid
                points[:, :, :, :, :, 2:3]
            ), dim=5
        )
        combine_transform = rotation.matmul(torch.inverse(intrinsics))  # 对内参矩阵求逆并乘上旋转矩阵
        points = combine_transform.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)  # 将combine_transform在中间维度上进行拓展，使得维度的数量与points对齐，
        points += translation.view(B, N, 1, 1, 1, 3)

        return points

    def encoder_forward(self, images):
        B, N, C, imH, imW = images.shape
        images = images.view(B*N, C, imH, imW)  # 把所有图像排在一起
        feature, depth = self.cam_encoder(images)  # 获取对特征和深度的编码

        feature = feature.view(B, N, *feature.shape[1:])  # 把图像解开，[B, N, C, D, dsH, dsW]
        feature = feature.permute(0, 1, 3, 4, 5, 2)  # 维度变换，[B, N, D, dsH, dsW, C]
        depth = depth.view(B, N, *depth.shape[1:])  # [B, N, D, dsH, dsW]
        return feature, depth

    def proj_bev_feature(self, geom, image_feature):
        # geom: [B, N, D, dsH, dsW, 3]
        # image_feature: [B, N, D, dsH, dsW, C]
        
        B, N, D, dsH, dsW, C = image_feature.shape
        output = torch.zeros((B, C, self.bev_dim[0], self.bev_dim[1]),
                             dtype=self.dtype, device=image_feature.device)
        Nprime = N * D * dsH * dsW
        for b in range(B):
            image_feature_b = image_feature[b]  # [N, D, dsH * dsW, C]
            geom_b = geom[b]  # [N, D, dsH * dsW, 3]

            x_b = image_feature_b.reshape(Nprime, C)

            geom_b = ((geom_b - (self.bev_start_pos - self.bev_res / 2.0)) / self.bev_res)
            geom_b = geom_b.view(Nprime, 3).long()

            mask = ((geom_b[:, 0] >= 0) & (geom_b[:, 0] < self.bev_dim[0])
                    & (geom_b[:, 1] >= 0) & (geom_b[:, 1] < self.bev_dim[1])
                    & (geom_b[:, 2] >= 0) & (geom_b[:, 2] < self.bev_dim[2]))
            x_b = x_b[mask]
            geom_b = geom_b[mask]

            ranks = ((geom_b[:, 0] * (self.bev_dim[1] * self.bev_dim[2])
                     + geom_b[:, 1] * self.bev_dim[2]) + geom_b[:, 2])
            sorts = ranks.argsort()
            x_b, geom_b, ranks = x_b[sorts], geom_b[sorts], ranks[sorts]

            x_b, geom_b = VoxelsSumming.apply(x_b, geom_b, ranks)

            bev_feature = torch.zeros(
                (self.bev_dim[2], self.bev_dim[0], self.bev_dim[1], C),
                dtype=self.dtype, device=image_feature_b.device
            )
            bev_feature[geom_b[:, 2], geom_b[:, 0], geom_b[:, 1]] = x_b
            tmp_bev_feature = bev_feature.permute((0, 3, 1, 2)).squeeze(0)
            output[b] = tmp_bev_feature

        return output

    def forward(self, image, post_tran, post_rot, intrinsic, extrinsic):
        # image = image.to(self.device)
        # intrinsic = intrinsic.to(self.device)
        # extrinsic = extrinsic.to(self.device)

        geom = self.get_geometry(post_tran, post_rot, intrinsic, extrinsic)  # [B, N, D, dsH, dsW, 3]
        feature, depth = self.encoder_forward(image)  # [B, N, D, dsH, dsW, c], [B*N, D, dsH, dsW]
        bev_feature = self.proj_bev_feature(geom, feature)  # [B, C, fH, fW]
        
        return bev_feature, depth

