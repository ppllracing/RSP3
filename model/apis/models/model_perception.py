import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import lightning as L
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..tools.config import Configuration
from .base.lightning_base import LightningBase
from .base.bev_model import BevModel
from .base.bev_encoder import BevEncoder
from .base.segmentation_head import SegmentationHead
# from .base.orientation_head import OrientationHead
from ..tools.util import get_depth_label

class ModelPerception(LightningBase):
    def __init__(self, **kwargs):
        super(ModelPerception, self).__init__(**kwargs)

        self.is_rsu = self.cfg.is_rsu

        # 定义模型内容
        self.bev_model = BevModel(self.cfg)
        self.bev_encoder = BevEncoder(self.cfg)
        self.segmentation_head = SegmentationHead(self.cfg)
        # self.orientation_head = OrientationHead(self.cfg)

    def create_bev_feature(self, image, post_tran, post_rot, intrinsic, extrinsic):
        # 对图像数据进行编码，生成BEV特征图和深度图
        bev_feature_origin, depth = self.bev_model(  # [B, C, fH, fW], [B*N, D, dsH, dsW]
            image, 
            post_tran, 
            post_rot, 
            intrinsic, 
            extrinsic
        )

        # 将目标车位信息插入进BEV特征图中，并对BEV进行降采样和展平
        bev_feature_downsample = self.bev_encoder(  # [B, 64 * 4, 16 * 16]
            bev_feature_origin
        )

        oups = {
            'bev_feature': bev_feature_downsample,
            'image_depth': depth
        }

        return oups

    def forward_for_perception(self, **kwargs):
        # 提取数据
        if self.is_rsu:
            image = kwargs['rsu_image']
            post_tran = kwargs['rsu_post_tran']
            post_rot = kwargs['rsu_post_rot']
            intrinsic = kwargs['rsu_intrinsic']
            extrinsic = kwargs['rsu_extrinsic']
        else:
            image = kwargs['obu_image']
            post_tran = kwargs['obu_post_tran']
            post_rot = kwargs['obu_post_rot']
            intrinsic = kwargs['obu_intrinsic']
            extrinsic = kwargs['obu_extrinsic']
        oups = {}

        # 生成BEV特征和深度
        oups_of_bev_feature = self.create_bev_feature(  # [B, 64 * 4, 16 * 16], [B*N, D, dsH, dsW]
            image, post_tran, post_rot, intrinsic, extrinsic
        )
        oups = {**oups, **oups_of_bev_feature}

        # 生成语义分割（保证梯度）
        # 生成对应的onehot（无梯度）
        oups_of_segmentation_head = self.segmentation_head(oups['bev_feature'])
        oups = {**oups, **oups_of_segmentation_head}

        # # # 估计自车的朝向
        # oups_of_orientation_head = self.orientation_head(oups['bev_feature'])
        # oups = {**oups, **oups_of_orientation_head}

        return oups

    def forward(self, **kwargs):
        oups = self.forward_for_perception(**kwargs)
        return oups

    def collate_outps_tgt(self, batch):
        if self.is_rsu:
            oups_tgt = {
                'image_depth': batch['rsu_image_depth'],
                'segmentation': batch['rsu_segmentation'],
                # 'start_orientation': batch['start_orientation'],
            }
        else:
            oups_tgt = {
                'image_depth': batch['obu_image_depth'],
                'segmentation': batch['obu_segmentation']
            }
        return oups_tgt

    def show_batch_result(self, outputs, name, global_step, save_to_disk=False, show_on_tensorboard=True):
        oups, oups_tgt, _, _ = outputs

        fig = plt.figure()

        plt.subplot(3, 2, 1)
        seg = oups['segmentation_onehot'][0].argmax(dim=0).cpu().numpy()
        seg[seg == 1] = 128
        seg[seg == 2] = 255
        plt.imshow(seg)
        plt.axis('off') 
        plt.title('Segmentation Prediction')

        plt.subplot(3, 2, 2)
        seg = oups_tgt['segmentation'][0].argmax(dim=0).cpu().numpy()
        seg[seg == 1] = 128
        seg[seg == 2] = 255
        plt.imshow(seg)
        plt.axis('off') 
        plt.title('Segmentation Ground Truth')

        num_depth = oups['image_depth'].shape[1]
        d_range = self.cfg.map_bev['d_range']
        down_sample_factor = self.cfg.bev_model_params['bev_down_sample']
        depth_channels = round((d_range[1] - d_range[0]) / d_range[2])
        depth_label = get_depth_label(
            oups_tgt['image_depth'],
            down_sample_factor,
            d_range,
            depth_channels
        )
        for i in range(num_depth):
            plt.subplot(3, num_depth, i + 1 + num_depth * 1)
            depth = oups['image_depth'][0, i].argmax(dim=0).cpu().numpy()
            depth = depth * self.cfg.map_bev['d_range'][2] + self.cfg.map_bev['d_range'][0]
            plt.imshow(depth, norm=mpl.colors.Normalize())
            plt.axis('off')

            plt.subplot(3, num_depth, i + 1 + num_depth * 2)
            depth = depth_label[0, i].cpu().numpy()
            depth = depth * self.cfg.map_bev['d_range'][2] + self.cfg.map_bev['d_range'][0]
            plt.imshow(depth, norm=mpl.colors.Normalize())
            plt.axis('off')

        # plt.subplot(2, 2, 3)
        # depth = oups['image_depth'][0, 0].argmax(dim=0).cpu().numpy()
        # depth = depth * self.cfg.map_bev['d_range'][2] + self.cfg.map_bev['d_range'][0]
        # plt.imshow(depth, norm=mpl.colors.Normalize())
        # plt.axis('off') 
        # plt.title('Depth Prediction')

        # plt.subplot(2, 2, 4)
        # d_range = self.cfg.map_bev['d_range']
        # down_sample_factor = self.cfg.bev_model_params['bev_down_sample']
        # depth_channels = round((d_range[1] - d_range[0]) / d_range[2])
        # depth = get_depth_label(
        #     oups_tgt['image_depth'],
        #     down_sample_factor,
        #     d_range,
        #     depth_channels
        # )[0, 0].cpu().numpy()
        # depth = depth * self.cfg.map_bev['d_range'][2] + self.cfg.map_bev['d_range'][0]
        # plt.imshow(depth, norm=mpl.colors.Normalize())
        # plt.axis('off')
        # plt.title('Depth Ground Truth')

        fig.tight_layout()
        if save_to_disk:
            p = os.path.join(self.cfg.path_results, 'open_loop', 'perception', name)
            os.makedirs(p, exist_ok=True)
            plt.savefig(os.path.join(p, f'{global_step}.png'))
        if show_on_tensorboard:
            self.logger.experiment.add_figure(f'{name}/Perception', fig, global_step=global_step)
        plt.close(fig)