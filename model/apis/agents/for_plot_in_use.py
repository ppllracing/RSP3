import os
import imageio
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from celluloid import Camera as CelluloidCamera
from tqdm import tqdm

from ..tools.config import Configuration
from .for_base import AgentBase

class AgentPlotInUse(AgentBase):
    def __init__(self, cfg: Configuration):
        super().__init__(cfg)
        self.init_all()
    
    def init_all(self):
        self.init_plot()
        self.logger.info('Finish to Initialize Plot')

        super().init_all()

    def init_plot(self):
        self.fig = plt.figure()
        self.celluloid_camera = CelluloidCamera(self.fig)
        gs = GridSpec(4, 3, figure=self.fig)

        self.axs_layers = [
            [self.fig.add_subplot(gs[0, i]) for i in range(3)],  # 展示Seg的结果
            self.fig.add_subplot(gs[1:3, :]),  # 展示规划结果
            [self.fig.add_subplot(gs[3, 0]), self.fig.add_subplot(gs[3, 1]), self.fig.add_subplot(gs[3, 2])],  # 展示相机数据和目标车位
            # self.fig.add_subplot(gs[4:6, :]),  # 展示loss
            # self.fig.add_subplot(gs[6:8, :]),  # 展示metric
        ]
        self.init_ax()

        # 紧凑排列
        self.fig.tight_layout()
    
    # 初始化各子图
    def init_ax(self, mode='cla'):
        assert mode in ['cla', 'clear'], 'Mode Error!'
        init_ax = lambda ax: ax.cla() if mode == 'cla' else ax.clear()
        for layer in self.axs_layers:
            if isinstance(layer, list):
                for ax in layer:
                    init_ax(ax)
            else:
                init_ax(layer)
    
    # 对模型输入输出数据进行整理和转换
    def process_datas(self, batch, oups, oups_tgt, losses, metrics):
        # 从batch中提取相机数据
        datas_camera = {
            'image_inp': batch['image'][0, 0].long().cpu().numpy(),
            'image_raw': batch['image_raw'][0, 0].long().cpu().numpy()
        }

        # 从oups和oups_tgt中提取segmentation数据
        pred = oups['segmentation_onehot'].squeeze(0)
        tgt = oups_tgt['segmentation'].squeeze(0)
        datas_seg = []
        # 给图层上色
        def layer_to_rgb(layer):
            # 将黑白图转换为rgb，绿色代表1，红色代表0，蓝色代表边界
            layer_r = F.pad(1 - layer, (1, 1, 1, 1), mode='constant', value=0)
            layer_g = F.pad(layer, (1, 1, 1, 1), mode='constant', value=0)
            layer_b = F.pad(torch.zeros_like(layer), (1, 1, 1, 1), mode='constant', value=1)
            layer_rgb = torch.stack((layer_r, layer_g, layer_b), dim=0)
            return layer_rgb
        for seg_pred, seg_tgt in zip(pred, tgt):
            seg_rgb = torch.cat([layer_to_rgb(seg_pred), layer_to_rgb(seg_tgt)], dim=2).cpu().numpy()
            datas_seg.append(seg_rgb)
        
        # 获取规划数据
        datas_path = {
            'Plan': oups['path_point'].squeeze(0).cpu().numpy()[..., :oups['effective_length'], :],
            'Ref': oups_tgt['path_point'].squeeze(0).cpu().numpy()[..., :oups_tgt['effective_length'], :] if oups_tgt['path_point'] is not None else None
        }

        # 获取目标车位数据
        datas_pp = layer_to_rgb(batch['aim_parking_plot_bev'][0, 0]).cpu().numpy()

        # 获取loss数据
        datas_loss = {
            'all': losses['all'].item(),
            'seg': losses['seg'].item(),
            'path': losses['path'].item()
        }
        
        return datas_camera, datas_seg, datas_path, datas_pp, datas_loss

    # 渲染数据
    def render_datas(self, datas_camera, datas_seg, datas_path, datas_pp, datas_loss):
        self.init_ax(mode='clear')

        # 展示Seg的结果
        for seg, ax in zip(datas_seg, self.axs_layers[0]):
            ax.imshow(seg.transpose(1, 2, 0))

        # 绘制路径
        # carla中x就是现实的y,y就是现实的x
        self.axs_layers[1].set_aspect('equal', adjustable='box')
        self.axs_layers[1].set_xlim([0, self.cfg.map_bev['final_dim'][1]])
        self.axs_layers[1].set_ylim([self.cfg.map_bev['final_dim'][0], 0])
        self.axs_layers[1].add_patch(
            patches.Circle(
                [datas_path['Plan'][0, 1], datas_path['Plan'][0, 0]],
                5.0
            )
        )
        # self.axs_layers[1].plot(datas_path['Plan'][0, 1], datas_path['Plan'][0, 0], 'bo')  # 起点
        self.axs_layers[1].plot(datas_path['Plan'][:, 1], datas_path['Plan'][:, 0], 'b--', label=f'Plan')
        self.axs_layers[1].plot(datas_path['Plan'][:, 1], datas_path['Plan'][:, 0], 'bo')
        if datas_path['Ref'] is not None:
            self.axs_layers[1].plot(datas_path['Ref'][:, 1], datas_path['Ref'][:, 0], 'r-', label=f'Ref')
            self.axs_layers[1].plot(datas_path['Ref'][:, 1], datas_path['Ref'][:, 0], 'r.')
        self.axs_layers[1].legend()

        # 展示相机视角的图
        for image, ax in zip(datas_camera.values(), self.axs_layers[2][:2]):
            ax.imshow(image.transpose(1, 2, 0))

        # 展示目标车位
        self.axs_layers[2][2].imshow(datas_pp.transpose(1, 2, 0))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        return self.fig

    def save_gif(self, path_folder, frames_path):
        frames = []
        for p in frames_path:
            frames.append(imageio.imread(p))
        imageio.mimsave(os.path.join(path_folder, 'Result.gif'), frames, duration=len(frames) / self.cfg.fps, loop=0)
        return