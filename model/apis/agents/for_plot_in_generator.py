import os
import imageio
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from celluloid import Camera as CelluloidCamera
from tqdm import tqdm

from ..tools.config import Configuration
from ..tools.util import plot_standard_rectangle, image_depth_pixel_to_meters
from .for_base import AgentBase

class AgentPlotInGenerator(AgentBase):
    def __init__(self, cfg: Configuration, bev_range):
        super().__init__(cfg)
        self.init_all()
        self.bev_range = bev_range
    
    def init_all(self):
        self.init_plot()
        self.logger.info('Finish to Initialize Plot')

        super().init_all()

    def init_plot(self):
        self.fig = plt.figure(figsize=(8, 12))
        gs = GridSpec(6, 4, figure=self.fig)

        self.axs_layers = [
            [self.fig.add_subplot(gs[0, i]) for i in range(4)],  # 展示BEV的感知数据和目标车位
            self.fig.add_subplot(gs[1:3, :2]),  # 展示车辆路径
            [self.fig.add_subplot(gs[1, 2:]), self.fig.add_subplot(gs[2, 2:])],  # 展示危险值和启发图
            [self.fig.add_subplot(gs[3, i]) for i in range(4)],  # 展示路端相机视角效果
            [self.fig.add_subplot(gs[4, i]) for i in range(4)],  # 展示车端相机视角效果
            [self.fig.add_subplot(gs[5, i]) for i in range(4)],  # 展示车端相机视角深度效果
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

    def render_datas(self, datas_camera=None, datas_bev=None, datas_aim=None, datas_vehicle=None, datas_path=None, datas_parking_plot=None):
        fig = self.fig
        self.init_ax(mode='clear')

        if datas_bev is not None:
            axs = self.axs_layers[0]
            # 展示路端感知和车位表示
            axs[0].imshow(datas_bev['map_bev'][:3].transpose(1, 2, 0))
            axs[0].set_title('RSU')
            axs[1].imshow(datas_bev['map_bev'][3])
            axs[1].set_title('Parking Plot')
            axs[2].imshow(datas_bev['map_bev_obu'][:3].transpose(1, 2, 0))
            axs[2].set_title('OBU')
            axs[3].imshow(datas_bev['map_bev_obu'][3])
            axs[3].set_title('Parking Plot')

        if datas_path is not None:
            ax = self.axs_layers[1]
            # 绘制map_bev的范围
            # map_bev的四个角点
            vertices = [
                [self.bev_range[0, 1], self.bev_range[1, 0], 0],
                [self.bev_range[0, 1], self.bev_range[1, 1], 0],
                [self.bev_range[0, 0], self.bev_range[1, 1], 0],
                [self.bev_range[0, 0], self.bev_range[1, 0], 0]
            ]
            plot_standard_rectangle(vertices, ax, 'r-')
            ax.set_aspect('equal', adjustable='box')

            if datas_path['success']:
                # 绘制路径
                # carla中x就是现实的y,y就是现实的x
                ax.plot(datas_path['path_points_rear'][:, 3], datas_path['path_points_rear'][:, 2], 'r.')
                ax.plot(datas_path['path_points_rear'][:, 3], datas_path['path_points_rear'][:, 2], 'r-')

        if datas_parking_plot is not None:
            ax = self.axs_layers[1]
            # 车位边界线
            plot_standard_rectangle(datas_parking_plot['vertices_up'], ax, 'r-')

            # 车位中心线
            center = datas_parking_plot['center_up']
            ax.plot(center[1], center[0], 'r*')
            ax.plot([center[1] - 1, center[1] + 1], [center[0], center[0]], 'r--')
            ax.plot([center[1], center[1]], [center[0] - 1, center[0] + 1], 'r--')

        if not datas_vehicle is None:
            ax = self.axs_layers[1]
            # 展示车辆轨迹
            # carla中x就是现实的y,y就是现实的x
            xyzPYR_rear = datas_vehicle['xyzPYR_rear']
            vertices = datas_vehicle['vertices_up']
            ax.plot(xyzPYR_rear[1], xyzPYR_rear[0], '*')
            plot_standard_rectangle(vertices, ax, 'b-', inner_point=xyzPYR_rear[0:2])

        if datas_path is not None:
            ax1, ax2 = self.axs_layers[2]

            # 危险值
            risk_degrees = datas_path['risk_degrees']
            if risk_degrees is not None:
                ax1.plot(risk_degrees * 100, 'r-')
                ax1.set_title('Risk Degrees')
                ax1.set_ylabel('%')
                ax1.set_ylim(0, 100)

            # 启发图，转换为RGB形式的
            h = datas_path['heuristic_fig']
            heuristic_fig = np.stack([np.zeros_like(h), np.zeros_like(h), h], axis=-1)  # [H, W, 3]
            # 加入点表示起点和终点
            path_point_start = datas_aim['start_id']
            path_point_end = datas_aim['end_id']
            for l, m in list(itertools.product(range(self.cfg.map_bev['final_dim'][0]), range(self.cfg.map_bev['final_dim'][1]))):
                if np.linalg.norm(np.array([l, m]) - path_point_start[:2]) < 3:
                    heuristic_fig[l, m, 0] = 1
                if np.linalg.norm(np.array([l, m]) - path_point_end[:2]) < 3:
                    heuristic_fig[l, m, 1] = 1
            ax2.imshow(heuristic_fig)
            ax2.set_title('Heuristic Fig')

        if not datas_camera is None:
            axs = self.axs_layers[3]
            # 展示路端相机视角的图
            image_npc = datas_camera['npc']['image'].transpose(1, 2, 0)
            image_rsu_rgb = datas_camera['rsu_rgb']['image'].transpose(1, 2, 0)
            image_rsu_rgb_crop = datas_camera['rsu_rgb']['image_crop'].transpose(1, 2, 0)
            image_rsu_depth_crop = image_depth_pixel_to_meters(datas_camera['rsu_depth']['image_crop'].transpose(1, 2, 0))
            axs[0].imshow(image_npc)
            axs[0].set_title('npc')
            axs[1].imshow(image_rsu_rgb)
            axs[1].set_title('rsu_rgb')
            axs[2].imshow(image_rsu_rgb_crop)
            axs[2].set_title('rsu_rgb_crop')
            axs[3].imshow(image_rsu_depth_crop, norm=mpl.colors.Normalize())
            axs[3].set_title('rsu_depth_crop')

            axs = self.axs_layers[4]
            # 展示车端相机视角的图
            img_front_rgb = datas_camera['obu_front_rgb']['image'].transpose(1, 2, 0)
            img_left_rgb = datas_camera['obu_left_rgb']['image'].transpose(1, 2, 0)
            img_right_rgb = datas_camera['obu_right_rgb']['image'].transpose(1, 2, 0)
            img_rear_rgb = datas_camera['obu_rear_rgb']['image'].transpose(1, 2, 0)
            axs[0].imshow(img_front_rgb)
            axs[0].set_title('front_rgb')
            axs[1].imshow(img_left_rgb)
            axs[1].set_title('left_rgb')
            axs[2].imshow(img_right_rgb)
            axs[2].set_title('right_rgb')
            axs[3].imshow(img_rear_rgb)
            axs[3].set_title('rear_rgb')

            axs = self.axs_layers[5]
            # 展示车端相机视角深度图
            img_front_depth = image_depth_pixel_to_meters(datas_camera['obu_front_depth']['image'].transpose(1, 2, 0))
            img_left_depth = image_depth_pixel_to_meters(datas_camera['obu_left_depth']['image'].transpose(1, 2, 0))
            img_right_depth = image_depth_pixel_to_meters(datas_camera['obu_right_depth']['image'].transpose(1, 2, 0))
            img_rear_depth = image_depth_pixel_to_meters(datas_camera['obu_rear_depth']['image'].transpose(1, 2, 0))
            axs[0].imshow(img_front_depth, norm=mpl.colors.Normalize())
            axs[0].set_title('front_depth')
            axs[1].imshow(img_left_depth, norm=mpl.colors.Normalize())
            axs[1].set_title('left_depth')
            axs[2].imshow(img_right_depth, norm=mpl.colors.Normalize())
            axs[2].set_title('right_depth')
            axs[3].imshow(img_rear_depth, norm=mpl.colors.Normalize())
            axs[3].set_title('rear_depth')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_gif(self, path_datas_folder, frames_path):
        frames = []
        for p in frames_path:
            frames.append(imageio.imread(p))
        imageio.mimsave(os.path.join(path_datas_folder, 'Data.gif'), frames, duration=len(frames) / self.cfg.fps, loop=0)
        return
