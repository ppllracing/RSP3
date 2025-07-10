import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .for_base import AgentBase
from ..tools.config import Configuration

class AgentAnalyseInGenerator(AgentBase):
    def __init__(self, cfg: Configuration):
        super().__init__(cfg)
        self.map_bev = None
        self.path_planning = None
        self.path_planning_local = None
        self.init_all()

    def init_all(self):
        self.map_bev = {  # 通过map_bev进行分析
            'obs': [],  # 障碍物
            'ego': [],  # 自车
            'parking': []  # 车位
        }
        self.path_planning_global = []
        self.path_planning_local = []
        self.num = 0
        super().init_all()

    def __call__(self, dataset, path_folder):
        # 重置
        self.init_all()

        os.makedirs(path_folder, exist_ok=True)

        # 遍历dataset中的所有datas
        for datas in tqdm(dataset, desc='Analyse', total=len(dataset), leave=False):
            self.path_planning_global.append(datas['global_path']['success'])
            for datas_seq in datas['sequence']:
                self.path_planning_local.append(datas_seq['path']['success'])
                self.num += 1
                self.map_bev['obs'].append(datas_seq['bev']['map_bev'][1])
                # 读取自车信息
                self.map_bev['ego'].append(datas_seq['bev']['map_bev'][2])
                # 读取车位信息
                self.map_bev['parking'].append(datas_seq['bev']['map_bev'][3])

        # 总数据量
        self.logger.info(f'Total Data: {self.num}')

        # 计算占比
        area = self.cfg.map_bev['final_dim'][0] * self.cfg.map_bev['final_dim'][1]
        for k, v in self.map_bev.items():
            v = sum(v) / area
            self.logger.info(f'{k} Ratio: {np.mean(v) * 100:.4f}%')

        # 输出成功率
        global_success_ratio = np.mean(self.path_planning_global)
        local_success_ratio = np.mean(self.path_planning_local)
        self.logger.info(f'Path Planning Success Ratio (global): {global_success_ratio * 100:.4f}%')
        self.logger.info(f'Path Planning Success Ratio (local): {local_success_ratio * 100:.4f}%')

        # 保存图片
        for k, v in self.map_bev.items():
            v = sum(v)
            fig = plt.figure()
            plt.title(k)
            plt.imshow(v, cmap='viridis')
            plt.savefig(os.path.join(path_folder, f'{k}.png'))
            plt.close(fig)
