import os
import torch
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from tqdm import tqdm
from lightning.pytorch.utilities import move_data_to_device
from PIL import Image

from .for_base import AgentBase
from ..tools.config import Configuration
from ..models.base.lightning_base import LightningBase
from ..models.model_perception import ModelPerception
from ..models.model_pathplanning import ModelPathPlanning
from ..datas.loader import DatasetLoader, custom_collate
from ..datas.dataset_perception import Dataset_Perception
from ..datas.dataset_path_planning import Dataset_PathPlanning
from ..tools.util import get_depth_label, find_credible_interval

class AgentModelCompare(AgentBase):
    def __init__(self, ckpt_paths: dict):
        super().__init__()
        self.device = torch.device('cuda:0')
        self.dtype_torch = torch.float32
        self.models = self.load_models(ckpt_paths)
        self.cfgs = self.load_cfgs()
        self.dataset_length = None
        self.datasets = self.load_datasets()
        self.path_folder = os.path.join(self.cfg.path_results, 'compare')
        os.makedirs(self.path_folder, exist_ok=True)

        super().__init__()

    def load_models(self, ckpt_paths: dict):
        # 主模型
        paths_pd = {p.split('/')[-2]: p for p in ckpt_paths['pd']}  # 分类
        # 对比模型
        path_obu = ckpt_paths['obu']
        path_pp = ckpt_paths['pp']

        # 加载感知模型
        models_perception = {
            'rsu': ModelPerception.load_from_checkpoint(paths_pd['new'], strict=False),
            'obu': ModelPerception.load_from_checkpoint(path_obu, strict=True),
        }
        for model in models_perception.values():
            model.to(self.device)
            model.freeze()
        models_pathplanning = {f'pd_{k}': ModelPathPlanning.load_from_checkpoint(paths_pd[k], strict=True) for k in paths_pd}
        models_pathplanning.update({'pp': ModelPathPlanning.load_from_checkpoint(path_pp, strict=True)})
        for model in models_pathplanning.values():
            model.to(self.device)
            model.freeze()

        return {
            'perception': models_perception,
            'pathplanning': models_pathplanning,
        }

    def load_cfgs(self):
        # 加载感知模型的cfg
        cfgs_perception = {}
        for k, model in self.models['perception'].items():
            cfgs_perception[k] = model.cfg
        # 加载路径规划模型的cfg
        cfgs_pathplanning = {}
        for k, model in self.models['pathplanning'].items():
            cfgs_pathplanning[k] = model.cfg
        return {
            'perception': cfgs_perception,
            'pathplanning': cfgs_pathplanning,
        }

    def load_datasets(self):
        lenghts = []
        # 加载感知模型的数据集
        datasets_perception = {}
        for k, cfg in self.cfgs['perception'].items():
            datasets_perception[k] = DatasetLoader(cfg, Dataset_Perception(cfg, mode='normal'))['test'].dataset
            lenghts.append(len(datasets_perception[k]))

        # 加载路径规划模型的数据集
        cfg = self.cfgs['pathplanning']['pd_new']
        _dataset = DatasetLoader(cfg, Dataset_PathPlanning(cfg, mode='normal'))['test'].dataset
        datasets_pathplanning = {}
        for k, cfg in self.cfgs['pathplanning'].items():
            datasets_pathplanning[k] = _dataset
            lenghts.append(len(datasets_pathplanning[k]))

        assert len(set(lenghts)) == 1, f'The length of datasets is not equal, {lenghts}'
        self.dataset_length = lenghts[0]

        return {
            'perception': datasets_perception,
            'pathplanning': datasets_pathplanning,
        }

    def get_oups_of(self, type):
        # 分离模型和数据集
        models = self.models[type]
        datasets = self.datasets[type]
        # 遍历所有数据
        oups = {}
        for i in tqdm(range(self.dataset_length), desc=type):
            for k, model in models.items():
                datas_b = move_data_to_device(custom_collate([datasets[k][i]]), self.device)
                # 获得输出
                oups_b = move_data_to_device(model(**datas_b), 'cpu')

                if k in oups:
                    oups[k].append(oups_b)
                else:
                    oups[k] = [oups_b]
        return oups

    def process_seg(self, seg):
        seg = seg.astype(np.uint8).argmax(0)
        seg[seg == 1] = 128
        seg[seg == 2] = 255
        return seg

    def process_depth(self, depth):
        d_range = self.cfg.map_bev['d_range']
        down_sample_factor = self.cfg.bev_model_params['bev_down_sample']
        depth_channels = round((d_range[1] - d_range[0]) / d_range[2])
        depth = get_depth_label(
            depth[None, :, :, :],
            down_sample_factor,
            d_range,
            depth_channels
        )
        return (depth * self.cfg.map_bev['d_range'][2] + self.cfg.map_bev['d_range'][0])[0].astype(np.uint8)

    def process_depth_pred(self, depth):
        return (depth.argmax(dim=-3) * self.cfg.map_bev['d_range'][2] + self.cfg.map_bev['d_range'][0])[0].numpy().astype(np.uint8)

    def save_perception_rsu(self, i, oups, path_folder):
        inps = self.datasets['perception']['rsu'][i]
        # 获取数据集中路端感知的图像
        rsu_image = inps['rsu_image'][0].astype(np.uint8)
        rsu_depth = self.process_depth(inps['rsu_image_depth'])[0]
        # 获取数据集中场景分割的结果
        rsu_seg = self.process_seg(inps['rsu_segmentation'])

        # 获取模型输出的路端感知结果
        rsu_seg_pred = self.process_seg(oups[i]['segmentation_onehot'][0].numpy())
        rsu_depth_pred = self.process_depth_pred(oups[i]['image_depth'])[0]

        # 保存结果
        plt.imsave(os.path.join(path_folder, 'rsu_image.png'), rsu_image.transpose(1, 2, 0))
        plt.imsave(os.path.join(path_folder, 'rsu_depth.png'), rsu_depth)
        plt.imsave(os.path.join(path_folder, 'rsu_seg.png'), rsu_seg)
        plt.imsave(os.path.join(path_folder, 'rsu_seg_pred.png'), rsu_seg_pred)
        plt.imsave(os.path.join(path_folder, 'rsu_depth_pred.png'), rsu_depth_pred)

        return [
            os.path.join(path_folder, 'rsu_image.png'),
            os.path.join(path_folder, 'rsu_depth.png'),
            os.path.join(path_folder, 'rsu_seg.png'),
            os.path.join(path_folder, 'rsu_seg_pred.png'),
            os.path.join(path_folder, 'rsu_depth_pred.png')
        ]

    def save_perception_obu(self, i, oups, path_folder):
        inps = self.datasets['perception']['obu'][i]
        # 获取数据集中车端感知的图像
        obu_images = inps['obu_image'].astype(np.uint8)
        obu_depths = self.process_depth(inps['obu_image_depth'])
        # 获取数据集中场景分割的结果
        obu_seg = self.process_seg(inps['obu_segmentation'])

        # 获取模型输出的车端感知结果
        obu_seg_pred = self.process_seg(oups[i]['segmentation_onehot'][0].numpy())
        obu_depths_pred = self.process_depth_pred(oups[i]['image_depth'])

        # 保存结果
        for j, (obu_image, obu_depth, obu_depth_pred) in enumerate(zip(obu_images, obu_depths, obu_depths_pred)):
            plt.imsave(os.path.join(path_folder, f'obu_image_{j}.png'), obu_image.transpose(1, 2, 0))
            plt.imsave(os.path.join(path_folder, f'obu_depth_{j}.png'), obu_depth)
            plt.imsave(os.path.join(path_folder, f'obu_depth_pred_{j}.png'), obu_depth_pred)
        plt.imsave(os.path.join(path_folder, 'obu_seg.png'), obu_seg)
        plt.imsave(os.path.join(path_folder, 'obu_seg_pred.png'), obu_seg_pred)

        return [
            *[os.path.join(path_folder, f'obu_image_{j}.png') for j in range(len(obu_images))],
            *[os.path.join(path_folder, f'obu_depth_{j}.png') for j in range(len(obu_images))],
            *[os.path.join(path_folder, f'obu_depth_pred_{j}.png') for j in range(len(obu_images))],
            os.path.join(path_folder, 'obu_seg.png'),
            os.path.join(path_folder, 'obu_seg_pred.png')
        ]

    def save_pathplanning_pd(self, i, oups, k, path_folder):
        inps = self.datasets['pathplanning'][k][i]
        # 获取数据集中路径规划的结果
        path_point = inps['path_point'][:inps['effective_length']]
        path_point_pred = oups[i]['path_point'][0, :oups[i]['effective_length']].numpy()
        pdx, pdy = oups[i]['path_point_probability']
        pdx, pdy = pdx[0, :oups[i]['effective_length'], 2:].numpy(), pdy[0, :oups[i]['effective_length'], 2:].numpy()

        # 获取数据集中感知结果
        seg = self.process_seg(inps['rsu_segmentation'])
        seg_pred = self.process_seg(oups[i]['segmentation_onehot'][0].numpy())

        # 计算概率分布
        pdx = np.exp(pdx - pdx.max(axis=-1, keepdims=True)) / np.sum(np.exp(pdx - pdx.max(axis=-1, keepdims=True)), axis=-1, keepdims=True)
        pdy = np.exp(pdy - pdy.max(axis=-1, keepdims=True)) / np.sum(np.exp(pdy - pdy.max(axis=-1, keepdims=True)), axis=-1, keepdims=True)

        # 保存结果
        fig = plt.figure(figsize=(10, 10))
        for path_point_, seg_, name in zip([path_point, path_point_pred], [seg, seg_pred], ['pathplanning', f'pathplanning_{k}_pred']):
            plt.imshow(seg_, alpha=0.5)
            plt.plot(path_point_[:, 1], path_point_[:, 0], 'r-', linewidth=2)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(path_folder, f'{name}.png'))
            plt.clf()
        plt.close(fig)

        return [
            os.path.join(path_folder, f'pathplanning.png'),
            os.path.join(path_folder, f'pathplanning_{k}_pred.png')
        ]

    def save_pathplanning_pp(self, i, oups, k, path_folder):
        seg_pred = self.process_seg(oups[i]['segmentation_onehot'][0].numpy())
        path_point_pred = oups[i]['path_point'][0, :oups[i]['effective_length']].numpy()

        # 保存结果
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(seg_pred, alpha=0.5)
        plt.plot(path_point_pred[:, 1], path_point_pred[:, 0], 'r-', linewidth=2)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(path_folder, f'pathplanning_{k}_pred.png'))
        plt.close(fig)

        return [
            os.path.join(path_folder, f'pathplanning_{k}_pred.png')
        ]

    def save_results(self, oups_perception, oups_pathplanning):
        for i in tqdm(range(self.dataset_length), desc='Show'):
            path_folder = os.path.join(self.path_folder, str(i))
            os.makedirs(path_folder, exist_ok=True)

            path_figs_group = []
            path_figs_group.append(self.save_perception_rsu(i, oups_perception['rsu'], path_folder))
            path_figs_group.append(self.save_perception_obu(i, oups_perception['obu'], path_folder))
            for k in oups_pathplanning:
                if k.startswith('pd'):
                    path_figs_group.append(self.save_pathplanning_pd(i, oups_pathplanning[k], k, path_folder))
                else:
                    path_figs_group.append(self.save_pathplanning_pp(i, oups_pathplanning[k], k, path_folder))

            # 将所有的结果保存成
            fig = plt.figure(figsize=(10, 10))
            l = len(path_figs_group)
            for j, path_figs in enumerate(path_figs_group):
                m = len(path_figs)
                for k, path_fig in enumerate(path_figs):
                    plt.subplot(l, m, j * m + k + 1)
                    plt.imshow(Image.open(path_fig))
                    plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.path_folder, f'{i}.png'))
            plt.close(fig)

    def run(self):
        # 先获取感知数据的结果
        oups_perception = self.get_oups_of('perception')

        # 再获取路径规划数据的结果
        oups_pathplanning = self.get_oups_of('pathplanning')

        # 根据结果绘图
        self.save_results(oups_perception, oups_pathplanning)

