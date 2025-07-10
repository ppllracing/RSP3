import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import lightning as L
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..tools.util import get_position_id_from_bev, generate_token_from_xy
from .model_perception import ModelPerception
from .base.feature_fusion import FeatureFusion
from .base.path_planning_head import PathPlanningHead
# from .base.heuristic_head import HeuristicHead
# from .base.risk_assessment_head import RiskAssessmentHead

class ModelPathPlanning(ModelPerception):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.feature_fusion = FeatureFusion(self.cfg)
        self.path_planning_head = PathPlanningHead(self.cfg)

        # assert self.cfg.use_heuristic is not None, 'use_heuristic must be set in cfg'
        # assert self.cfg.use_risk_assessment is not None, 'use_risk_assessment must be set in cfg'
        # if self.cfg.use_heuristic:
        #     self.heuristic_head = HeuristicHead(self.cfg)
        # if self.cfg.use_risk_assessment:
        #     self.risk_assessment_head = RiskAssessmentHead(self.cfg)

    def configure_optimizers(self):
        if self.cfg.train['lr_pre_trained'] is None:
            # 说明当前没有加载预训练模型
            optimizer = self.get_optimizer(
                [self.parameters()],
                [self.cfg.train['lr']],
                ['main'],
                [1e-4]
            )
        else:
            # 获取父类（ModelPerception）的参数名集合
            perception_param_names = set(name for name, _ in super().named_parameters())

            # 分组
            perception_params = []
            pathplanning_params = []

            for name, param in self.named_parameters():
                if name in perception_param_names:
                    perception_params.append(param)
                else:
                    pathplanning_params.append(param)

            # 构建优化器和学习率调度器
            optimizer = self.get_optimizer(
                [pathplanning_params, perception_params],
                [self.cfg.train['lr'], self.cfg.train['lr_pre_trained']],
                ['main', 'pre_trained'],
                [1e-4, 1e-4]
            )

        scheduler = self.get_scheduler(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

    def forward_for_path_planning(self, **kwargs):
        # 提取数据
        bev_feature = kwargs['bev_feature']
        aim_parking_plot_id = kwargs['aim_parking_plot_id']
        oups = {}

        # 将目标车位的id和bev_feature融合
        fused_feature = self.feature_fusion(bev_feature, aim_parking_plot_id)  # [B, N+1, F]

        # 基于fused_feature对轨迹进行规划
        if self.training:
            # 训练阶段
            # planning_start_token = kwargs['start_point_center_token']  # 数据集中自车的位置
            planning_start_token = kwargs['planning_start_token']
            planning_input = torch.cat([  # [B, max_num, 2]
                planning_start_token,  # [B, 1, 2]
                kwargs['path_point_token'][:, :-1]  # [B, max_num - 1, 2]，不需要关注max_num之后的轨迹点，因为第max_num项一般是pad，抑或是end
            ], dim=1)
        else:
            # 测试阶段
            planning_input = kwargs['planning_start_token']
            # # 从segmentation_onehot中获取ego的位置
            # ego_bev = kwargs['segmentation_onehot'][:, 2].to(self.dtype_torch)
            # epo_position = [get_position_id_from_bev(ego_bev_b) for ego_bev_b in ego_bev]
            # # 将-1替换为地图中心位置
            # for i in range(len(epo_position)):
            #     if epo_position[i][0, 0] == -1:
            #         epo_position[i][0, 0] = self.cfg.map_bev['final_dim'][0] // 2
            #         epo_position[i][0, 1] = self.cfg.map_bev['final_dim'][1] // 2
            # # 将位置信息转换为token
            # planning_input = torch.stack([
            #     generate_token_from_xy(epo_position_b, flag_cat=True) for epo_position_b in epo_position
            # ], dim=0)
        oups_of_path_planning_head = self.path_planning_head(
            fused_feature,
            planning_input
        )
        oups = {**oups, **oups_of_path_planning_head}
        
        # # 对每个路点进行风险评估
        # if self.cfg.use_risk_assessment:
        #     path_points_token = kwargs['path_point_token'].long() if self.training else oups['path_point_token'].long()  # [B, max_num, 2]
        #     path_points_token_feature = self.path_planning_head.get_path_points_token_feature(path_points_token)
        #     oups_of_risk_assessment_head = self.risk_assessment_head(
        #         bev_feature,  # [B, N, 256]
        #         path_points_token_feature
        #     )
        #     oups = {**oups, **oups_of_risk_assessment_head}

        return oups

    def forward(self, oups_of_perception=None, **kwargs):
        if oups_of_perception is None:
            if self.cfg.train['lr_pre_trained'] == 0.0:
                with torch.no_grad():
                    oups_of_perception = self.forward_for_perception(**kwargs)
            else:
                oups_of_perception = self.forward_for_perception(**kwargs)
        kwargs = {**kwargs, **oups_of_perception}  # 主要是为了从perception中获取bev_feature
        oups_of_path_planning = self.forward_for_path_planning(**kwargs)
        oups = {**oups_of_perception, **oups_of_path_planning}
        return oups

    def collate_outps_tgt(self, batch):
        oups_tgt_of_segmentation = super().collate_outps_tgt(batch)
        oups_tgt_of_path_planning = {
            'path_point': batch['path_point'],
            'path_point_token': batch['path_point_token'],
            'effective_length': batch['effective_length'],
            'start_point': batch['start_point'],
            'aim_point': batch['aim_point'],
            'start_xyt': batch['start_xyt'],
            'end_xyt': batch['end_xyt']
        }
        # if self.cfg.use_heuristic:
        #     oups_tgt_of_path_planning['heuristic_fig'] = batch['heuristic_fig']
        # if self.cfg.use_risk_assessment:
        #     oups_tgt_of_path_planning['risk_degree'] = batch['risk_degree']
        oups_tgt = {**oups_tgt_of_segmentation, **oups_tgt_of_path_planning}
        # oups_tgt = oups_tgt_of_path_planning  # 不考虑segmentation的输出
        return oups_tgt
    
    def show_batch_result(self, outputs, name, global_step, save_to_disk=False, show_on_tensorboard=True):
        super().show_batch_result(outputs, name, global_step, save_to_disk, show_on_tensorboard)

        oups, oups_tgt, _, _ = outputs

        # 根据batch_id获取数据
        oups_ = {
            'path_point': oups['path_point'].squeeze(0),
            'effective_length': oups['effective_length'].squeeze(0),
        }
        if 'path_point_probability' in oups:
            oups_['path_point_probability'] = [f.squeeze(0) for f in oups['path_point_probability']]
        oups_tgt_ = {
            'path_point': oups_tgt['path_point'].squeeze(0),
            'effective_length': oups_tgt['effective_length'].squeeze(0)
        }
        # if self.cfg.use_heuristic:
        #     oups_['heuristic_fig'] = oups['heuristic_fig'].squeeze(0)
        #     oups_tgt_['heuristic_fig'] = oups_tgt['heuristic_fig'].squeeze(0)
        # if self.cfg.use_risk_assessment:
        #     oups_['risk_degree'] = oups['risk_degree'].squeeze(0)
        #     oups_tgt_['risk_degree'] = oups_tgt['risk_degree'].squeeze(0)

        fig = plt.figure()
        self.plot_path_point(oups_, oups_tgt_, [111])
        # if self.cfg.use_heuristic and not self.cfg.use_risk_assessment:
        #     # 绘制路径规划结果
        #     self.plot_path_point(oups_, oups_tgt_, [121])
        #     # 绘制启发图结果
        #     self.plot_heuristic_fig(oups_, oups_tgt_, [222, 224])
        # elif self.cfg.use_risk_assessment and not self.cfg.use_heuristic:
        #     # 绘制路径规划结果
        #     self.plot_path_point(oups_, oups_tgt_, [121])
        #     # 绘制风险评估结果
        #     self.plot_risk_degree(oups_, oups_tgt_, [122])
        # elif self.cfg.use_heuristic and self.cfg.use_risk_assessment:
        #     # 绘制路径规划结果
        #     self.plot_path_point(oups_, oups_tgt_, [221])
        #     # 绘制启发图结果
        #     self.plot_heuristic_fig(oups_, oups_tgt_, [222, 224])
        #     # 绘制风险评估结果
        #     self.plot_risk_degree(oups_, oups_tgt_, [223])
        # else:
        #     self.plot_path_point(oups_, oups_tgt_, [111])

        plt.tight_layout()
        if save_to_disk:
            p = os.path.join(self.cfg.path_results, 'open_loop', 'pathplanning', name)
            os.makedirs(p, exist_ok=True)
            plt.savefig(os.path.join(p, f'{global_step}.png'))
        if show_on_tensorboard:
            self.logger.experiment.add_figure(f'{name}/PathPlanning', fig, global_step=global_step)
        plt.close(fig)

    def plot_path_point(self, oups_, oups_tgt_, position: list):
        # path_point = oups_['path_point'][:(oups_['effective_length'] + 1)].cpu().numpy()
        # path_point_probability = [f[:(oups_['effective_length'] + 1)].cpu().numpy() for f in oups_['path_point_probability']]
        # path_point_tgt = oups_tgt_['path_point'][:(oups_tgt_['effective_length'] + 1)].cpu().numpy()
        path_point = oups_['path_point'].cpu().numpy()
        path_point_tgt = oups_tgt_['path_point'].cpu().numpy()
        if 'path_point_probability' in oups_:
            path_point_probability = [f.cpu().numpy() for f in oups_['path_point_probability']]
        else:
            path_point_probability = None

        plt.subplot(position[0])
        # 绘制规划的点
        l = oups_['effective_length'].item()
        plt.plot(path_point[:l, 1], path_point[:l, 0], 'b-', label='path_point')
        plt.plot(path_point[:l, 1], path_point[:l, 0], 'bo')
        plt.plot(path_point[l:, 1], path_point[l:, 0], 'b--')
        if path_point_probability is not None:
            # 绘制每个点的概率分布
            for i in range(oups_['effective_length']):
                # xy的id（bev坐标系下的）
                x, y = path_point[i, 0], path_point[i, 1]
                # xy的概率分布，排除pad和end的部分
                px, py = path_point_probability[0][i, 2:], path_point_probability[1][i, 2:]
                px = np.exp(px) / np.sum(np.exp(px))
                py = np.exp(py) / np.sum(np.exp(py))
                # assert px.argmax() == x and py.argmax() == y, 'feature与xy的对应关系出错'

                # 找出坐标点周围大于1/len的点的个数，作为分布
                a = np.sum(px > (1 / len(px)))
                b = np.sum(py > (1 / len(py)))
                # a = a * self.cfg.map_bev['resolution'][0]
                # b = b * self.cfg.map_bev['resolution'][1]
                
                # 绘制概率分布y
                ellipse = patches.Ellipse((y, x), width=b, height=a, angle=0, edgecolor='r', facecolor='blue', alpha=0.3)
                plt.gca().add_patch(ellipse)

        # 绘制参考点
        plt.plot(path_point_tgt[:, 1], path_point_tgt[:, 0], 'r--', label='path_point_tgt')
        plt.plot(path_point_tgt[:, 1], path_point_tgt[:, 0], 'r.')
        plt.legend()
        plt.xlim(0, self.cfg.map_bev['final_dim'][0])
        plt.ylim(self.cfg.map_bev['final_dim'][1], 0)
        plt.gca().set_aspect('equal', adjustable='box')

    def plot_heuristic_fig(self, oups_, oups_tgt_, position: list):
        heuristic_fig = oups_['heuristic_fig'].cpu().numpy()
        heuristic_fig_tgt = oups_tgt_['heuristic_fig'].cpu().numpy()

        plt.subplot(position[0])
        plt.imshow(heuristic_fig)
        plt.title('heuristic_fig')
        plt.subplot(position[1])
        plt.imshow(heuristic_fig_tgt)
        plt.title('heuristic_fig_tgt')

    def plot_risk_degree(self, oups_, oups_tgt_, position: list):
        l, l_tgt = oups_['effective_length'].item(), oups_tgt_['effective_length'].item()
        risk_degree = oups_['risk_degree'][:l].cpu().numpy() * 100
        risk_degree_tgt = oups_tgt_['risk_degree'][:l_tgt].cpu().numpy() * 100

        plt.subplot(position[0])
        # plt.bar(np.arange(l), risk_degree, width=1.0, alpha=0.5, color='b', label='risk_degree')
        plt.plot(np.arange(l), risk_degree, 'b', label='risk_degree')
        plt.plot([l-1, l-1], [0, 100], 'b--')
        # plt.bar(np.arange(l_tgt), risk_degree_tgt, width=1.0, alpha=0.5, color='r', label='risk_degree_tgt')
        plt.plot(np.arange(l_tgt), risk_degree_tgt, 'r', label='risk_degree_tgt')
        plt.plot([l_tgt-1, l_tgt-1], [0, 100], 'r--')
        plt.ylim(0, 100)
        plt.legend()
        plt.title('risk_degree')
