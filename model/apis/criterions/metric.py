import itertools
import math
import torch
import torch.nn as nn

from .base import (
    mAPMetric,
    SegmentationEgoDistanceMetric,
    PathPointDistanceMetric,
    PathPointFrechetMetric,
    PathPointDTWMetric,
    PathCorrelationCosineMetric,
    EffectiveLengthMetric,
    PathCurvatureMetric,
    RiskAssessmentMetric
)
from ..tools.config import Configuration

class MetricBatch(nn.Module):
    def __init__(self, cfg: Configuration):
        super(MetricBatch, self).__init__()
        self.cfg = cfg
        self.dtype = cfg.dtype_model_torch

        self.segmentation_ego_distance_metric = SegmentationEgoDistanceMetric(cfg)
        self.path_point_distance_metric = PathPointDistanceMetric(cfg)
        self.path_point_frechet_metric = PathPointFrechetMetric(cfg)
        self.path_point_dtw_metric = PathPointDTWMetric(cfg)
        self.path_correlation_cosine_metric = PathCorrelationCosineMetric(cfg)
        # self.path_correlation_sdtw_metric = pysdtw.SoftDTW(gamma=1.0, dist_func=pysdtw.distance.pairwise_l2_squared, use_cuda=True)
        self.effective_length_metric = EffectiveLengthMetric(cfg)
        self.path_curvature_metric = PathCurvatureMetric(cfg)
        # self.risk_assessment_metric = RiskAssessmentMetric(cfg)

    def forward(self, oups, oups_tgt):
        metrics = {}

        if 'segmentation_onehot' in oups and 'segmentation' in oups_tgt:
            # 计算在segmentation中，对自车进行估计的精度
            metric_seg_ego_dis = self.segmentation_ego_distance_metric(
                oups['segmentation_onehot'][:, 2].to(self.dtype),
                oups_tgt['segmentation'][:, 2]
            )
            metrics['seg_ego_distance(m)'] = metric_seg_ego_dis

        if 'path_point' in oups:
            assert 'effective_length' in oups
            path_point_ = [path_point[:l.long()] for path_point, l in zip(oups['path_point'], oups['effective_length'])]

            # 起点精度
            metric_start_point_distance = self.path_point_distance_metric(
                oups['start_point'],
                oups_tgt['start_point'],
            )
            # 终点的精度，期待能正常泊入
            metric_end_point_distance = self.path_point_distance_metric(
                oups['end_point'],
                oups_tgt['aim_point'],
            )
            metrics['start_point_distance(m)'] = metric_start_point_distance
            metrics['end_point_distance(m)'] = metric_end_point_distance

            # 曲率和曲率半径
            for method, mode in itertools.product(['spline', 'three_point'], ['mean', 'min', 'max']):
            # for method, mode in itertools.product(['three_point'], ['mean']):
                metric_path_pred_curvature = self.path_curvature_metric(path_point_, method=method, mode=mode)
                metric_path_pred_radius = 1 / (metric_path_pred_curvature + 1e-6)
                metrics[f'path_pred_curvature_{method}_{mode}(1/m)'] = metric_path_pred_curvature
                metrics[f'path_pred_radius_{method}_{mode}(m)'] = metric_path_pred_radius

            if oups_tgt['path_point'] is not None:
                assert 'effective_length' in oups_tgt
                path_point_tgt_ = [path_point[:l.long()] for path_point, l in zip(oups_tgt['path_point'], oups_tgt['effective_length'])]

                # 曲率和曲率半径
                for method, mode in itertools.product(['spline', 'three_point'], ['mean', 'min', 'max']):
                # for method, mode in itertools.product(['three_point'], ['mean']):
                    metric_path_tgt_curvature = self.path_curvature_metric(path_point_tgt_, method=method, mode=mode)
                    metric_path_tgt_radius = 1 / (metric_path_tgt_curvature + 1e-6)
                    metrics[f'path_tgt_curvature_{method}_{mode}(1/m)'] = metric_path_tgt_curvature
                    metrics[f'path_tgt_radius_{method}_{mode}(m)'] = metric_path_tgt_radius

                def get_x_i(x, i):
                    return [x_[:i] for x_ in x]

                # 局部路径点的精度
                for i in [3, 5, 7]:
                    metric_points_distance_i = self.path_point_distance_metric(
                        get_x_i(path_point_, i),
                        get_x_i(path_point_tgt_, i)
                    )
                    metrics[f'points_distance_{i}(m)'] = metric_points_distance_i
                
                # 全局路径点的精度
                metric_points_distance_hole = self.path_point_distance_metric(
                    path_point_,
                    path_point_tgt_,
                )
                metrics['points_distance_hole(m)'] = metric_points_distance_hole

                # 基于Frechet距离的路径点精度
                metric_path_point_frechet = self.path_point_frechet_metric(
                    path_point_,
                    path_point_tgt_
                )
                metrics['path_point_frechet(m)'] = metric_path_point_frechet

                # 基于DTW距离的路径点精度
                metric_path_point_dtw = self.path_point_dtw_metric(
                    path_point_,
                    path_point_tgt_
                )
                metrics['path_point_dtw(m)'] = metric_path_point_dtw

                # 基于余弦相似的相关系数
                metric_path_correlation_cosine = self.path_correlation_cosine_metric(
                    path_point_,
                    path_point_tgt_
                )
                metrics['path_correlation'] = metric_path_correlation_cosine

                # 有效长度误差
                metric_effective_length = self.effective_length_metric(
                    oups['effective_length'],
                    oups_tgt['effective_length']
                )
                metrics['effective_length'] = metric_effective_length
            else:
                print
        # if oups.get('risk_degree') is not None and oups_tgt.get('risk_degree') is not None:
        #     risk_degree_ = [risk_degree[:l.long()] for risk_degree , l in zip(oups['risk_degree'], oups['effective_length'])]
        #     risk_degree_tgt_ = [risk_degree[:l.long()] for risk_degree, l in zip(oups_tgt['risk_degree'], oups_tgt['effective_length'])]
        #     # 局部风险评估误差
        #     for i in [3, 5, 7]:
        #         metric_risk_assessment_i = self.risk_assessment_metric(
        #             get_x_i(risk_degree_, i),
        #             get_x_i(risk_degree_tgt_, i)
        #         )
        #         metrics[f'risk_assessment_{i}'] = metric_risk_assessment_i
        #     # 全局风险评估误差
        #     metric_risk_assessment_hole = self.risk_assessment_metric(
        #         risk_degree_,
        #         risk_degree_tgt_
        #     )
        #     metrics['risk_assessment_hole'] = metric_risk_assessment_hole

        # 检查是否存在nan
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any():
                    raise ValueError(f"Metric {k} has nan")
            elif isinstance(v, float):
                if math.isnan(v):
                    raise ValueError(f"Metric {k} has nan")
            else:
                raise ValueError(f"Metric {k} has unsupported type {type(v)}")

        return metrics

class MetricAll(nn.Module):
    def __init__(self, cfg: Configuration):
        super(MetricAll, self).__init__()
        self.cfg = cfg

        self.oups = []
        self.oups_tgt = []

        self.mAP_metric = mAPMetric(cfg)

    def restore_outputs(self, oups, oups_tgt):
        pass
