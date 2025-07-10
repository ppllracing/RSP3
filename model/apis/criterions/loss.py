import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .base import (
    DepthLoss, 
    SegCrossEntropyLoss,
    SegDiceLoss, 
    OrientationLoss, 
    PathPlanningDistributionLoss, 
    PathPlanningPathPointLoss,
    # HeuristicLoss, 
    # RiskAssessmentLoss
)

class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.cfg = cfg

        self.depth_loss = DepthLoss(cfg)
        self.seg_ce_loss = SegCrossEntropyLoss(cfg)
        self.seg_dice_loss = SegDiceLoss(cfg)
        self.orientation_loss = OrientationLoss(cfg)
        self.path_planning_distribution_loss = PathPlanningDistributionLoss(cfg)
        self.path_planning_pathpoint_loss = PathPlanningPathPointLoss(cfg)
        # self.heuristic_loss = HeuristicLoss(cfg)
        # self.risk_assessment_loss = RiskAssessmentLoss(cfg)

    def forward(self, oups, oups_tgt):
        losses = {}

        if 'image_depth' in oups and 'image_depth' in oups_tgt:
            # 计算深度估计的loss
            loss_depth = self.depth_loss(
                oups['image_depth'], 
                oups_tgt['image_depth']
            )
            losses['depth'] = loss_depth
        
        # if 'start_orientation' in oups and 'start_orientation' in oups_tgt:
        #     # 计算orientation的loss
        #     loss_orientation = self.orientation_loss(
        #         oups['start_orientation'], 
        #         oups_tgt['start_orientation']
        #     )
        #     losses['start_orientation'] = loss_orientation

        if 'segmentation' in oups and 'segmentation' in oups_tgt:
            # 计算segmentation的loss
            loss_ce = self.seg_ce_loss(
                oups['segmentation'], 
                oups_tgt['segmentation']
            )
            loss_dice = self.seg_dice_loss(
                oups['segmentation'], 
                oups_tgt['segmentation']
            )
            loss_seg = 0.7 * loss_ce + 0.3 * loss_dice
            losses['seg'] = loss_seg

        if 'path_point_probability' in oups and 'path_point_token' in oups_tgt:
            # 计算path planning的loss
            if oups_tgt['path_point_token'] is None:
                loss_path = torch.tensor(0.0, device=loss_seg.device)
            else:
                loss_path = self.path_planning_distribution_loss(
                    oups['path_point_probability'],
                    oups_tgt['path_point_token']
                )
                # loss_path_end = self.path_planning_distribution_loss(
                #     [torch.stack([p_b[l-1:l] for p_b, l in zip(p, oups_tgt['effective_length'].long())], dim=0) for p in oups['path_point_probability']],
                #     torch.stack([p_b[l-1:l] for p_b, l in zip(oups_tgt['path_point_token'], oups_tgt['effective_length'].long())], dim=0)
                # )
                # alpha = 0.8
                # loss_path = alpha * loss_path + (1 - alpha) * loss_path_end
            losses['path'] = loss_path
        elif (not 'path_point_probability' in oups) and 'path_point_token' in oups and 'path_point_token' in oups_tgt:
            loss_path = self.path_planning_pathpoint_loss(
                oups['path_point_token'],
                oups_tgt['path_point_token']
            )
            losses['path'] = loss_path

        # if 'heuristic_fig' in oups and 'heuristic_fig' in oups_tgt:
        #     # 计算heuristic的loss
        #     loss_heuristic = self.heuristic_loss(
        #         oups['heuristic_fig'],
        #         oups_tgt['heuristic_fig']
        #     )
        #     losses['heuristic'] = loss_heuristic

        # if 'risk_degree_feature' in oups and 'risk_degree' in oups_tgt:
        #     # 计算risk assessment的loss
        #     loss_risk_degree = self.risk_assessment_loss(
        #         oups['risk_degree_feature'],
        #         oups_tgt['risk_degree']
        #     )
        #     losses['risk_degree'] = loss_risk_degree

        # 整理loss
        losses['all'] = sum(losses.values())

        # 检查是否存在nan
        for k, v in losses.items():
            if torch.isnan(v):
                raise ValueError(f"Loss {k} is nan")

        return losses
