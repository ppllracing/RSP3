from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
import queue
import cv2
import numpy as np
import copy
from scipy.ndimage import label
from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from frechetdist import frdist
from dtaidistance import dtw, dtw_ndim

from ..tools.util import get_position_id_from_bev, cal_curvature, split_path_point, get_depth_label, frechet_distance
from ..tools.config import Configuration
from ..tools.sumup_handle import SumUpHandle

## Losses

class DepthLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(DepthLoss, self).__init__()
        self.cfg = cfg
        self.dtype = cfg.dtype_model_torch
        self.d_range = self.cfg.map_bev['d_range']
        self.down_sample_factor = self.cfg.bev_model_params['bev_down_sample']
        self.depth_channels = round((self.d_range[1] - self.d_range[0]) / self.d_range[2])
    
    # def get_down_sampled_gt_depth(self, gt_depths):
    #     B, N, H, W = gt_depths.shape
    #     dH, dW = H // self.down_sample_factor, W // self.down_sample_factor
    #     gt_depths = gt_depths.view(  # [B, dH, down_sample_factor, dW, down_sample_factor, 1]
    #         B * N, 
    #         dH, self.down_sample_factor,
    #         dW, self.down_sample_factor
    #     )
    #     gt_depths = gt_depths.permute(0, 1, 3, 2, 4).contiguous()  # [B * N, dH, dW, down_sample_factor, down_sample_factor]
    #     gt_depths = gt_depths.view(-1, self.down_sample_factor * self.down_sample_factor)  # [B * dH * dW, down_sample_factor * down_sample_factor]
    #     # 我不明白，为什么要取最小值，而不是取平均值
    #     gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
    #     gt_depths = torch.min(gt_depths_tmp, dim=-1).values  # 取某一区域的最小值最为深度值
    #     # gt_depths = torch.mean(gt_depths, dim=-1)  # 取某一区域的平均值最为深度值
    #     gt_depths = torch.round((gt_depths - self.d_range[0]) / self.d_range[2])
    #     gt_depths = torch.where((0 <= gt_depths) & (gt_depths < self.depth_channels), gt_depths, torch.zeros_like(gt_depths))

    #     # 将深度值离散成depth_channels个类别
    #     gt_depths = gt_depths.view(B * N, dH, dW)
    #     gt_depths = F.one_hot(gt_depths.long(), num_classes=self.depth_channels).view(-1, self.depth_channels)

    #     gt_depths = gt_depths.to(self.dtype)
    #     return gt_depths

    # def get_depth_label(self, gt_depths):
    #     B, N, H, W = gt_depths.shape
    #     dH, dW = H // self.down_sample_factor, W // self.down_sample_factor

    #     gt_depths = gt_depths.view(
    #         B * N, 
    #         dH, self.down_sample_factor,
    #         dW, self.down_sample_factor
    #     )
    #     gt_depths = gt_depths.permute(0, 1, 3, 2, 4).contiguous()
    #     gt_depths = gt_depths.view(B * N, dH, dW, -1)

    #     # 过滤无效（0）深度，取最小非零值作为有效深度
    #     valid_mask = gt_depths > 0.0
    #     gt_depths[~valid_mask] = 1e6
    #     min_depth = torch.min(gt_depths, dim=-1).values
    #     min_depth[min_depth == 1e6] = -1.0  # 无效像素设为 -1

    #     # 离散化：depth -> class
    #     depth_labels = torch.round((min_depth - self.d_range[0]) / self.d_range[2]).long()
    #     depth_labels = torch.where(
    #         (depth_labels >= 0) & (depth_labels < self.depth_channels),
    #         depth_labels,
    #         torch.tensor(-1, device=depth_labels.device)  # 无效标签
    #     )
    #     return depth_labels

    def forward(self, pred_depths, gt_depths):
        # gt_depths = self.get_down_sampled_gt_depth(gt_depths)
        # pred_depths = pred_depths.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_channels).softmax(dim=1)
        # fg_mask = torch.max(gt_depths, dim=1).values > 0.0
        # depth_loss = F.binary_cross_entropy(
        #     pred_depths[fg_mask],
        #     gt_depths[fg_mask],
        #     reduction='mean',
        # )
        pred_depths = pred_depths.permute(0, 1, 3, 4, 2).reshape(-1, self.depth_channels)
        gt_labels = get_depth_label(gt_depths, self.down_sample_factor, self.d_range, self.depth_channels).view(-1)
        depth_loss = F.cross_entropy(
            pred_depths, 
            gt_labels, 
            ignore_index=-1
        )
        return depth_loss

class SegCrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super(SegCrossEntropyLoss, self).__init__()

    def forward(self, seg, seg_tgt):

        # 为bev的每个位置生成seg的id
        _seg_tgt = seg_tgt.argmax(dim=1)

        seg_loss = F.cross_entropy(
            seg,
            _seg_tgt,
            reduction='mean'
        )

        return seg_loss

class SegDiceLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(SegDiceLoss, self).__init__()
    
    def forward(self, seg, seg_tgt):
        pred_soft = F.softmax(seg, dim=1)
        target_idx = seg_tgt.argmax(dim=1)  # [B, H, W]
        target_onehot = F.one_hot(target_idx, num_classes=3).permute(0, 3, 1, 2).to(pred_soft.dtype)  # [B, C, H, W]

        dims = (0, 2, 3)  # across batch and spatial dims
        intersection = (pred_soft * target_onehot).sum(dim=dims)
        union = pred_soft.sum(dim=dims) + target_onehot.sum(dim=dims)
        dice_loss = 1 - ((2 * intersection + 1e-6) / (union + 1e-6)).mean()

        return dice_loss

class OrientationLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(OrientationLoss, self).__init__()
        self.cfg = cfg
    
    def forward(self, orientation, orientation_tgt):
        loss = F.mse_loss(
            orientation,
            orientation_tgt,
            reduction='mean'
        )
        return loss

class AimPointLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(AimPointLoss, self).__init__()
        self.cfg = cfg
    
    def forward(self, aim_point, aim_point_tgt):
        loss = F.mse_loss(
            aim_point,
            aim_point_tgt,
            reduction='mean'
        )
        return loss

class PathPlanningDistributionLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(PathPlanningDistributionLoss, self).__init__()
        self.cfg = cfg
        self.dtype = cfg.dtype_model_torch
        self.weight_sequence = torch.linspace(1.0, 1.5, self.cfg.max_num_for_path, dtype=self.dtype)
    
    def forward(self, path_point_probability, path_point_token_tgt):
        loss = []
        for path_point_probability_, path_point_token_tgt_ in zip(path_point_probability, path_point_token_tgt.permute(2, 0, 1)):
            path_point_probability_ = path_point_probability_.permute(0, 2, 1)  # [B, N, C] -> [B, C, N]
            loss_ = F.cross_entropy(
                path_point_probability_,
                path_point_token_tgt_.long(),
                ignore_index=int(self.cfg.pad_value_for_path_point_token),
                reduction='mean',
                label_smoothing=self.cfg.path_planning_head_params['label_smoothing']
            )
            # loss_ = loss_ * self.weight_sequence.to(loss_.device)
            # loss_ = loss_.mean()
            loss.append(loss_)
        # loss = sum(loss) / len(loss)
        loss = sum(loss)
        return loss

class PathPlanningPathPointLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(PathPlanningPathPointLoss, self).__init__()
        self.cfg = cfg
        self.dtype = cfg.dtype_model_torch
        self.weight_sequence = torch.linspace(1.0, 1.5, self.cfg.max_num_for_path, dtype=self.dtype)
    
    def forward(self, path_point_token, path_point_token_tgt):
        loss = F.mse_loss(
            path_point_token,
            path_point_token_tgt,
            reduction='none'
        )
        loss = loss * self.weight_sequence.unsqueeze(1).to(loss.device)
        loss = loss.mean()

        return loss

class HeuristicLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(HeuristicLoss, self).__init__()
        self.cfg = cfg
    
    def forward(self, heuristic_fig, heuristic_fig_tgt):
        loss = F.l1_loss(
            heuristic_fig,
            heuristic_fig_tgt,
            reduction='mean'
        )
        return loss

class RiskAssessmentLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(RiskAssessmentLoss, self).__init__()
        self.cfg = cfg
        self.dtype = cfg.dtype_model_torch
    
    def forward(self, risk_degree_feature, risk_degree_tgt):
        num_classes = risk_degree_feature.shape[-1]
        risk_degree_tgt_ = torch.round(risk_degree_tgt * (num_classes - 1)).flatten(0, 1).long()
        risk_degree_feature_ = risk_degree_feature.flatten(0, 1).softmax(dim=1)

        # loss = F.cross_entropy(
        #     risk_degree_feature_,
        #     risk_degree_tgt_,
        #     reduction='mean',
        #     ignore_index=0
        # )
        risk_degree_tgt_ = F.one_hot(risk_degree_tgt_, num_classes=num_classes).to(self.dtype)
        loss = F.binary_cross_entropy(
            risk_degree_feature_,
            risk_degree_tgt_,
            reduction='mean'
        )
        return loss

## Metrics

class mAPMetric(nn.Module):
    def __init__(self, cfg: Configuration):
        super(mAPMetric, self).__init__()
        self.cfg = cfg
        self.annotations_pred = []
        self.annotations_gt = []
    
    def forward(self, seg_pred, seg_pred_one_hot, seg_tgt):
        annotations_pred = []
        annotations_gt = []
        # 遍历batch中的数据
        for seg_pred_b, seg_pred_one_hot_b, seg_tgt_b in zip(seg_pred, seg_pred_one_hot, seg_tgt):
            annotations_pred_b = []
            annotations_gt_b = []
            seg_pred_b = seg_pred_b.softmax(dim=0)

            # 转为numpy
            seg_pred_b, seg_pred_one_hot_b, seg_tgt_b = seg_pred_b.cpu().numpy(), seg_pred_one_hot_b.cpu().numpy(), seg_tgt_b.cpu().numpy()

            # 处理freespace
            freespace_occs = self.split_obstacles(seg_pred_one_hot_b[0], seg_pred_one_hot_b[1:].sum(axis=0))  # 以车辆层作为背景
            freespace_cfds = self.get_confidence(freespace_occs, seg_pred_b[0])
            freespace_occ_gt = seg_tgt_b[0]  # 场景中有且只有一个freespace
            # 处理obstacles
            obs_occs = self.split_obstacles(seg_pred_one_hot_b[1], seg_pred_one_hot_b[0])
            obs_cfds = self.get_confidence(obs_occs, seg_pred_b[1])
            obs_occs_gt = self.split_obstacles(seg_tgt_b[1], seg_tgt_b[0])
            # 处理ego
            ego_occs = self.split_obstacles(seg_pred_one_hot_b[2], seg_pred_one_hot_b[0])
            ego_cfds = self.get_confidence(ego_occs, seg_pred_b[2])
            ego_occ_gt = seg_tgt_b[2]  # 场景中有且只有一个ego
            
            # 存放数据
            for freespace_occ, freespace_cfd in zip(freespace_occs, freespace_cfds):
                annotations_pred_b.append([freespace_cfd.item(), ('freespace', freespace_occ)])
            for obs_occ, obs_cfd in zip(obs_occs, obs_cfds):
                annotations_pred_b.append([obs_cfd.item(), ('obs', obs_occ)])
            for ego_occ, ego_cfd in zip(ego_occs, ego_cfds):
                annotations_pred_b.append([ego_cfd.item(), ('ego', ego_occ)])
            annotations_pred_b = sorted(annotations_pred_b, key=lambda x: x[0], reverse=True)  # 按置信度排序

            annotations_gt_b.append(['freespace', freespace_occ_gt])
            for obs_occ_gt in obs_occs_gt:
                annotations_gt_b.append(['obs', obs_occ_gt])
            annotations_gt_b.append(['ego', ego_occ_gt])
            
            annotations_pred.append(annotations_pred_b)
            annotations_gt.append(annotations_gt_b)
        self.annotations_pred.extend(annotations_pred)
        self.annotations_gt.extend(annotations_gt)

    def get_results(self, annotations_pred=None, annotations_gt=None):
        # COCO标准IoU阈值：0.5到0.95，步长0.05
        iou_thresholds = np.linspace(0.5, 0.95, 10)
        APs = SumUpHandle()
        for iou_thresh in tqdm(iou_thresholds, desc='IoU'):
            # 获取当前IoU阈值下的TP，FP，FN（需重新计算）
            tpfp = self.get_TP_FP(annotations_pred, annotations_gt, iou_thresh)
            # 计算当前IoU下的APs
            APs_ = {}
            for label in tpfp:
                APs_[label] = self.cal_ap(*tpfp[label])
            APs(APs_)
        APs = APs.get_sumup_result(method='mean', has_postfix=False)
        mAP = sum(APs.values()) / len(APs)
        return APs, mAP

    # @staticmethod
    # def worker(annotations_pred, annotations_gt, iou_thresh, get_TP_FP, cal_ap):
    #     annotations_pred_ = deepcopy(annotations_pred)
    #     annotations_gt_ = deepcopy(annotations_gt)
    #     tpfp = get_TP_FP(annotations_pred_, annotations_gt_, iou_thresh)
    #     APs_ = {}
    #     for label in tpfp:
    #         APs_[label] = cal_ap(*tpfp[label])
    #     return APs_

    # def get_results(self, annotations_pred=None, annotations_gt=None):
    #     annotations_pred = annotations_pred or self.annotations_pred
    #     annotations_gt = annotations_gt or self.annotations_gt
    #     iou_thresholds = np.linspace(0.5, 0.95, 10)

    #     APs = SumUpHandle()
    #     with ProcessPoolExecutor() as executor:
    #         futures = [
    #             executor.submit(
    #                 self.worker, 
    #                 annotations_pred, annotations_gt, iou, self.get_TP_FP, self.cal_ap
    #             ) for iou in iou_thresholds
    #         ]
    #         for future in tqdm(as_completed(futures), total=len(futures), desc='IoU'):
    #             APs(future.result())

    #     APs = APs.get_sumup_result(method='mean', has_postfix=False)
    #     mAP = sum(APs.values()) / len(APs)
    #     return APs, mAP

    def get_TP_FP(self, annotations_pred, annotations_gt, iou_threshold=0.5):
        annotations_pred = annotations_pred or self.annotations_pred
        annotations_gt = annotations_gt or self.annotations_gt

        # 分别计算AP
        datas = {}
        for label in ['freespace', 'ego', 'obs']:
            # 分batch计算
            tpfp = queue.PriorityQueue()
            num_gt = 0
            for anns_pred_b, anns_gt_b in zip(annotations_pred, annotations_gt):
                # 获取当前类别的预测值和真实值
                anns_pred = []
                cfd_b = []
                anns_pred_b_cp = copy.deepcopy(anns_pred_b)
                for cfd, (ann_pred_label, ann_pred_occ) in anns_pred_b_cp:
                    if ann_pred_label == label:
                        anns_pred.append(ann_pred_occ)
                        cfd_b.append(cfd)
                anns_gt = []
                for ann_gt_label, ann_gt_occ in anns_gt_b:
                    if ann_gt_label == label:
                        anns_gt.append(ann_gt_occ)
                num_gt += len(anns_gt)

                # 计算TP和FP
                tp_b, fp_b = self.evaluate_anns(anns_pred, anns_gt, iou_threshold)
                for cfd_, tp_, fp_ in zip(cfd_b, tp_b, fp_b):
                    tpfp.put((cfd_, (tp_, fp_)))
            # 按顺序取出TP，FP
            tp = []
            fp = []
            while not tpfp.empty():
                cfd, (tp_b, fp_b) = tpfp.get()
                tp.append(tp_b)
                fp.append(fp_b)
            datas[label] = [tp, fp, num_gt]
        return datas
                
    def evaluate_anns(self, anns_pred, anns_gt, iou_threshold):
        # 初始化数据结构
        tp = []
        fp = []
        gt_matched = [False] * len(anns_gt)

        # 遍历预测值
        for pred_idx, ann_pred in enumerate(anns_pred):
            max_iou = 0.0
            matched_idx = -1
            
            # 遍历所有真实框
            for i, ann_gt in enumerate(anns_gt):
                iou = self.cal_iou(ann_pred, ann_gt)
                if iou > max_iou:
                    max_iou = iou
                    matched_idx = i
        
            # 判断TP/FP
            if max_iou >= iou_threshold and not gt_matched[matched_idx]:
                tp.append(1)
                fp.append(0)
                gt_matched[matched_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        return tp, fp

    def cal_iou(self, ann_pred_occ, ann_gt_occ):
        # 交集
        intersection = (ann_pred_occ * ann_gt_occ).sum()
        # 并集
        union = ann_pred_occ.sum() + ann_gt_occ.sum() - intersection
        # 计算IoU
        iou = intersection / (union + 1e-6)
        return iou

    def cal_ap(self, tp, fp, num_gt):
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        if num_gt == 0:
            return 0.0

        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (num_gt + 1e-6)
        
        # 保证 recall 从 0 开始
        if recall[0] > 0:
            recall = np.insert(recall, 0, 0.0)
            precision = np.insert(precision, 0, precision[0])

        # 保证 recall 到达 1
        if recall[-1] < 1:
            recall = np.append(recall, 1.0)
            precision = np.append(precision, 0.0)

        # 平滑PR曲线：每个点的precision取右侧最大值
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        
        # 计算面积（AP）
        ap = np.trapz(precision, recall)
        return ap

    def split_obstacles(self, obstacles_occ, background_occ):
        # 转换成np并使用np进行分割
        obstacles_occ_np = obstacles_occ.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(obstacles_occ_np, connectivity=4)
        # 提取每个obstacle对应的occ
        obs_occ = []
        for i in range(num_labels):
            min_area = 10  # 一辆车至少要识别出10个点
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            _occ = np.zeros_like(obstacles_occ)
            _occ[labels == i] = 1
            if (background_occ * _occ).sum() >= 1:
                continue
            obs_occ.append(_occ)
        return obs_occ

    def get_confidence(self, occs, pred):
        # 计算每个occ的置信度
        cfds = []
        for occ in occs:
            cfd_map = pred * occ
            cfd = cfd_map[cfd_map > 0.0].mean()
            cfds.append(cfd)
        return cfds

class SegmentationEgoDistanceMetric(nn.Module):
    def __init__(self, cfg: Configuration):
        super(SegmentationEgoDistanceMetric, self).__init__()
        self.cfg = cfg
        self.fH, self.fW = self.cfg.map_bev['final_dim'][:2]
        self.resolution = sum(self.cfg.map_bev['resolution'][:2]) / 2
    
    def forward(self, seg, seg_tgt):
        # 从seg和seg_tgt中提取ego的位置
        ego_position_id = torch.cat([
            get_position_id_from_bev(seg_b) for seg_b in seg
        ], dim=0)
        ego_position_id_tgt = torch.cat([
            get_position_id_from_bev(seg_tgt_b) for seg_tgt_b in seg_tgt
        ], dim=0)

        # m = F.l1_loss(ego_position_id, ego_position_id_tgt, reduction='mean')
        m = torch.norm(ego_position_id - ego_position_id_tgt, p=2, dim=1).mean()
        metric = m * self.resolution
        # metric = torch.nan_to_num(metric, nan=(self.fH**2 + self.fW**2)**0.5)
        return metric

class PathPointDistanceMetric(nn.Module):
    def __init__(self, cfg):
        super(PathPointDistanceMetric, self).__init__()
        self.cfg = cfg
        self.fH, self.fW = self.cfg.map_bev['final_dim'][:2]
        self.resolution = sum(self.cfg.map_bev['resolution'][:2]) / 2
    
    def forward(self, path_point, path_point_tgt):
        metric = []
        for path_point_b, path_point_tgt_b in zip(path_point, path_point_tgt):
            l = min(len(path_point_b), len(path_point_tgt_b))
            m = torch.norm(path_point_b[:l] - path_point_tgt_b[:l], p=2, dim=1).mean()
            metric.append(m)
        metric = torch.stack(metric, dim=0).mean() * self.resolution
        return metric

class PathPointFrechetMetric(nn.Module):
    def __init__(self, cfg):
        super(PathPointFrechetMetric, self).__init__()
        self.cfg = cfg
        self.resolution = sum(self.cfg.map_bev['resolution'][:2]) / 2
    
    def forward(self, path_point, path_point_tgt):
        metric = []
        for path_point_b, path_point_tgt_b in zip(path_point, path_point_tgt):
            path_pred = (path_point_b * self.resolution).cpu().numpy()
            path_gt = (path_point_tgt_b * self.resolution).cpu().numpy()

            metric.append(frechet_distance(path_pred, path_gt))
        metric = torch.tensor(metric, device=path_point[0].device).mean()
        return metric

class PathPointDTWMetric(nn.Module):
    def __init__(self, cfg):
        super(PathPointDTWMetric, self).__init__()
        self.cfg = cfg
        self.resolution = sum(self.cfg.map_bev['resolution'][:2]) / 2
    
    def forward(self, path_point, path_point_tgt):
        metric = []
        for path_point_b, path_point_tgt_b in zip(path_point, path_point_tgt):
            path_pred = (path_point_b * self.resolution).cpu().numpy()
            path_gt = (path_point_tgt_b * self.resolution).cpu().numpy()

            metric.append(dtw_ndim.distance(path_pred, path_gt) / max(len(path_pred), len(path_gt)))
        metric = torch.tensor(metric, device=path_point[0].device).mean()
        return metric

class PathCorrelationCosineMetric(nn.Module):
    def __init__(self, cfg):
        super(PathCorrelationCosineMetric, self).__init__()
    
    def forward(self, path_point, path_point_tgt):
        metric = []
        for path_point_b, path_point_tgt_b in zip(path_point, path_point_tgt):
            l = min(len(path_point_b), len(path_point_tgt_b))
            if l < 2:
                m = torch.tensor(0.0, device=path_point_b.device, dtype=path_point_tgt_b.dtype)
            else:
                diff_b = torch.diff(path_point_b[:l], dim=0)
                diff_tgt_b = torch.diff(path_point_tgt_b[:l], dim=0)
                mask = ~((diff_tgt_b[:, 0] == 0) & (diff_tgt_b[:, 1] == 0))
                m = F.cosine_similarity(
                    diff_b[mask],
                    diff_tgt_b[mask],
                    dim=1
                ).mean()
            metric.append(torch.nan_to_num(m, 0))
        metric = torch.stack(metric, dim=0).mean()

        return metric

class EffectiveLengthMetric(nn.Module):
    def __init__(self, cfg: Configuration):
        super(EffectiveLengthMetric, self).__init__()
        self.cfg = cfg
    
    def forward(self, effective_length, effective_length_tgt):
        metric = F.l1_loss(effective_length.float(), effective_length_tgt.float(), reduction='mean')
        return metric

class PathCurvatureMetric(nn.Module):
    def __init__(self, cfg):
        super(PathCurvatureMetric, self).__init__()
        self.cfg = cfg
        self.resolution = sum(self.cfg.map_bev['resolution'][:2]) / 2
    
    def forward(self, path_point, method='spline', mode='mean'):
        metric = []
        for path_point_b in path_point:
            path_point_b_ = path_point_b.cpu().numpy()
            path_point_b_ = path_point_b_ * self.resolution  # 转为m

            # 分割路径点
            paths = split_path_point(path_point_b_)

            # 遍历所有的路径，计算曲率
            curvature = []
            for p in paths:
                cur = cal_curvature(p, method=method)
                curvature.extend(cur.tolist())

            # l = len(path_point_b)
            # if l < 3:
            #     # 没有意义的计算
            #     m = 0.0
            # else:
            #     # 计算相邻点之间的向量
            #     vectors = path_point_b[1:] - path_point_b[:-1]

            #     # 计算向量的长度
            #     lengths = torch.norm(vectors, dim=1)

            #     # 计算向量之间的夹角
            #     dot_products = torch.sum(vectors[:-1] * vectors[1:], dim=1)
            #     cos_angles = dot_products / (lengths[:-1] * lengths[1:])
            #     # 限制点积结果范围
            #     cos_angles = torch.clamp(cos_angles, -1.0 + 1e-6, 1.0 - 1e-6)
            #     angles = torch.acos(cos_angles)

            #     # 计算曲率
            #     curvatures = torch.abs(angles) / lengths[:-1]

            #     if len(curvatures) == 0:
            #         m = 0.0
            #     else:
            #         # 找到最大曲率
            #         curvature = torch.max(curvatures).item()

            #         # 转换为m
            #         m = curvature * self.resolution
            if mode =='mean':
                m = np.mean(curvature).item()
            elif mode =='max':
                m = np.max(curvature).item()
            elif mode =='min':
                m = np.min(curvature).item()
            else:
                raise ValueError(f"Invalid mode: {mode}")
            metric.append(m)
        metric = torch.tensor(metric, device=path_point[0].device).mean()
        return metric

class RiskAssessmentMetric(nn.Module):
    def __init__(self, cfg: Configuration):
        super(RiskAssessmentMetric, self).__init__()
        self.cfg = cfg
    
    def forward(self, risk_degree, risk_degree_tgt):
        metric = []
        for risk_degree_b, risk_degree_tgt_b in zip(risk_degree, risk_degree_tgt):
            l = min(len(risk_degree_b), len(risk_degree_tgt_b))
            m = F.l1_loss(risk_degree_b[:l], risk_degree_tgt_b[:l], reduction='mean')
            metric.append(m)
        metric = torch.stack(metric, dim=0).mean()
        return metric
