import itertools
import os
import time
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import move_data_to_device
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from prettytable import PrettyTable

from .for_base import AgentBase
from ..tools.config import Configuration
from ..models.base.lightning_base import LightningBase
from ..models.model_perception import ModelPerception
from ..models.model_pathplanning import ModelPathPlanning
from ..datas.loader import DatasetLoader
from ..datas.dataset_perception import Dataset_Perception
from ..datas.dataset_path_planning import Dataset_PathPlanning
from ..criterions.base import *
from ..criterions.metric import MetricBatch
from ..agents.for_planner import AgentPlanner
from ..tools.util import check_collision, timefunc

CURV_MEAN = 0.12762490583837288
CURV_STD = 0.05496410119483789

class AgentModelTest(AgentBase):
    def __init__(self, model_type: str, ckpt_path: str, device: str, *args, **kwargs):
        self.Model_Type = {
            'perception': ModelPerception,
            'pathplanning': ModelPathPlanning
        }
        self.Dataset_Type = {
            'perception': Dataset_Perception,
            'pathplanning': Dataset_PathPlanning
        }
        self.model_lightning, self.cfg = self.load_model_from_ckpt(model_type, ckpt_path, device)
        self.ckpt_path = ckpt_path
        self.model_type = model_type
        self.device = device
        self.agent_planner = AgentPlanner(self.cfg, origin_xy=np.array([292.9, -213.245], dtype=np.float32), vehicle_params=self.cfg.ego['base_params'])
        self.path_folder = os.path.join(self.cfg.path_results, 'open_loop', f'{self.model_type}')
        if self.model_type == 'perception':
            if self.cfg.is_rsu:
                self.path_folder = os.path.join(self.path_folder, 'rsu')
            else:
                self.path_folder = os.path.join(self.path_folder, 'obu')
        else:
            self.path_folder = os.path.join(
                self.path_folder, self.cfg.path_planning_head_params['method'], 
                f"{self.cfg.train['lr_pre_trained']:.5f}" if self.cfg.train['lr_pre_trained'] is not None else 'new',
            )
        self.curv_mean = CURV_MEAN
        self.curv_std = CURV_STD
        os.makedirs(self.path_folder, exist_ok=True)
        super().__init__(self.cfg)
        super().init_all()

    def load_model_from_ckpt(self, model_type: str, ckpt_path: str, device: str):
        model_lightning: LightningBase = self.Model_Type[model_type].load_from_checkpoint(
            ckpt_path,
            strict=False
        )
        model_lightning.freeze()
        model_lightning.to(device)
        cfg: Configuration = model_lightning.cfg
        return model_lightning, cfg
    
    def get_model(self):
        return self.model_lightning

    def get_dataset(self, batch_size=None, mode='normal', type='test'):
        dataset = DatasetLoader(
            self.cfg,
            self.Dataset_Type[self.model_type](self.cfg, mode=mode),
            batch_size
        )[type]
        return dataset

    def get_tester(self):
        tester = L.Trainer(
            log_every_n_steps=1, check_val_every_n_epoch=1,
            accelerator=self.device.split(':')[0], devices=1, strategy='auto',
            logger=TensorBoardLogger(
                save_dir=os.path.join(self.cfg.path_ckpts, 'lightning_logs'), 
                name=f'step_{self.cfg.current_step}', default_hp_metric=False
            ),
            default_root_dir=self.cfg.path_ckpts
        )
        return tester

    def run(self):
        # 可视化一些参数
        self.logger.info(f'Testing Model {self.model_type} with ckpt {self.ckpt_path} on device {self.device}')
        formatted_params = self.model_lightning.format_dict_for_print(self.model_lightning.params_log, keys_per_line=2)
        self.logger.info(f'Params_log:\n{formatted_params}')

        # # 先测试一般性误差
        # self.run_by_batch()

        # 遍历所有数据
        self.logger.info('Start to test by Datset-test')
        self.run_by_test()

        if self.model_type == 'pathplanning':
            # 评估整体的成功率
            self.logger.info('Start to test by Datset-free')
            self.run_by_free()
    
    def run_by_batch(self):
        # 获取测试用的模型、数据集和trainer
        model = self.get_model()
        dataset = self.get_dataset()
        tester = self.get_tester()

        tester.test(model, dataloaders=dataset)
    
    def run_by_test(self):
        # 按照batch size为1进行加载数据
        dataset = self.get_dataset(batch_size=1)
        model = self.get_model()
        metrics = {}

        # 计算模型推理结果
        oups, oups_tgt, metrics_duration = self.get_oups(model, dataset, cal_duration=True and self.model_type == 'pathplanning')
        metrics.update(metrics_duration)

        # 添加其他指标
        metrics.update(self.cal_metrics(oups, oups_tgt))

        # 计算AP
        metrics.update(self.cal_AP(oups, oups_tgt))

        # 添加概率密度
        metrics.update(self.cal_density(metrics))

        # 将metrics保存成excel文件
        self.save_metrics(metrics, mode='normal')

        # 展示metrics
        metrics_table = self.show_metrics(metrics, mode='normal')

        # 记录曲率相关的内容
        if self.model_type == 'pathplanning':
            self.curv_mean = metrics_table['path_tgt_curvature_three_point_mean'][0]
            self.curv_std = metrics_table['path_tgt_curvature_three_point_mean'][2]
            self.logger.info(f'The curvature mean is {self.curv_mean} and the curvature std is {self.curv_std}')

    def run_by_free(self):
        dataset = self.get_dataset(batch_size=1, mode='free', type='free')
        model = self.get_model()
        metrics = {}

        # 计算模型推理结果
        oups, oups_tgt, metrics_duration = self.get_oups(model, dataset, cal_duration=False)
        metrics.update(metrics_duration)

        # 计算碰撞
        metrics.update(self.cal_collision(oups, oups_tgt))

        # 添加其他指标
        metrics.update(self.cal_metrics(oups, oups_tgt))

        # 添加概率密度
        metrics.update(self.cal_density(metrics))

        # 将metrics保存成excel文件
        self.save_metrics(metrics, mode='free')

        # 展示metrics
        self.show_metrics(metrics, mode='free')

        # 展示成功率
        self.show_success_rate(metrics)

    def get_oups(self, model, dataset, cal_duration=False):
        oups = []
        oups_tgt = []
        metrics = {}
        if cal_duration:
            distance = []
            duration_hybridAstar = []
            duration_model = []
        
        def process_batch(model, datas_b, device):
            datas_b = move_data_to_device(datas_b, device)
            oups_b = model(**datas_b)
            oups_tgt_b = model.collate_outps_tgt(datas_b)
            return oups_b, oups_tgt_b

        for datas_b in tqdm(dataset, desc='Dataset'):
            if cal_duration:
                # 获取起始距离，左负右正
                start_xyt = datas_b['start_xyt']
                end_xyt = datas_b['end_xyt']
                distance_ = torch.linalg.norm(start_xyt - end_xyt).item()
                if start_xyt[0, 0, 1] > end_xyt[0, 0, 1]:
                    distance_ = -distance_

                duration_hybridAstar_ = datas_b['planning_duration'].item()
                duration_model_, (oups_b, oups_tgt_b) = timefunc(process_batch, model, datas_b, self.device)

                distance.append(distance_)
                duration_hybridAstar.append(duration_hybridAstar_)
                duration_model.append(duration_model_)
            else:
                # 只使用神经网络模型进行推理
                oups_b, oups_tgt_b = process_batch(model, datas_b, self.device)
            oups.append(oups_b)
            oups_tgt.append(oups_tgt_b)
        if cal_duration:
            metrics['distance'] = distance
            metrics['duration_hybridAstar'] = duration_hybridAstar
            metrics['duration_model'] = duration_model
        return oups, oups_tgt, metrics

    def get_useful_data(self, value):
        # 剔除data中的None
        value = [v for v in value if v is not None]
        return value

    def cal_metrics(self, oups, oups_tgt):
        metrics = {}

        ## 计算独立的指标
        sumup = SumUpHandle()
        metric_handle = MetricBatch(self.cfg)
        for oups_b, oups_tgt_b in tqdm(zip(oups, oups_tgt), total=len(oups), desc='Metric-Step1'):
            # # 先剔除oups_tgt_b中关于path_point的部分
            # oups_tgt_b_ = copy.deepcopy(oups_tgt_b)
            # oups_tgt_b_['path_point'] = None
            sumup(metric_handle(oups_b, oups_tgt_b))
        metrics['seg_ego_distance'] = sumup['seg_ego_distance(m)']
        for i in [3, 5, 7, 'hole']:
            metrics[f'points_distance_{i}'] = sumup[f'points_distance_{i}(m)']
            # metrics[f'risk_assessment_{i}'] = sumup[f'risk_assessment_{i}']
        metrics['start_point_distance'] = sumup['start_point_distance(m)']
        metrics['end_point_distance'] = sumup['end_point_distance(m)']
        metrics['path_point_frechet'] = sumup['path_point_frechet(m)']
        metrics['path_point_dtw'] = sumup['path_point_dtw(m)']
        metrics['path_correlation'] = sumup['path_correlation']
        for method, mode in itertools.product(['spline', 'three_point'], ['mean', 'min', 'max']):
            metrics[f'path_pred_curvature_{method}_{mode}'] = sumup[f'path_pred_curvature_{method}_{mode}(1/m)']
            metrics[f'path_pred_radius_{method}_{mode}'] = sumup[f'path_pred_radius_{method}_{mode}(m)']
            metrics[f'path_tgt_curvature_{method}_{mode}'] = sumup[f'path_tgt_curvature_{method}_{mode}(1/m)']
            metrics[f'path_tgt_radius_{method}_{mode}'] = sumup[f'path_tgt_radius_{method}_{mode}(m)']
        
        # 剔除None的部分，此处只是剔除完全没有计算到的数据
        metrics = {k: v for k, v in metrics.items() if v is not None}
        
        assert len(set(len(v) for v in metrics.values())) == 1, 'The length of metrics is not equal'

        return metrics

    def cal_density(self, metrics):
        # 计算每个指标的密度曲线
        metrics_more = {}
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                continue
            assert isinstance(metric_value, list), f'The type of {metric_name} is not list'

            # 剔除None并转换为numpy数组
            data = np.array(self.get_useful_data(metric_value))
            x_values = np.linspace(data.min(), data.max(), len(metric_value))

            # 计算数据的均值和标准差
            mean = np.mean(data)
            std = np.std(data, ddof=1)  # ddof=1 表示使用样本标准差

            # 创建正态分布对象
            normal_dist = stats.norm(loc=mean, scale=std)

            # 计算密度值
            density_values = normal_dist.pdf(x_values)

            metrics_more[f'{metric_name}_density'] = density_values.tolist()
            metrics_more[f'{metric_name}_density_x'] = x_values.tolist()
        return metrics_more
    
    def cal_AP(self, oups, oups_tgt):
        metrics = {}
        metric_handle = mAPMetric(self.cfg)
        for oups_b, oups_tgt_b in tqdm(zip(oups, oups_tgt), total=len(oups), desc='Calculating mAP'):
            metric_handle(oups_b['segmentation'], oups_b['segmentation_onehot'], oups_tgt_b['segmentation'])
        APs, mAP = metric_handle.get_results()
        metrics['mAP'] = mAP
        for k in APs:
            metrics[f'AP@{k}'] = APs[k]
        return metrics

    def cal_collision(self, oups, oups_tgt):
        metrics = {
            'path_pred_collision': []
        }
        for oups_b, oups_tgt_b in tqdm(zip(oups, oups_tgt), total=len(oups), desc='Calculating Collision'):
            path_point = oups_b['path_point'].squeeze(0)[:oups_b['effective_length']].cpu().numpy()
            map_bev = oups_tgt_b['segmentation'].squeeze(0).cpu().numpy()
            start_xyt = oups_tgt_b['start_xyt'].squeeze(0).squeeze(0).cpu().numpy()
            end_xyt = oups_tgt_b['end_xyt'].squeeze(0).squeeze(0).cpu().numpy()
            collision = check_collision(path_point, map_bev, start_xyt, end_xyt, self.cfg)

            metrics['path_pred_collision'].append(collision)
        return metrics

    def save_metrics(self, metrics, mode='normal'):
        p = os.path.join(self.path_folder, mode)
        os.makedirs(p, exist_ok=True)

        df = pd.DataFrame(metrics)
        df.to_excel(
            os.path.join(p, f'metrics.xlsx'),
            index=False,
            engine='openpyxl'
        )

    def save_table_to_fig(self, table, path_fig):
        # 获取列名和数据
        columns = table.field_names
        data = table.rows

        # 创建一个 matplotlib 图像
        fig, ax = plt.subplots()  # 调整图像大小
        ax.axis('off')  # 关闭坐标轴

        # 设置每列宽度，使总和为 1（单位是 Figure 宽度比例）
        num_cols = len(columns)
        col_widths = [1.0 / num_cols] * num_cols  # 平均铺满

        # 创建表格
        table_plot = ax.table(
            cellText=data, colLabels=columns, 
            cellLoc='center',
            loc='upper left',
            # colWidths=col_widths,
            bbox=[0, 0, 1, 1]  # [left, bottom, width, height]，铺满整个 Axes 区域
        )

        # 调整表格样式
        table_plot.auto_set_font_size(True)

        # 保存为图像文件
        plt.savefig(path_fig, dpi=1200, bbox_inches='tight')
        plt.close(fig)  # 关闭图像窗口

    def show_metrics(self, metrics, mode='normal'):
        p = os.path.join(self.path_folder, mode)
        os.makedirs(p, exist_ok=True)

        metrics_table = {}
        for metric_name, metric_value in tqdm(metrics.items(), desc='Show-Metrics', total=len(metrics)):
            assert isinstance(metric_value, (float, list)), f'The type of {metric_name} is not float or list'

            if 'density' in metric_name:
                continue

            if isinstance(metric_value, list):
                if isinstance(metric_value[0], bool):
                    metric_value = np.array(metric_value, dtype=int)
                    bins = 2
                else:
                    metric_value = np.array(metric_value)
                    bins = 100

                # 提取有效部分
                metric_value = self.get_useful_data(metric_value)

                # # 绘制直方图
                # fig = plt.figure()
                # plt.hist(metric_value, bins=bins, density=True, alpha=0.7, color='blue')
                # if f'{metric_name}_density' in metrics:
                #     # 绘制密度曲线
                #     x_values = metrics[f'{metric_name}_density_x']
                #     density_values = metrics[f'{metric_name}_density']
                #     plt.plot(x_values, density_values, color='red', label='Density Curve')
                # plt.legend()
                # plt.title(f'Histogram of {metric_name}')
                # fig.tight_layout()
                # plt.savefig(os.path.join(self.path_folder, p, f'{metric_name}.png'))
                # plt.close(fig)

                # 进行统计
                metrics_table[metric_name] = [
                    np.mean(metric_value).item(), 
                    np.var(metric_value).item(),
                    np.std(metric_value).item(),
                    np.max(metric_value).item(), 
                    np.min(metric_value).item(), 
                ]
            else:
                metrics_table[metric_name] = [metric_value] * 5

        # 使用prettytable展示metrics_table
        table = PrettyTable()
        table.field_names = ['Metric', 'Mean', 'Var', 'Std', 'Max', 'Min']
        for metric_name, metric_value in metrics_table.items():
            table.add_row([metric_name, *metric_value])
        self.logger.info(f'\n{table}')
        self.save_table_to_fig(table, os.path.join(self.path_folder, f'{mode}_metrics.png'))

        return metrics_table

    def show_success_rate(self, metrics):
        # 提取碰撞率
        collision = np.array(metrics['path_pred_collision'])

        # 提取出发率
        start = np.array(metrics['start_point_distance']) < 0.5

        # 提取到达率
        arrive = np.array(metrics['end_point_distance']) < 0.5

        collision_rate = np.mean(collision)
        start_rate = np.mean(start)
        arrive_rate = np.mean(arrive)

        self.logger.info(f'Collision Rate: {collision_rate}')
        self.logger.info(f'Start Rate: {start_rate}')
        self.logger.info(f'Arrive Rate: {arrive_rate}')

        # 提取曲率
        curvatures = np.array(metrics['path_pred_curvature_three_point_mean'])

        # 设定不同阈值
        curv_thresholds = [self.curv_mean + i * self.curv_std for i in [1.645, 1.960, 2.576, 3.291]]
        conf_levels = ['90%', '95%', '99%', '99.9%']

        # 使用prettytable展示Rates
        table = PrettyTable()
        table.field_names = ['Level', 'Collision', 'Start', 'Arrive', 'Curvature', 'Success']

        # 在不同置信区间下计算成功率
        for conf_level, curv_threshold in zip(conf_levels, curv_thresholds):
            curvature_in_range_i = curvatures < curv_threshold
            prob = np.mean(curvature_in_range_i)
            # self.logger.info(f'path_pred curvature in range({conf_level}): {prob}')
            success_rate = np.mean(curvature_in_range_i & ~collision & start & arrive)
            # self.logger.info(f'Success Rate({conf_level}): {success_rate}')
            table.add_row([conf_level, collision_rate, start_rate, arrive_rate, prob, success_rate])
        self.logger.info(f'\n{table}')
        self.save_table_to_fig(table, os.path.join(self.path_folder, 'rates.png'))
        
        # 计算不同置信区间下数据集中路径曲率在范围内的概率
        curvatures = np.array(self.get_useful_data(metrics['path_tgt_curvature_three_point_mean']))
        for conf_level, curv_threshold in zip(conf_levels, curv_thresholds):
            curvature_in_range_i = curvatures < curv_threshold
            prob = np.mean(curvature_in_range_i)
            self.logger.info(f'path_tgt curvature in range({conf_level}): {prob}')