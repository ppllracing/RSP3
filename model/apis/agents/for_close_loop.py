import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .for_base import AgentBase
from .for_carla import AgentCarla
from .for_map import AgentMap
from .for_get_datas_from_carla import AgentGetDatasFromCarla
from ..tools.config import Configuration
from ..tools.util import save_datas_to_disk, plot_standard_rectangle

class AgentCloseLoopBase(AgentBase):
    def __init__(self, cfg: Configuration, agent_carla: AgentCarla, agent_map: AgentMap, agent_get_datas_from_carla: AgentGetDatasFromCarla):
        super().__init__(cfg)
        self.agent_carla = agent_carla
        self.agent_map = agent_map
        self.agent_get_datas_from_carla = agent_get_datas_from_carla
        super().init_all()

        self.records = {
            'xyzPYR': [],
            'xyzPYR_rear': []
        }

    def reset(self):
        self.records = {
            'xyzPYR': [],
            'xyzPYR_rear': []
        }

    def transform_to_inputs(self, datas):
        raise NotImplementedError

    def main(self, inputs):
        raise NotImplementedError

    def plot_infos(self, infos, path_fig):
        assert NotImplementedError

    def record_vehicle_xyzPYR(self, datas_vehicle):
        self.records['xyzPYR'].append(datas_vehicle['xyzPYR'])
        self.records['xyzPYR_rear'].append(datas_vehicle['xyzPYR_rear'])

    def cal_errors(self):
        datas = self.agent_get_datas_from_carla.get_datas(
            datas_vehicle=True, 
            datas_parking_plot=True
        )
        datas = {
            'vehicle': self.agent_get_datas_from_carla.trans_for_datas_vehicle(datas['vehicle']),
            'parking_plot': self.agent_get_datas_from_carla.trans_for_parking_plot(datas['parking_plot'])
        }

        # 检测自车和目标车位的距离差
        xy_ego = datas['vehicle']['xyzPYR_rear'][:2]
        xy_pp = datas['parking_plot']['xyzPYR_aim'][:2]
        error_distance = np.linalg.norm(xy_ego - xy_pp).item()

        # 检测自车和目标车位的航向差
        yaw_ego = datas['vehicle']['xyzPYR_rear'][4]
        yaw_pp = datas['parking_plot']['xyzPYR_aim'][4]
        error_yaw = abs(yaw_ego - yaw_pp).item()

        errors = {
            'distance(m)': error_distance,
            'yaw(degree)': error_yaw
        }
        return errors

    def run(self, datas, path_save=None):
        inputs = self.transform_to_inputs(datas)
        infos = self.main(inputs)
        errors = self.cal_errors()

        # 整理errors
        infos = {
            **infos,
            'errors': errors
        }

        # 保存数据
        if path_save is not None:
            os.makedirs(path_save, exist_ok=True)
            save_datas_to_disk(infos, path_save, 'infos', 'json')
            self.plot_infos(infos, os.path.join(path_save, 'fig.png'))

        self.reset()
        return infos

# 基于本文模型的闭环测试Agent
from ..models.model_pathplanning import ModelPathPlanning
from ..tools.util import extract_data_for_path_planning_from_datas
class AgentCloseLoopOurModel(AgentCloseLoopBase):
    def __init__(self, cfg: Configuration, model: ModelPathPlanning, agent_carla: AgentCarla, agent_map: AgentMap, agent_get_datas_from_carla: AgentGetDatasFromCarla):
        super().__init__(cfg, agent_carla, agent_map, agent_get_datas_from_carla)
        self.model = model
    
    def transform_to_inputs(self, datas):
        device = self.model.device

        # 将datas转换为dataset
        dataset = extract_data_for_path_planning_from_datas(datas, self.cfg)
        dataset = {k: v[0] for k, v in dataset.items()}  # 去除list的封装
        # 将dataset转换为tensor的格式
        batch = {}
        for k, v in dataset.items():
            if isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).to(device).unsqueeze(0)
            elif isinstance(v, float):
                batch[k] = torch.tensor(v).to(device).unsqueeze(0)
            elif isinstance(v, int):
                batch[k] = torch.tensor(v).to(device).unsqueeze(0)
            elif v is None:
                # 当前规划的内容为空
                batch[k] = None
            else:
                raise ValueError(f'Unsupported type: {type(v)}')

        return batch

    def follow_path_point(self, path_point):
        flags = {
            'plan': False,
            'arrive': False,
            'collision': False,
        }

        # 获取终点，并通过终点的距离来判断是否规划成功
        datas_end = self.agent_get_datas_from_carla.trans_for_parking_plot(
            self.agent_get_datas_from_carla.datas_parking_plot_once()
        )
        flags['plan'] = np.linalg.norm(path_point[-1, 2:] - datas_end['xyzPYR_aim'][:2]).item() < 1.0

        start_time = time.time()
        if flags['plan']:
            flags['arrive'] = True
            for point in path_point:
                aim_xyz = point[2:4].tolist() + [0]
                success = self.agent_carla.call_ego_to_location(aim_xyz, time_limit=10, dis_limit=2.0 * self.cfg.collect['jump_dis'])
                self.record_vehicle_xyzPYR(self.agent_get_datas_from_carla.datas_vehicle_once())
                if not success:
                    flags['arrive'] = False
                    break
        flags['collision'] = self.agent_carla.actors_dict['collision']['ego']['event'] is not None
        time_end = time.time()
        duration = time_end - start_time

        infos = {
            'duration': duration,
            'flags': flags,
            'records': self.records.copy(),
        }
        return infos

    def transform_to_path_point(self, oups):
        effective_length = oups['effective_length']
        path_point = oups['path_point'].squeeze(0)[:effective_length].cpu().numpy()

        # 通过id补充坐标
        x_ids, y_ids = path_point[:, 0], path_point[:, 1]
        x_ = self.agent_map.bev_points[0, 0, 0, 0] + x_ids * self.cfg.map_bev['resolution'][0] * -1.0
        y_ = self.agent_map.bev_points[0, 0, 0, 1] + y_ids * self.cfg.map_bev['resolution'][1]
        path_point = np.stack([x_ids, y_ids, x_, y_], axis=1)
        return path_point

    def main(self, inputs):
        oups = self.model(None, **inputs)
        path_point = self.transform_to_path_point(oups)
        infos = self.follow_path_point(path_point)
        infos.update({'planned_path_point': path_point.tolist()})
        return infos
    
    def plot_infos(self, infos, path_fig):
        fig, axs = plt.subplots(1, 2)
        fig.tight_layout()

        ## subplot: 车辆轨迹
        ax = axs[0]
        planned_path_point_rear = np.array(infos['planned_path_point']).reshape(-1, 4)
        record_xyzPYR_rear = np.array(self.records['xyzPYR_rear']).reshape(-1, 6)
        # 绘制地图四个角点
        vertices = self.agent_map.get_vertices_of_global()
        plot_standard_rectangle(vertices, ax, 'k-')
        ax.set_aspect('equal', adjustable='box')
        # 绘制规划出的路径点
        ax.plot(planned_path_point_rear[:, 3], planned_path_point_rear[:, 2], 'r-', label='planned_path_point')
        ax.plot(planned_path_point_rear[:, 3], planned_path_point_rear[:, 2], 'r.')
        # 绘制车辆实际走的点
        ax.plot(record_xyzPYR_rear[:, 1], record_xyzPYR_rear[:, 0], 'b-', label='actual_path_point')
        ax.plot(record_xyzPYR_rear[:, 1], record_xyzPYR_rear[:, 0], 'b.')
        ax.legend(loc='upper right')

        # 绘制点的误差
        ax = axs[1]
        l = min(len(planned_path_point_rear), len(record_xyzPYR_rear))
        point_error = np.linalg.norm(record_xyzPYR_rear[:l, :2] - planned_path_point_rear[:l, 2:], axis=1)
        ax.bar(np.arange(l), point_error, color='r')

        # 保存图片
        plt.savefig(path_fig)
        plt.close(fig)


# 基于E2EParking的闭环测试Agent
# sys.path.append(os.path.join(*(['/'] + os.path.dirname(__file__).split('/')[1:-3] + ['E2EParkingCARLA'])))
# from data_generation.network_evaluator import NetworkEvaluator
# from data_generation.keyboard_control import KeyboardControl
# class AgentCloseLoopE2EParkingCARLA(AgentCloseLoopBase):
#     def __init__(self, cfg: Configuration, agent_carla: AgentCarla, agent_map: AgentMap, agent_get_datas_from_carla: AgentGetDatasFromCarla):
#         super().__init__(cfg, agent_carla, agent_map, agent_get_datas_from_carla)

#         network_evaluator = NetworkEvaluator(agent_carla.world, args, settings)


