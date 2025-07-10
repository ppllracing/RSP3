import os
import time
from matplotlib.gridspec import GridSpec
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from prettytable import PrettyTable

from .for_map import AgentMap
from .for_parking_plot import AgentParkingPlot
from .for_base import AgentBase
from .for_carla import AgentCarla
from .for_model_dl import AgentModelDL
from .for_planner import AgentPlanner
from .for_get_datas_from_carla import AgentGetDatasFromCarla
from ..tools.config import Configuration
from ..tools.sumup_handle import SumUpHandle
from ..tools.util import extract_data_for_path_planning_from_datas

class PathPlanningFail(Exception):
    pass

class FolowPathFail(Exception):
    pass

class AgentTestCase(AgentBase):
    def __init__(self, cfg: Configuration):
        super().__init__(cfg)
        self.agent_map = None
        self.agent_parking_plot = None
        self.agent_carla = None
        self.agent_planner = None
        self.agent_get_datas_from_carla = None
        self.models = {
            'dl': None,
            'rl': None
        }
        self.sumup_handle = {
            'A*': SumUpHandle(),
            'E2E': SumUpHandle()
        }
        self.tbar = tqdm(range(self.cfg.test_case['num']), desc='Case', leave=False)

        self.snapshot = None
        self.case_result = {
            'runtime_all': None,
            'runtime_path_planning': None,
            'runtime_follow_path': None,
            'flag_success': None,
            'flag_path_planning': None,
            'flag_follow_path': None,
            'error_position': None,
            'error_orientation': None
        }

        self.init_all()

    def init_all(self):
        self.agent_parking_plot = AgentParkingPlot(self.cfg)
        self.logger.info(f'Finish to Initialize AgentParkingPlot')

        self.agent_map = AgentMap(self.cfg)
        self.agent_map.init_from_seleted_parking_plot(self.agent_parking_plot)
        self.logger.info(f'Finish to Initialize AgentMap')

        self.agent_carla = AgentCarla(
            self.cfg, self.cfg.carla_client,
            agent_map=self.agent_map, agent_parking_plot=self.agent_parking_plot
        )
        self.logger.info(f'Finish to Initialize AgentCarla')

        self.agent_planner = AgentPlanner(
            self.cfg,
            self.agent_map.bev_points[0, 0, 0, :2], 
            self.agent_carla.actors_dict['vehicle']['ego']['base_params']
        )
        self.logger.info(f'Finish to Initialize AgentPlanner')

        self.agent_get_datas_from_carla = AgentGetDatasFromCarla(
            self.cfg,
            {'carla': self.agent_carla, 'parking_plot': self.agent_parking_plot, 'map': self.agent_map, 'planner': self.agent_planner}
        )
        self.logger.info(f'Finish to Initialize AgentGetDatasFromCarla')

        self.models['dl'] = AgentModelDL(self.cfg).get_model_final()
        self.logger.info(f'Finish to Load Model')

        self.reset()

        super().init_all()
    
    def reset(self):
        self.agent_carla.reset()
        # self.agent_parking_plot.reset_pp_infos_usable()
        # self.agent_parking_plot.reset_pp_vehicle_infos()
        # self.agent_parking_plot.reset_selecet_a_new_pp_id()
        # self.agent_carla.reset_vehicle()
        # self.agent_carla.reset_collision()
        # self.agent_carla.reset_obstacle()
    
    def record_from_snapshot(self):
        self.snapshot = self.agent_carla.record_vehicles_from_snapshot()
        self.logger.debug(f'Record snapshot at {self.snapshot["timestamp"]}')
    
    def restore_to_snapshot(self):
        self.agent_carla.restore_vehicles_to_snapshot(self.snapshot)
        time.sleep(0.1)
        self.logger.debug(f'Restore to snapshot at {self.snapshot["timestamp"]}')

    def get_global_path(self, mode, datas):

        # 获取全局规划路径
        if mode == 'A*':
            _datas = self.agent_get_datas_from_carla.get_datas(
                datas_path=True, is_global_path=True, **datas
            )
            if _datas['path']['success']:
                global_path = _datas['path']['path_points_rear'][:, :4]  # 只提取xy的id和坐标
            else:
                raise PathPlanningFail()
        else:
            # 将datas转换为pkl文件中存储的数据格式
            dataset = extract_data_for_path_planning_from_datas({
                'stamp': self.agent_get_datas_from_carla.trans_for_stamp(datas['stamp'], datas['stamp']),
                'camera': self.agent_get_datas_from_carla.trans_for_datas_camera(datas['camera'], datas['stamp']),
                'bev': self.agent_get_datas_from_carla.trans_for_datas_bev(datas['bev']),
                'aim': self.agent_get_datas_from_carla.trans_for_datas_aim(datas['aim'])
            }, cfg=self.cfg)
            dataset = {k: v[0] for k, v in dataset.items()}
            
            # 整理成batch
            batch = {}
            for k, v in dataset.items():
                if isinstance(v, np.ndarray):
                    batch[k] = torch.from_numpy(v).to(self.cfg.device).unsqueeze(0)
                elif isinstance(v, float):
                    batch[k] = torch.tensor(v).to(self.cfg.device).unsqueeze(0)
                elif isinstance(v, int):
                    batch[k] = torch.tensor(v).to(self.cfg.device).unsqueeze(0)
                elif v is None:
                    # 当前规划的内容为空
                    batch[k] = None
                else:
                    raise ValueError(f'Unsupported type: {type(v)}')

            # 推理
            oups = self.models['dl'](**batch)

            # 获取规划结果
            effective_length = oups['effective_length']
            global_path = oups['path_point'][0, :effective_length].cpu().numpy()

            # 通过id补充坐标
            x_ids, y_ids = global_path[:, 0], global_path[:, 1]
            x_ = self.agent_map.bev_points[0, 0, 0, 0] + x_ids * self.cfg.map_bev['resolution'][0] * -1.0
            y_ = self.agent_map.bev_points[0, 0, 0, 1] + y_ids * self.cfg.map_bev['resolution'][1]
            global_path = np.stack([x_ids, y_ids, x_, y_], axis=1)

        global_path = np.array(global_path, dtype=self.cfg.dtype_model)
        return global_path

    def control_and_collect_data(self, aim_xyz):
        success = self.agent_carla.call_ego_to_location(aim_xyz, time_limit=10, dis_limit=2.0 * self.cfg.collect['jump_dis'])
        if not success:
            raise FolowPathFail()

    def cal_position_error(self, datas_current):
        vehicle_xyzPYR = datas_current['vehicle']['xyzPYR_rear']
        aim_xyzPYR = datas_current['parking_plot']['xyzPYR_aim']
        position_error = np.linalg.norm(np.array(vehicle_xyzPYR[:2]) - np.array(aim_xyzPYR[:2]))
        return position_error

    def cal_orientation_error(self, datas_current):
        vehicle_xyzPYR = datas_current['vehicle']['xyzPYR_rear']
        aim_xyzPYR = datas_current['parking_plot']['xyzPYR_aim']
        orientation_error = abs(vehicle_xyzPYR[4] - aim_xyzPYR[4])
        return orientation_error

    def case(self, mode):
        assert mode in ['A*', 'E2E'], 'The mode should be A* or E2E'
        self.logger.debug(f'Start to run [{mode}] case')

        # 获取当前carla内的数据
        datas_start = self.agent_get_datas_from_carla.get_datas(
            datas_stamp=True, datas_camera=True, datas_vehicle=True, 
            datas_parking_plot=True, datas_bev=True, datas_aim=True
        )

        # 获取路径规划结果
        self.case_result['runtime_path_planning'] = time.time()
        global_path = self.get_global_path(mode, datas_start)
        self.case_result['flag_path_planning'] = True
        self.case_result['runtime_path_planning'] = time.time() - self.case_result['runtime_path_planning']

        # 控制车辆行驶，并实时采集数据
        self.case_result['runtime_follow_path'] = time.time()
        for path_point in tqdm(
            global_path,
            desc='Follow Path',
            unit='point',
            leave=False
        ):
            aim_xyz = path_point[2:4].tolist() + [0]
            self.control_and_collect_data(aim_xyz)
        self.case_result['runtime_follow_path'] = time.time() - self.case_result['runtime_follow_path']
    
    def run(self, mode):
        try:
            self.case_result['runtime_all'] = time.time()
            self.case(mode)
            self.case_result['runtime_all'] = time.time() - self.case_result['runtime_all']
        except PathPlanningFail:
            # 规划都失败了，自然所有的指标都没有意义了
            self.case_result['runtime_all'] = None
            self.case_result['runtime_path_planning'] = None
            self.case_result['runtime_follow_path'] = None
            self.case_result['flag_success'] = False
            self.case_result['flag_path_planning'] = False
            self.case_result['flag_follow_path'] = False
            self.case_result['error_position'] = None
            self.case_result['error_orientation'] = None
            self.logger.debug(f'PathPlanningFail')
        except FolowPathFail:
            self.case_result['runtime_all'] = None
            self.case_result['runtime_follow_path'] = None
            self.case_result['flag_success'] = False
            self.case_result['flag_follow_path'] = False
            self.case_result['error_position'] = None
            self.case_result['error_orientation'] = None
            self.logger.debug(f'FolowPathFail')
        else:
            self.case_result['flag_path_planning'] = True
            self.case_result['flag_follow_path'] = True
            self.case_result['flag_success'] = True
            self.logger.debug(f'Success')

            # 计算车停好之后的误差
            datas_current = self.agent_get_datas_from_carla.get_datas(
                datas_vehicle=True, datas_parking_plot=True, datas_aim=True
            )
            self.case_result['error_position'] = self.cal_position_error(datas_current)
            self.case_result['error_orientation'] = self.cal_orientation_error(datas_current)
        finally:
            # 记录case_result
            self.sumup_handle[mode](self.case_result)

            # 重置case_result
            self.case_result = {k: None for k in self.case_result.keys()}
    
    def get_sumup_result(self, mode, **kwargs):
        result = self.sumup_handle[mode].get_sumup_result(**kwargs)
        return result
    
    def show_result(self, mode):
        result_rec = self.sumup_handle[mode].datas
        result = self.get_sumup_result(mode, method='mean', has_postfix=False)
        tabel = PrettyTable()
        tabel.field_names = ['id'] + list(result.keys())

        # 展示每个id对应的结果
        for i in range(self.cfg.test_case['num']):
            row = [i]
            for k, v in result_rec.items():
                if k.startswith(f'runtime_') or k.startswith(f'error_'):
                    if v[i] is None:
                        row.append(None)
                    else:
                        row.append(f'{v[i]:.4f}')
                elif k.startswith(f'flag_'):
                    row.append(bool(v[i]))
                else:
                    raise ValueError(f'Unsupported key: {k}')
            tabel.add_row(row)
        
        # 中间加一行进行隔断
        tabel.add_row(['-'] * len(tabel.field_names))

        # 展示总体均值结果
        row = ['mean']
        for k, v in result.items():
            if k.startswith(f'runtime_') or k.startswith(f'error_'):
                row.append(f'{v:.4f}')
            elif k.startswith(f'flag_'):
                row.append(f'{v * 100:.2f}%')
            else:
                raise ValueError(f'Unsupported key: {k}')
        tabel.add_row(row)
        self.logger.info(f'Result of [{mode}]:\n{tabel}')
    
    def plot_result(self, mode):
        # 提取结果
        result_rec = self.sumup_handle[mode].datas
        result = self.get_sumup_result(mode, method='mean', has_postfix=False)

        ids = np.linspace(1, self.cfg.test_case['num'], self.cfg.test_case['num'])
        num = len(self.case_result)

        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(int(np.ceil(num/2)), 2, figure=fig)
        axs_layers = [fig.add_subplot(g) for g in gs]

        for i in range(num):
            ax = axs_layers[i]

            k = list(result_rec.keys())[i]
            v = np.array(result_rec[k], dtype=np.float32)
            mask = ~np.isnan(v)
            ids_ = ids[mask]
            v_ = v[mask]

            ax.set_title(k)
            for x, y in zip(ids_, v_):
                ax.plot([x, x], [0, y], 'k-')
                ax.plot(x, y, 'bo')
            ax.plot([ids.min(), ids.max()], [result[k], result[k]], 'r--')
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
            if k.startswith(f'runtime_') or k.startswith(f'error_'):
                ax.text(ids.max(), result[k], f'{result[k]:.4f}', ha='right', va='bottom', bbox=bbox)
            elif k.startswith(f'flag_'):
                ax.text(ids.max(), result[k], f'{result[k] * 100:.2f}%', ha='right', va='bottom', bbox=bbox)
            else:
                raise ValueError(f'Unsupported key: {k}')
            ax.set_xlim(ids.min() - 1, ids.max() + 1)
            ax.grid(True)
        plt.tight_layout()
        
        path_folder = os.path.join(self.cfg.path_results, 'test_case')
        os.makedirs(path_folder, exist_ok=True)
        path_fig = os.path.join(path_folder, f'{mode}.png')
        plt.savefig(path_fig)
