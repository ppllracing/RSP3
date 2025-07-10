import os
import random
import copy
import datetime
import carla
import math
import numpy as np
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

from ..tools.config import Configuration
from ..agents.for_carla import AgentCarla
from ..agents.for_parking_plot import AgentParkingPlot
from ..agents.for_map import AgentMap
from ..agents.for_planner import AgentPlanner
from ..agents.for_plot_in_generator import AgentPlotInGenerator
from ..agents.for_get_datas_from_carla import AgentGetDatasFromCarla
from ..agents.for_condition_setting import AgentConditionSetting
from ..tools.util import init_logger

class Collector():
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.fps = cfg.fps
        self.jump_dis = cfg.collect['jump_dis']
        self.flag_show = cfg.collect['flag_show']
        self.flag_save = cfg.collect['flag_save']
        self.agent_carla: AgentCarla = None
        self.agent_parking_plot: AgentParkingPlot = None
        self.agent_map: AgentMap = None
        self.agent_planner: AgentPlanner = None
        self.agent_plot: AgentPlotInGenerator = None
        self.agent_get_datas_from_carla: AgentGetDatasFromCarla = None
        self.agent_condition_setting: AgentConditionSetting = None

        self.logger = None
        self.fig = None
        self.axs_layers = None
        self.celluloid_camera = None
        self.tbar_collect_origin_data = None
        self.seq_num = 0

        self.collision_stamp = 0.0
        self.datas = {
            'global_aim': None,
            'global_path': None,
            'parking_plot': None,
            'setting_id': None,
            'stamp': [],
            'camera': [],
            'vehicle': [],
            'aim': [],
            'path': [],
            'bev': []
        }
        self.datas_copy = copy.deepcopy(self.datas)
        self.path_datas = None
        self.path_dataset = None
        self.datas_folders = None
        self.num_folder = 0
        self.num_folder_max = cfg.collect['num_folder_max']
        self.flag_collect_origin_data = True
        self.flag_origin_data_to_dataset = False
        self.flag_finish = False
        self.flag_change_ego_pose = True

        self.init_all()

    # 全部初始化
    def init_all(self):
        self.logger = init_logger(self.cfg, self.__class__.__name__)
        self.logger.info('Finish to Initialize Logger')

        self.init_data_folder()
        self.logger.info(f"There has been {self.num_folder}/{self.num_folder_max} datas. \n{self.datas_folders}")

        if self.num_folder < self.num_folder_max:
            self.agent_parking_plot = AgentParkingPlot(self.cfg)
            self.logger.info('Finish to Initialize AgentParkingPlot')

            self.agent_map = AgentMap(self.cfg)
            self.agent_map.init_from_seleted_parking_plot(self.agent_parking_plot)
            self.logger.info('Finish to Initialize AgentMap')

            self.agent_carla = AgentCarla(
                self.cfg, self.cfg.carla_client,
                agent_map=self.agent_map, agent_parking_plot=self.agent_parking_plot
            )
            self.logger.info('Finish to Initialize AgentCarla')

            self.agent_planner = AgentPlanner(
                self.cfg,
                self.agent_map.get_origin_xy(), 
                self.agent_carla.actors_dict['vehicle']['ego']['base_params']
            )
            self.logger.info('Finish to Initialize AgentPlanner')

            # # 保存Planner实例化时候所需要的参数
            # import pickle
            # with open('for_init.pkl', 'wb') as f:
            #     pickle.dump({
            #         'cfg': self.cfg,
            #         'origin_xy': self.agent_map.bev_points[0, 0, 0, :2],
            #         'vehicle_params': self.agent_carla.actors_dict['vehicle']['ego']['base_params']
            #     }, f)

            self.agent_get_datas_from_carla = AgentGetDatasFromCarla(
                self.cfg,
                {'carla': self.agent_carla, 'parking_plot': self.agent_parking_plot, 'map': self.agent_map, 'planner': self.agent_planner}
            )
            self.logger.info('Finish to Initialize AgentGetDatasFromCarla')

            self.reset()

            self.agent_plot = AgentPlotInGenerator(
                self.cfg,
                bev_range=self.agent_map.bev_range_global
            )
            self.logger.info('Finish to Initialize AgentPlotInGenerator')

            # 读取condition_settings，并补充至完善
            self.agent_condition_setting = AgentConditionSetting(self.cfg)
            if len(self.agent_condition_setting) < self.num_folder_max:
                self.agent_condition_setting.generate_settings(
                    self.num_folder_max - len(self.agent_condition_setting),
                    agent_carla=self.agent_carla,
                    agent_parking_plot=self.agent_parking_plot
                )
            # import time
            # for i in tqdm(range(len(self.agent_condition_setting))):
            #     setting = self.agent_condition_setting.get_setting_by_id(i)
            #     for vehicle_info in ([setting.data['actor']['vehicle']['ego']] + setting.data['actor']['vehicle']['obstacle']):
            #         if vehicle_info['xyzPYR'][2] == 1:
            #             self.agent_condition_setting.set_agents_from_setting(
            #                 setting, check_equal=False,
            #                 agent_carla=self.agent_carla, agent_parking_plot=self.agent_parking_plot
            #             )
            #             self.agent_condition_setting.change_setting_by_id(
            #                 i, self.agent_condition_setting.get_current_setting(self.agent_carla, self.agent_parking_plot)
            #             )
            #             break
            #         else:
            #             time.sleep(0.1)
            #             continue
            self.agent_condition_setting.save_settings()
            self.agent_condition_setting.analyse_settings()
            self.logger.info('Finish to Initialize AgentConditionSetting')

    # 读取已有的数据量
    def init_data_folder(self):
        self.path_datas = self.cfg.path_datas
        assert os.path.exists(self.path_datas), 'Path Error!'
        self.datas_folders = []
        for f in os.listdir(self.path_datas):
            p = os.path.join(self.path_datas, f)
            if os.path.isdir(p):
                if os.path.exists(os.path.join(p, 'datas.pkl')):
                    self.datas_folders.append(os.path.join(self.path_datas, f))
                else:
                    # 虽然有这个文件夹，但是没有有效数据，删除掉
                    shutil.rmtree(p)
        self.datas_folders = sorted(
            self.datas_folders,
            key=lambda x: int(x.split('/')[-1].split('_')[-1]),
        )[0:self.num_folder_max]
        self.num_folder = len(self.datas_folders)
        self.path_dataset = self.cfg.path_dataset

    # 重置采集数据中的所有设定
    def reset(self, setting=None):
        if setting is None:
            # self.agent_carla.reset()
            pass
        else:
            self.agent_condition_setting.set_agents_from_setting(
                setting,
                agent_carla=self.agent_carla, agent_parking_plot=self.agent_parking_plot
            )
        self.datas = copy.deepcopy(self.datas_copy)
        self.seq_num = 0
        self.flag_change_ego_pose = True

    # 首次路径规划（全局路径规划）
    def global_path_planning(self):
        datas = self.agent_get_datas_from_carla.get_datas(
            datas_vehicle=True, datas_parking_plot=True,
            datas_bev=True, datas_aim=True, datas_path=True,
            is_global_path=True
        )
        return datas['aim'], datas['path'], datas['parking_plot']

    # 检测是否正常运行
    def check_running(self):
        # 检测碰撞
        if not self.agent_carla.check_no_collision():
            self.logger.debug('Stopped by Collision')
            return False

        # 检测是否出界
        # transform_ego = self.agent_carla.actors_dict['vehicle']['ego']['actor'].get_transform()
        # bounding_box_ego = self.agent_carla.actors_dict['vehicle']['ego']['actor'].bounding_box
        # bound_vertices_ego = bounding_box_ego.get_world_vertices(transform_ego)
        # bev_map_box = self.agent_map.bev_map_info['box_show']
        # vertice_names = [
        #     'rear left down', 'rear left up', 'rear right down', 'rear right up',
        #     'front left down', 'front left up', 'front right down', 'front right up'
        # ]
        # for bound_vertice, vertice_name in zip(bound_vertices_ego, vertice_names):
        #     if not bev_map_box.contains(bound_vertice, carla.Transform()):
        #         self.logger.debug(f"Stopped by '{vertice_name}' Out of BEV")
        #         return False
        # if not self.agent_carla.check_vehicle_in_range('ego'):
        #     self.logger.debug(f"Stopped by Ego Out of Map")
        #     return False
        
        # 检测更改车辆位置是否正常
        if not self.flag_change_ego_pose:
            self.logger.debug(f"Stopped while Changing Ego Pose")
            return False

        # 检测是否到循迹的终点
        if self.seq_num == len(self.datas['global_path']['path_points_rear']):
            self.logger.debug(f"Stopped by Finishing Following Path")
            return False
        return True

    # 更改自车的位置
    def change_ego_pose(self, mode):
        if mode == 'random':
            return self.change_ego_VehicleControl_random()
        elif mode == 'follow_path':
            self.flag_change_ego_pose = self.change_ego_Follow_Path()
        elif mode == 'place_directly':
            self.flag_change_ego_pose = self.change_ego_Place_Directly()
        else:
            assert False, 'Mode Error!'

    # 随即控制指令，以改变车辆位置
    def change_ego_VehicleControl_random(self):
        vehicle_ego = self.agent_carla.actors_dict['vehicle']['ego']
        vehicle_ego_actor = vehicle_ego['actor']

        # 通过随机和一阶低通滤波来控制车辆的节气门开度,方向盘转角和制动
        gamma = 0.6
        vc_last = vehicle_ego_actor.get_control()
        throttle_last, steer_last, brake_last = vc_last.throttle, vc_last.steer, vc_last.brake
        if random.random() <= 0.8:
            throttle = gamma * throttle_last + (1 - gamma) * random.uniform(0.5, 1.0)
            steer = gamma * steer_last + (1 - gamma) * random.uniform(-1.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            steer = gamma * steer_last + (1 - gamma) * random.uniform(-1.0, 1.0)
            brake = gamma * brake_last + (1 - gamma) * random.uniform(0.5, 1.0)
        vc = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        vehicle_ego_actor.apply_control(vc)
        return True

    # 通过循迹的方式改变车辆位置
    def change_ego_Follow_Path(self):
        # 计算目标点的坐标
        aim_point = self.datas['global_path']['path_points_rear'][self.seq_num]
        aim_xyz = aim_point[2:4].tolist() + [0]
        self.seq_num += 1

        return self.agent_carla.call_ego_to_location(aim_xyz, time_limit=10, dis_limit=2.0 * self.jump_dis)

    # 通过直接放置的方式改变车辆位置
    def change_ego_Place_Directly(self):
        # 计算目标点的坐标
        aim_point = self.datas['global_path']['path_points_center'][self.seq_num].tolist()
        aim_xyzPYR = aim_point[2:4] + [0.5] + [0.0] + [math.degrees(aim_point[4])] + [0.0]
        self.seq_num += 1

        # 获取车辆的actor
        actor = self.agent_carla.actors_dict['vehicle']['ego']['actor']

        # 直接放置
        self.agent_carla.set_vehicle_from_xyzPYR(actor, aim_xyzPYR)

        # 返回碰撞检测
        return self.agent_carla.actors_dict['collision']['ego']['event'] is None

    # 采集主要的原始数据
    def collect_origin_data(self, tbar_current, mode_change_ego_pose):
        def collect():
            datas = self.agent_get_datas_from_carla.get_datas(
                datas_stamp=True, datas_camera=True, datas_vehicle=True,
            )
            self.datas['stamp'].append(datas['stamp'])
            self.datas['camera'].append(datas['camera'])
            self.datas['vehicle'].append(datas['vehicle'])

        # for _ in range(5):
        while self.check_running():
            # 采集数据
            self.change_ego_pose(mode_change_ego_pose)
            self.agent_carla.tick()
            if self.flag_change_ego_pose:
                collect()
                # 更新进度条
                tbar_current.update(1)

    # 计算伴生数据，包括bev和path
    def generate_associated_data(self):
        for i, datas_vehicle in tqdm(
            enumerate(self.datas['vehicle']), 
            total=len(self.datas['stamp']), desc='Associated Data', unit='step', leave=False
        ):
            if i == 0:
                datas = self.agent_get_datas_from_carla.get_datas(
                    datas_bev=True, datas_aim=True, parking_plot=self.datas['parking_plot'],
                    vehicle=datas_vehicle
                )
                self.datas['bev'].append(datas['bev'])
                self.datas['aim'].append(datas['aim'])
                path_ = self.datas['global_path'].copy()
                path_['type'] = 'local'
                self.datas['path'].append(path_)
            else:
                datas = self.agent_get_datas_from_carla.get_datas(
                    datas_bev=True, datas_aim=True, datas_path=True,
                    vehicle=datas_vehicle, parking_plot=self.datas['parking_plot'], is_global_path=False
                )
                self.datas['bev'].append(datas['bev'])
                self.datas['aim'].append(datas['aim'])
                self.datas['path'].append(datas['path'])
    
    # 创建用于存储数据的文件夹
    def create_folder(self, folder_name=None, path_datas_folder=None):
        if folder_name is None:
            folder_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if path_datas_folder is None:
            path_datas_folder = os.path.join(self.path_datas, folder_name)
        os.mkdir(path_datas_folder)
        self.num_folder += 1
        self.datas_folders.append(path_datas_folder)
        return folder_name, path_datas_folder
    
    # 数据转换
    def transform_origin_datas(self, path_datas_folder, setting_id):
        # 定义一些函数，用于处理不同的数据类型
        def for_stamp(inp, stamp_start):
            return self.agent_get_datas_from_carla.trans_for_stamp(inp, stamp_start)
        def for_parking_plot(inp):
            return self.agent_get_datas_from_carla.trans_for_parking_plot(inp)
        def for_datas_camera(inp, stamp):
            return self.agent_get_datas_from_carla.trans_for_datas_camera(inp, stamp, path_datas_folder)
        def for_datas_vehicle(inp):
            return self.agent_get_datas_from_carla.trans_for_datas_vehicle(inp)
        def for_datas_bev(inp):
            return self.agent_get_datas_from_carla.trans_for_datas_bev(inp)
        def for_datas_aim(inp):
            return self.agent_get_datas_from_carla.trans_for_datas_aim(inp)
        def for_datas_path(inp):
            return self.agent_get_datas_from_carla.trans_for_datas_path(inp)

        # 使用一个dict来包含所有的数据，主要包括三部分：全局路径、车位信息、时序信息
        datas = {
            'global_path': for_datas_path(self.datas['global_path']),
            'global_aim': for_datas_aim(self.datas['global_aim']),
            'parking_plot': for_parking_plot(self.datas['parking_plot']),
            'setting_id': setting_id,
            'sequence': []
        }

        for datas_stamp, datas_camera, datas_vehicle, datas_bev, datas_aim, datas_path in tqdm(
            zip(self.datas['stamp'], self.datas['camera'], self.datas['vehicle'], self.datas['bev'], self.datas['aim'], self.datas['path']),
            total=len(self.datas['stamp']),
            desc='Datas_list', unit='step', leave=False
        ):
            data_dict = {}

            # 将datas_stamp写入data_dict
            data_dict['stamp'] = for_stamp(datas_stamp, self.datas['stamp'][0])

            # 将datas_camera写入data_dict,并保存图片
            data_dict['camera'] = for_datas_camera(datas_camera, datas_stamp)
            
            # 将datas_vehicle写入data_dict
            data_dict['vehicle'] = for_datas_vehicle(datas_vehicle)

            # 将datas_bev写入data_dict,并保存图片
            data_dict['bev'] = for_datas_bev(datas_bev)

            # 将datas_aim写入data_dict
            data_dict['aim'] = for_datas_aim(datas_aim)

            # 将datas_path写入data_dict
            data_dict['path'] = for_datas_path(datas_path)
            
            datas['sequence'].append(data_dict)

        return datas

    # 将采集的数据进行可视化展示
    def show_save_datas(self, path_datas_folder, flag_show, flag_save):
        datas_global_path = self.datas['global_path']
        datas_parking_plot = self.datas['parking_plot']

        # 按时序渲染
        frames_path = []
        for i in tqdm(
            range(len(self.datas['stamp'])),
            desc='Fig', unit='step', leave=False
        ):
            self.agent_plot.render_datas(
                self.datas['camera'][i], 
                self.datas['bev'][i], 
                self.datas['aim'][i], 
                self.datas['vehicle'][i], 
                self.datas['path'][i], 
                datas_parking_plot
            )
            if flag_show:
                plt.pause(1 / self.fps)
            if flag_save:
                path_image = os.path.join(path_datas_folder, f'Datas_{i}.png')
                plt.savefig(path_image)
                frames_path.append(path_image)
        
        # 将数据保存成gif文件
        if flag_save:
            self.agent_plot.save_gif(path_datas_folder, frames_path=frames_path)
