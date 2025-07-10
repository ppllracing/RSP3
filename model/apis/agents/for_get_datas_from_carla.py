import itertools
import os
import math
import time
import carla
import numpy as np
import matplotlib.pyplot as plt

from ..tools.config import Configuration, camera_lock
from .for_base import AgentBase
from .for_carla import AgentCarla
from .for_parking_plot import AgentParkingPlot
from .for_planner import AgentPlanner
from .for_map import AgentMap
from ..tools.util import (
    select_bev_points, crop_image, cal_post_tran_rot, cal_rear_axle_location,
    get_bbox_vertices_ordered, cal_risk_degrees_for_path_points, timefunc, get_relative_matrix, move_transform_by_relative_matrix
)

class AgentGetDatasFromCarla(AgentBase):
    def __init__(self, cfg: Configuration, agents):
        super().__init__(cfg)
        self.dtype = cfg.dtype_model
        self.agent_carla: AgentCarla = agents['carla']
        self.agent_parking_plot: AgentParkingPlot = agents['parking_plot']
        self.agent_map: AgentMap = agents['map']
        self.agent_planner: AgentPlanner = agents['planner']

        self.init_all()
    
    def get_datas(
            self, 
            datas_stamp=False, datas_camera=False, datas_vehicle=False, 
            datas_parking_plot=False, datas_bev=False, datas_aim=False, datas_path=False, 
            **kwargs
        ):
        datas = {}
        if datas_stamp:
            datas.update({'stamp': self.datas_stamp_once()})
            kwargs.update(datas)
        if datas_camera:
            datas.update({'camera': self.datas_camera_once()})
            kwargs.update(datas)
        if datas_vehicle:
            datas.update({'vehicle': self.datas_vehicle_once()})
            kwargs.update(datas)
        if datas_parking_plot:
            datas.update({'parking_plot': self.datas_parking_plot_once()})
            kwargs.update(datas)
        if datas_aim:
            datas.update({'aim': self.datas_aim_once(kwargs['vehicle'], kwargs['parking_plot'])})
            kwargs.update(datas)
        if datas_bev:
            datas.update({'bev': self.datas_bev_once(kwargs['vehicle'], kwargs['parking_plot'], kwargs['aim'])})
            kwargs.update(datas)
        if datas_path:
            datas.update({'path': self.datas_path_once(kwargs['bev'], kwargs['aim'], kwargs['is_global_path'])})
            kwargs.update(datas)
        return datas

    # 记录时间戳
    def datas_stamp_once(self):
        datas_stamp = {
            'global': time.time(),
            'elapsed_seconds': self.agent_carla.world_snapshot.elapsed_seconds
        } 
        return datas_stamp

    # 记录相机数据
    def datas_camera_once(self):
        datas_camera = {}
        for role, role_data in self.agent_carla.actors_dict['camera'].items():
            with camera_lock:
                image_origin = role_data['image'].copy()
            image_crop, factors = crop_image(image_origin, self.cfg.collect['image_crop'], 'linspace')
            post_tran, post_rot = cal_post_tran_rot(factors, self.dtype)
            datas_camera.update({
                role: {
                    'h_w_fov': role_data['h_w_fov'].copy(),
                    'image': image_origin,
                    'image_crop': image_crop,
                    'post_tran': post_tran,
                    'post_rot': post_rot,
                    'intrinsic': role_data['intrinsic'].copy(),
                    'extrinsic': role_data['extrinsic'].copy()
                }
            })
        return datas_camera

    # 记录ego的数据
    def datas_vehicle_once(self):
        vehicle_ego = self.agent_carla.actors_dict['vehicle']['ego']
        vehicle_ego_actor = vehicle_ego['actor']

        # 获取定位
        transform = vehicle_ego_actor.get_transform()
        location, rotation = transform.location, transform.rotation
        rear_axle_location = cal_rear_axle_location(
            transform, 
            self.agent_carla.actors_dict['vehicle']['ego']['base_params']['wheelbase']
        )
        x, y, z = location.x, location.y, location.z
        x_r, y_r, z_r = rear_axle_location.x, rear_axle_location.y, rear_axle_location.z
        pitch, yaw, roll = rotation.pitch, rotation.yaw, rotation.roll
        bounding_box = vehicle_ego_actor.bounding_box
        vertices_up, vertices_down = get_bbox_vertices_ordered(bounding_box, transform)
        
        datas_vehicle = {
            'role': 'ego',
            'xyzPYR': [x, y, z, pitch, yaw, roll],
            'xyzPYR_rear': [x_r, y_r, z_r, pitch, yaw, roll],
            'transform': transform,
            'location': location,
            'rotation': rotation,
            'rear_axle_location': rear_axle_location,
            'bounding_box': bounding_box,
            'vertices_up': vertices_up,
            'vertices_down': vertices_down
        }
        return datas_vehicle

    # 记录所选泊位的数据
    def datas_parking_plot_once(self):

        # id直接来源于cfg
        selected_id = self.agent_parking_plot.current_pp_id

        pp_point_info = self.agent_parking_plot.get_pp_info_from_id(selected_id)

        # 提取车位中心点和泊车目标点
        xyzPYR = pp_point_info['xyzPYR']
        xyzPYR_aim = pp_point_info['xyzPYR_aim']

        # 提取角点
        vertices_up, vertices_down = get_bbox_vertices_ordered(
            pp_point_info['box'],
            pp_point_info['transform']
        )
        center_up = np.array(vertices_up).mean(axis=0).tolist()
        center_down = np.array(vertices_down).mean(axis=0).tolist()

        # # 将选择的车位进行展示
        # self.agent_carla.world.debug.draw_box(
        #     box=pp_point_info['box'],
        #     rotation=pp_point_info['rotation']
        # )

        datas_parking_plot = {
            'selected_id': selected_id,
            'xyzPYR': xyzPYR,
            'xyzPYR_aim': xyzPYR_aim,
            'vertices_up': vertices_up,
            'vertices_down': vertices_down,
            'center_up': center_up,
            'center_down': center_down
        }
        return datas_parking_plot

    # 记录bev的数据
    def datas_bev_once(self, datas_vehicle, datas_parking_plot, datas_aim):
        # 获取ego的信息
        ego_box = datas_vehicle['bounding_box']
        ego_transform = datas_vehicle['transform']
        ego_info = {
            'box': ego_box,
            'transform': ego_transform
        }

        # 获取目标车位的信息
        pp_choice_info = self.agent_parking_plot.pp_infos[datas_parking_plot['selected_id']]
        pp_vehicle_infos = self.agent_parking_plot.pp_vehicle_infos

        # 获取原始bev_map
        map_bev = self.agent_map.map_bev.copy()
        map_bev_obu = self.agent_map.map_bev.copy()

        # 获取每个bev_point_info和pp之间的相对变化，由于select_bev_points只用到了location，所以这里保存的是location
        bev_point_infos_obu = np.where(np.ones_like(self.agent_map.bev_point_infos), None, np.zeros_like(self.agent_map.bev_point_infos))
        _transform = carla.Transform(
            carla.Location(*self.agent_map.bev_center),
            self.agent_parking_plot.get_pp_info_from_id()['rotation']
        )
        for i in range(bev_point_infos_obu.shape[0]):
            for j in range(bev_point_infos_obu.shape[1]):
                for k in range(bev_point_infos_obu.shape[2]):
                    info = self.agent_map.bev_point_infos[i, j, k]
                    relative_transform = get_relative_matrix(_transform, info['transform'])
                    info_transform = move_transform_by_relative_matrix(ego_transform, relative_transform)
                    bev_point_infos_obu[i, j, k] = {
                        'location': info_transform.location,
                        'rotation': info_transform.rotation,
                        'transform': info_transform,
                        'box_show': carla.BoundingBox(
                            info_transform.location, 
                            carla.Vector3D(*(self.agent_map.bev_resolution / 2))
                        )
                    }
        
        # 分层处理
        def process_free_space(map_bev, map_bev_ego, bev_point_infos, pp_vehicle_infos):
            for _info in pp_vehicle_infos:
                map_bev -= select_bev_points(np.zeros_like(map_bev), bev_point_infos, _info)
            map_bev -= map_bev_ego  # 减去ego的位置
            map_bev = np.clip(map_bev, 0.0, 1.0, dtype=self.dtype)  # 限制范围
            return map_bev
        def process_obstacle(map_bev_frees_space, map_bev_ego):
            map_bev = 1.0 - (map_bev_frees_space + map_bev_ego)
            map_bev = np.clip(map_bev, 0.0, 1.0, dtype=self.dtype)  # 限制范围
            return map_bev

        # 获取ego对应的bev
        _map_bev_ego = select_bev_points(np.zeros_like(map_bev[0]), self.agent_map.bev_point_infos, ego_info)
        _map_bev_obu_ego = select_bev_points(np.zeros_like(map_bev_obu[0]), bev_point_infos_obu, ego_info)
        # self.agent_carla.show_bev_layer(bev_point_infos_obu, _map_bev_obu_ego)

        # 自由区域
        map_bev[0] = process_free_space(map_bev[0], _map_bev_ego, self.agent_map.bev_point_infos, pp_vehicle_infos)
        map_bev_obu[0] = process_free_space(map_bev_obu[0], _map_bev_obu_ego, bev_point_infos_obu, pp_vehicle_infos)

        # 障碍物
        map_bev[1] = process_obstacle(map_bev[0], _map_bev_ego)
        map_bev_obu[1] = process_obstacle(map_bev_obu[0], _map_bev_obu_ego)

        # 自车区域
        map_bev[2] = _map_bev_ego
        map_bev_obu[2] = _map_bev_obu_ego

        # 目标车位区域
        map_bev[3] = select_bev_points(np.zeros_like(map_bev[0]), self.agent_map.bev_point_infos, pp_choice_info)
        map_bev_obu[3] = select_bev_points(np.zeros_like(map_bev_obu[0]), bev_point_infos_obu, pp_choice_info)

        assert self.agent_map.check_map_bev(map_bev), 'map_bev is illegal'
        assert self.agent_map.check_map_bev(map_bev_obu), 'map_bev_obu is illegal'

        datas_bev = {
            'map_bev': map_bev,
            'map_bev_obu': map_bev_obu,
            'bev_channel_names': self.agent_map.bev_channel_name
        }
        return datas_bev

    # 记录目标车位的数据
    def datas_aim_once(self, datas_vehicle, datas_parking_plot):
        # 获取ego的信息
        ego_xyzPYR_rear = datas_vehicle['xyzPYR_rear']
        # 获取目标车位的信息
        pp_info = self.agent_parking_plot.pp_infos[datas_parking_plot['selected_id']]
        
        # 计算车辆的初始位置和末端位置
        start_xyt = [
            ego_xyzPYR_rear[0],
            ego_xyzPYR_rear[1],
            ego_xyzPYR_rear[4]
        ]
        end_xyt = [pp_info['xyzPYR_aim'][0], pp_info['xyzPYR_aim'][1], pp_info['xyzPYR_aim'][4]]

        # 进行坐标变换
        start_xyt, end_xyt = self.agent_planner.convert_xyt(start_xyt), self.agent_planner.convert_xyt(end_xyt)

        # 转换到map_bev的离散点中
        start_xyt = self.agent_planner.add_grid_id_to_path_points(start_xyt)[0, 2:].tolist()
        end_xyt = self.agent_planner.add_grid_id_to_path_points(end_xyt)[0, 2:].tolist()

        # cal_id = lambda xyt: [np.round(xyt[0] / self.cfg.map_bev['resolution'][0]), np.round(xyt[1] / self.cfg.map_bev['resolution'][1])]
        datas_aim = {
            'parking_plot_id': datas_parking_plot['selected_id'],
            'start_xyt': start_xyt,
            'end_xyt': end_xyt,
            'start_id': self.agent_planner.cal_id(start_xyt),
            'end_id': self.agent_planner.cal_id(end_xyt)
        }
        return datas_aim

    # 记录路径规划的数据
    def datas_path_once(self, datas_bev, datas_aim, is_global_path=True):

        map_bev = datas_bev['map_bev'].copy()
        start_xyt = datas_aim['start_xyt']
        end_xyt = datas_aim['end_xyt']

        # 路径规划
        if self.cfg.collect['plan_time_limit'] <= 0:
            self.logger.warning('No time limit for path planning, may take a long time')
            duration, path_points_info = timefunc(self.agent_planner.plan, map_bev, start_xyt, end_xyt)
            path_points_info.update({'duration': duration})
        else:
            path_points_info = self.agent_planner.plan_with_time_limit(
                map_bev, start_xyt, end_xyt,
                time_limit=self.cfg.collect['plan_time_limit'],
                raise_exception=False
            )

        # 计算风险值
        risk_degrees = cal_risk_degrees_for_path_points(
            path_points_info['center'], map_bev, self.cfg.map_bev['resolution'], radius=8.0, angle=160.0
        )

        # 计算启发图
        datas_init = self.agent_planner.planner.cal_datas_init(map_bev, start_xyt, end_xyt, self.cfg.map_bev)
        handle = self.agent_planner.planner.get_handle()
        heuristic_fig = handle.cal_heuristic_fig(
            datas_init=datas_init, final_dim=self.cfg.map_bev['final_dim'], flag_normalize=True
        )
        heuristic_fig = heuristic_fig.astype(self.dtype)

        datas_path = {
            'type': 'global' if is_global_path else 'local',
            'success': path_points_info['success'],
            'duration': path_points_info['duration'],
            'path_points_rear': path_points_info['rear'],
            'path_points_center': path_points_info['center'],
            'risk_degrees': risk_degrees,
            'heuristic_fig': heuristic_fig
        }
        return datas_path

    def trans_for_stamp(self, inp, stamp_start):
        oup = {
            'global': inp['global'],
            'duration': inp['global'] - stamp_start['global'],
            'elapsed_seconds': inp['elapsed_seconds']
        }
        return oup

    def trans_for_parking_plot(self, inp):
        oup = {
            'selected_id': inp['selected_id'] - 11,
            'xyzPYR': np.array(inp['xyzPYR'], dtype=self.dtype),
            'xyzPYR_aim': np.array(inp['xyzPYR_aim'], dtype=self.dtype),
            'center_up': np.array(inp['center_up'], dtype=self.dtype),
            'center_down': np.array(inp['center_down'], dtype=self.dtype),
            'vertices_up': np.array(inp['vertices_up'], dtype=self.dtype),
            'vertices_down': np.array(inp['vertices_down'], dtype=self.dtype)
        }
        return oup

    def trans_for_datas_camera(self, inp, stamp, path_datas_folder=None):
        oup = {}
        for role, role_data in inp.items():
            file_image = f'{stamp["global"]}_camera_{role}.png'
            file_image_crop = f'{stamp["global"]}_camera_{role}_crop.png'
            oup.update({
                role: {
                    'image': role_data['image'],
                    'image_crop': role_data['image_crop'],
                    'h_w_fov': np.array(role_data['h_w_fov'], dtype=self.dtype),
                    'post_tran': np.array(role_data['post_tran'], dtype=self.dtype),
                    'post_rot': np.array(role_data['post_rot'], dtype=self.dtype),
                    'intrinsic': np.array(role_data['intrinsic'], dtype=self.dtype),
                    'extrinsic': np.array(role_data['extrinsic'], dtype=self.dtype)
                }
            })
            if path_datas_folder:
                plt.imsave(os.path.join(path_datas_folder, file_image), role_data['image'].transpose(1, 2, 0))
                plt.imsave(os.path.join(path_datas_folder, file_image_crop), role_data['image_crop'].transpose(1, 2, 0))
        return oup

    def trans_for_datas_vehicle(self, inp):
        oup = {
            'xyzPYR': np.array(inp['xyzPYR'], dtype=self.dtype),
            'xyzPYR_rear': np.array(inp['xyzPYR_rear'], dtype=self.dtype),
            'vertices_up': np.array(inp['vertices_up'], dtype=self.dtype),
            'vertices_down': np.array(inp['vertices_down'], dtype=self.dtype)
        }
        return oup

    def trans_for_datas_bev(self, inp):
        oup = {
            'map_bev': np.array(inp['map_bev'], dtype=np.uint8),
            'map_bev_obu': np.array(inp['map_bev_obu'], dtype=np.uint8),
            'bev_channel_names': np.array(inp['bev_channel_names'], dtype=str),
        }
        return oup

    def trans_for_datas_aim(self, inp):
        oup = {
            'parking_plot_id': inp['parking_plot_id'],
            'start_xyt': np.array(inp['start_xyt'], dtype=self.dtype),
            'end_xyt': np.array(inp['end_xyt'], dtype=self.dtype),
            'start_id': np.array(inp['start_id'], dtype=self.dtype),
            'end_id': np.array(inp['end_id'], dtype=self.dtype)
        }
        return oup

    def trans_for_datas_path(self, inp):
        oup = {
            'type': inp['type'],
            'success': inp['success'],
            'duration': inp['duration'],
            'path_points_rear': np.array(inp['path_points_rear'], dtype=self.dtype),
            'path_points_center': np.array(inp['path_points_center'], dtype=self.dtype),
            'risk_degrees': np.array(inp['risk_degrees'], dtype=self.dtype),
            'heuristic_fig': np.array(inp['heuristic_fig'], dtype=self.dtype)
        }
        return oup