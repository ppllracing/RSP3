import math
import os
import sys
import random
import carla
import time
import numpy as np

from .for_base import AgentBase
from .for_map import AgentMap
from .for_parking_plot import AgentParkingPlot
from ..tools.config import Configuration, camera_lock
from ..tools.util import (
    cal_trans, get_camera_intrinsic, get_camera_extrinsic, 
    image2np, cal_rear_axle_location, check_0_to_1, FPSCountroller
)

path_folder_parking_learning_A_star = os.path.join(*(['/'] + os.path.dirname(__file__).split('/')[1:-3] + ['CARLA_PythonAPIs']))

# 调用“CARLA_PythonAPIs”下的算法
sys.path.append(path_folder_parking_learning_A_star)
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.controller import VehiclePIDController

class WayPoint:
    def __init__(self, transform):
        self.transform = transform


class AgentCarla(AgentBase):
    def __init__(self, cfg: Configuration, carla_client=None, *args, **kwargs):
        super().__init__(cfg)
        self.carla_client = carla_client or self.cfg.carla_client
        self.dtype = cfg.dtype_carla
        self.agent_map: AgentMap = kwargs.get('agent_map', None)
        self.agent_parking_plot: AgentParkingPlot = kwargs.get('agent_parking_plot', None)

        # actors的基本参数
        self.cameras = cfg.cameras
        self.ego = cfg.ego
        
        # 数据存储空间
        self.actors_dict = {
            'camera': {}, 
            'vehicle': {},
            'collision': {}
        }

        self.client = None
        self.world = None
        self.world_spectator = None
        self.bp_library = None
        self.spawn_points = None
        self.map_lib = None
        self.collision_bp = None
        self.world_snapshot = None

        # 初始化
        self.init_all()

    # 全部初始化
    def init_all(self):
        if self.agent_parking_plot is None:
            self.agent_parking_plot = AgentParkingPlot(self.cfg)
            self.logger.info('Finish to Initialize AgentParkingPlot')

        if self.agent_map is None:
            self.agent_map = AgentMap(self.cfg)
            self.agent_map.init_from_seleted_parking_plot(self.agent_parking_plot)
            self.logger.info('Finish to Initialize AgentMap')

        self.init_carla()
        while True:
            self.world.tick()
            flag = True
            for _, camera in self.actors_dict['camera'].items():
                if camera['image'] is None:
                    flag = False
                    break
            if flag:
                break
            self.logger.info('Waiting for camera ...')
        self.logger.info('Finish to Initialize CARLA')

        super().init_all()

    # 固定频率使用
    def tick(self):
        # 世界刷新
        self.world.tick()
        self.world_snapshot = self.world.get_snapshot()

        # 在carla中可视化一些基础设定
        # self.show_custom_points()
        # pass

        # # 视角定格在ego上
        # ego_location = self.actors_dict['vehicle']['ego']['actor'].get_location()
        # self.world_spectator.set_transform(cal_trans([ego_location.x, ego_location.y, 10.0] + self.cameras[0]['xyzPYR'][3:]))

    # 初始化carla
    def init_carla(self):
        # 初始化
        self.client = carla.Client(self.carla_client['ip'], self.carla_client['port'])
        self.client.set_timeout(10.0)
        self.client.load_world(self.carla_client['map'])

        # 获取carla的各种东西
        self.world = self.client.get_world()
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)  # 载入地图时不载入停车的车辆
        self.world_spectator = self.world.get_spectator()
        self.bp_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.map_lib = self.world.get_map()
        self.collision_bp = self.bp_library.find('sensor.other.collision')

        # 生成相机和车辆
        type_list, role_list, paras_list, xyzPYR_list = self.cal_for_actor()
        for type, role, paras, xyzPYR in zip(type_list, role_list, paras_list, xyzPYR_list):
            self.actors_dict[type].update({role: self.create_actor(type, role, paras, xyzPYR)})
        
        # 调整carla中的视角
        self.world_spectator.set_transform(cal_trans(self.cameras['npc']['xyzPYR']))

        self.tick()

    def listen_camera_rgb(self, camera_dict, image):
        _image = image2np(image)
        with camera_lock:
            camera_dict['image'] = _image

    def listen_camera_depth(self, camera_dict, image):
        _image = image2np(image)
        with camera_lock:
            camera_dict['image'] = _image

    def listen_collision(self, collision_dict, event):
        collision_dict['event'] = event
    
    # 计算用以生成actor的参数
    def cal_for_actor(self):
        type_list = []
        role_list = []
        paras_list = []
        xyzPYR_list = []

        # ego
        type_list.append('vehicle')
        role_list.append('ego')
        paras_list.append([
            self.ego['autopilot'],
            self.ego['speed']
        ])
        xyzPYR_list.append(self.ego['xyzPYR'])
        
        # collision
        type_list.append('collision')
        role_list.append('ego')
        paras_list.append([
            'ego'
        ])
        xyzPYR_list.append([0, 0, 0, 0, 0, 0])

        # cameras
        for k, v in self.cameras.items():
            type_list.append('camera')
            role_list.append(k)
            paras_list.append([v['h'], v['w'], v['fov']])
            if 'obu' in k:
                xyzPYR = v['xyzPYR_relative']
            else:
                xyzPYR = self.agent_parking_plot.transform_xyzPYR_from_pp(v['xyzPYR_relative'])
            xyzPYR_list.append(xyzPYR)
            self.cameras[k]['xyzPYR'] = xyzPYR

        return type_list, role_list, paras_list, xyzPYR_list

    # 创建actor
    def create_actor(self, type: str, role: str, paras: list, xyzPYR=None, location=None, rotation=None, type_id=None):
        # 计算坐标
        trans = cal_trans(xyzPYR, location, rotation)

        # 区分类型
        if type == 'camera':
            h, w, fov = paras
            actor_dict = {
                'actor': None,
                'h_w_fov': [h, w, fov],
                'image': None,
                'intrinsic': None,
                'extrinsic': None
            }
            is_depth = True if 'depth' in role else False

            # 设定参数
            if is_depth:
                _bp = self.bp_library.find('sensor.camera.depth')
            else:
                _bp = self.bp_library.find('sensor.camera.rgb')
            _bp.set_attribute("image_size_x", f"{w}")
            _bp.set_attribute("image_size_y", f"{h}")
            _bp.set_attribute("fov", f"{fov}")

            # 实例化
            if 'obu' in role:
                ego = self.actors_dict['vehicle']['ego']['actor']
                actor = self.world.spawn_actor(_bp, trans, attach_to=ego)
                intrinsic = get_camera_intrinsic(h, w, fov).astype(self.dtype)
                extrinsic = get_camera_extrinsic(trans).astype(self.dtype)
            else:
                actor = self.world.spawn_actor(_bp, trans)
                intrinsic = get_camera_intrinsic(h, w, fov).astype(self.dtype)
                extrinsic = get_camera_extrinsic(
                    cal_trans(location=trans.location - carla.Location(*self.agent_map.bev_center), rotation=trans.rotation)
                ).astype(self.dtype)
            
            if is_depth:
                actor.listen(lambda image: self.listen_camera_depth(actor_dict, image))
            else:
                actor.listen(lambda image: self.listen_camera_rgb(actor_dict, image))
            self.world.tick()

            # 记录数据
            actor_dict['actor'] = actor
            actor_dict['intrinsic'] = intrinsic
            actor_dict['extrinsic'] = extrinsic
        elif type == 'vehicle':
            actor_dict = {
                'init_xyzPYR': xyzPYR,
                'actor': None
            }

            if type_id is None:
                # 设定参数并实例化
                if role == 'ego':
                    _bp = self.bp_library.filter('model3')[0]
                    _bp.set_attribute('role_name', 'hero')
                else:
                    vehicles_type = [
                        'vehicle.audi.*', 'vehicle.tesla.*', 'vehicle.citroen.*', 
                        'vehicle.bmw.*', 'vehicle.mercedes.*', 'vehicle.nissan.*', 
                        'vehicle.seat.*', 'vehicle.toyota.*', 'vehicle.volkswagen.*'
                    ]
                    vehicles_bp = [v for t in vehicles_type for v in self.bp_library.filter(t)]
                    _bp = random.choice(vehicles_bp)
            else:
                _bp = self.bp_library.find(type_id)
            
            # 实例化
            actor = self.world.spawn_actor(_bp, trans)
            actor.set_autopilot(paras[0])
            self.world.tick()
            self.stop_vehicle(actor)
            self.wait_until_actor_stable(actor)

            if role == 'ego':
                actor_dict['target_speed'] = paras[1]
                # 使用Basicgent进行封装
                actor_dict['basic_agent'] = BasicAgent(actor, paras[1])
                # 设定PID控制器
                actor_dict['pid_controller_forward'] = VehiclePIDController(
                    actor, **self.ego['pid_controller_args']['forward'],
                    max_steering=1.0
                )
                actor_dict['pid_controller_backward'] = VehiclePIDController(
                    actor, **self.ego['pid_controller_args']['backward'],
                    max_steering=1.0
                )
                # 设定车辆运动参数
                actor_dict['base_params'] = self.ego['base_params']
            elif role.startswith('obs'):
                # 保存当前车辆的box
                box = actor.bounding_box
                box_show = actor.bounding_box
                box_show.location = actor.get_transform().location

                self.agent_parking_plot.pp_vehicle_infos.append({
                    'id': paras[1],
                    'role': role,
                    'location': actor.get_transform().location,
                    'rotation': actor.get_transform().rotation,
                    'transform': actor.get_transform(),
                    'box': box,
                    'box_show': box_show
                })

                self.agent_parking_plot.pp_infos[paras[1]]['usable'] = False

            # 记录数据
            actor_dict['actor'] = actor
            actor_dict['type_id'] = actor.type_id
        elif type == 'collision':
            aim_vehicle = paras[0]
            actor_dict = {
                'aim_vehicle': aim_vehicle,
                'actor': None,
                'event': None
            }

            # 设定参数
            _bp = self.bp_library.find('sensor.other.collision')

            # 实例化
            actor = self.world.spawn_actor(_bp, trans, self.actors_dict['vehicle'][aim_vehicle]['actor'])
            actor.listen(lambda event: self.listen_collision(actor_dict, event))
            self.world.tick()

            # 记录数据
            actor_dict['actor'] = actor
        else: 
            assert False, f'{type} is not support in my codes!'

        return actor_dict

    def show_waypoint(self, wp_list):
        for wp in wp_list:
            self.world.debug.draw_string(
                location=wp.transform.location,
                text='x'
            )

    def show_custom_points(self):
        # 展示BEV地图的边界
        self.world.debug.draw_box(
            box=self.agent_map.bev_map_info['box_show'],
            rotation=self.agent_map.bev_map_info['rotation'],
            thickness=0.05,
            # life_time=2.0
        )

        # 展示BEV地图
        for i in range(0, self.agent_map.bev_dimension[0], self.agent_map.bev_point_infos.shape[0] // 16):
            for j in range(0, self.agent_map.bev_dimension[1], self.agent_map.bev_point_infos.shape[1] // 16):
                # for k in range(self.agent_map.bev_dimension[2]):
                k = int(self.agent_map.bev_dimension[2] / 2)
                bev_point_info = self.agent_map.bev_point_infos[i, j, k]
                self.world.debug.draw_box(
                    box=bev_point_info['box_show'], 
                    rotation=bev_point_info['rotation'],
                    thickness=0.05,
                    life_time=2.0
                )

        # 展示泊位
        for pp_info in self.agent_parking_plot.pp_infos:
            self.world.debug.draw_box(
                box=pp_info['box_show'],
                rotation=pp_info['rotation'],
                thickness=0.05,
                color=carla.Color(r=0, g=255, b=0, a=255) if pp_info['usable'] else carla.Color(r=255, g=0, b=0, a=255),
                # life_time=2.0
            )

        # 展示选择的车位
        pp_info = self.agent_parking_plot.get_pp_info_from_id(self.agent_parking_plot.current_pp_id)
        self.world.debug.draw_box(
            box=pp_info['box_show'],
            rotation=pp_info['rotation'],
            thickness=0.05,
            color=carla.Color(r=0, g=0, b=0, a=255),
            life_time=2.0
        )

        # 展示泊位内已有的车辆
        for pp_vehicle in self.agent_parking_plot.pp_vehicle_infos:
            self.world.debug.draw_box(
                box=pp_vehicle['box_show'],
                rotation=pp_vehicle['rotation'],
                thickness=0.05,
                life_time=2.0
            )

    # 展示bev中某一层的结果
    def show_bev_layer(self, bev_point_infos, map_bev_layer):
        infos_layer = bev_point_infos[:, :, 5]  # 取某一层进行展示
        for i in range(map_bev_layer.shape[0]):
            for j in range(map_bev_layer.shape[1]):
                info = infos_layer[i, j]
                if map_bev_layer[i, j] == 1:
                    thickness = 0.1
                else:
                    continue  # 不画其他的
                    # thickness = 0.05  # 其他的细一些
                self.world.debug.draw_box(
                    box=info['box_show'], 
                    rotation=info['rotation'],
                    thickness=thickness
                )

    # 绘制规划的结果
    def show_planning_result(self, path_points_xyt):
        for xyt in path_points_xyt:
            self.world.debug.draw_string(
                location=carla.Location(x=xyt[2], y=xyt[3], z=0.0),
                text='o',
                life_time=100.0
            )

    # 控制自车去往哪里
    def call_ego_to_location(self, location, time_limit=5.0, dis_limit=1.0):
        if isinstance(location, list):
            x, y, z = location
            aim_location = carla.Location(x=x, y=y, z=z)
        elif isinstance(location, carla.Location):
            aim_location = location
        else:
            assert False, f'location should be list or carla.Location, but get {type(location)}'

        self.world.debug.draw_string(
            location=aim_location,
            text='x',
            life_time=time_limit
        )

        # 控制
        return self.control_by_VehiclePIDController(aim_location, time_limit, dis_limit)

    def stop_vehicle(self, actor):
        # 立马刹停
        actor.apply_control(
            carla.VehicleControl(
                throttle=0,
                steer=0,
                brake=1.0
            )
        )
        self.tick()

    def control_by_VehiclePIDController(self, aim_location, time_limit, dis_limit):
        vehicle = self.actors_dict['vehicle']['ego']['actor']
        target_speed = self.actors_dict['vehicle']['ego']['target_speed']

        # 计算目标点和车辆前进轴的夹角,当夹角的绝对值在[0, pi/2)之间时,说明目标点车辆正前方,否则说明车辆后方
        rear_axle_location = cal_rear_axle_location(
            vehicle.get_transform(), 
            self.actors_dict['vehicle']['ego']['base_params']['wheelbase']
        )
        xyt_0 = [rear_axle_location.x, rear_axle_location.y, math.radians(vehicle.get_transform().rotation.yaw)]
        xyt_1 = [aim_location.x, aim_location.y, 0.0]
        is_forward, _ = check_0_to_1(xyt_0, xyt_1)

        self.stop_vehicle(vehicle)

        dis = lambda delta_l: (delta_l.x ** 2 + delta_l.y ** 2 + delta_l.z ** 2) ** 0.5
        time_0= time.time()
        dis_0 = dis(
            cal_rear_axle_location(
                vehicle.get_transform(), 
                self.actors_dict['vehicle']['ego']['base_params']['wheelbase']
            ) - aim_location
        )
        self.logger.debug(f'Start to track. Distance: {dis_0}m/{dis_limit}m')
        arrive_dis = max((self.cfg.map_bev['resolution'][0] ** 2 + self.cfg.map_bev['resolution'][1] ** 2) ** 0.5, 0.5)
        flag_in_arrive_range = None
        with FPSCountroller(self.cfg.fps) as fps_controller:
            while True:
                # 获取当前车辆后轴中心
                rear_axle_location = cal_rear_axle_location(
                    vehicle.get_transform(), 
                    self.actors_dict['vehicle']['ego']['base_params']['wheelbase']
                )
                dis_axle_to_aim = dis(rear_axle_location - aim_location)
                time_now = time.time()

                # 判断是否跟踪失败或者到达目的地
                if time_now - time_0 > time_limit:
                    self.stop_vehicle(vehicle)
                    self.logger.debug(f'Stopped by out of time limit {time_now - time_0}s/{time_limit}s')
                    return False
                elif dis_axle_to_aim > dis_limit:
                    self.stop_vehicle(vehicle)
                    self.logger.debug(f'Stopped by out of distance limit {dis_axle_to_aim}m/{dis_limit}m')
                    return False
                elif self.actors_dict['collision']['ego']['event'] is not None:
                    self.stop_vehicle(vehicle)
                    self.logger.debug(f'Stopped by collision')
                    return False
                elif dis_axle_to_aim < arrive_dis:
                    self.logger.debug(f'Change flag_in_arrive_range: {flag_in_arrive_range}m -> {dis_axle_to_aim}m')
                    # 进入到达范围
                    if flag_in_arrive_range is None:
                        # 第一次进入到达范围
                        flag_in_arrive_range = dis_axle_to_aim
                    else:
                        if dis_axle_to_aim > flag_in_arrive_range or dis_axle_to_aim < 0.1:
                            # 不能再减小了，或者已经小于0.1m，直接判定到达目的地
                            self.stop_vehicle(vehicle)
                            self.logger.debug(f'Arrived at destination. Duration: {time_now - time_0}s, Distance: {dis_axle_to_aim}m')
                            return True
                        else:
                            # 继续减小到达范围
                            flag_in_arrive_range = dis_axle_to_aim

                # 循迹尚未完成，继续
                aim_waypoint = WayPoint(carla.Transform(aim_location, carla.Rotation()))
                if is_forward:
                    pid_controller = self.actors_dict['vehicle']['ego']['pid_controller_forward']
                else:
                    pid_controller = self.actors_dict['vehicle']['ego']['pid_controller_backward']
                control = pid_controller.run_step(target_speed, aim_waypoint)
                control.reverse = not is_forward
                vehicle.apply_control(control)

                self.logger.debug(f'Distance: {dis_axle_to_aim:.4f}m/{arrive_dis}m')
                self.tick()
                fps_controller.tick()

    def wait_until_actor_stable(self, actor, max_wait_time=3.0, check_interval=0.05, speed_threshold=1e-6):
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            vel = actor.get_velocity()
            speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
            if speed < speed_threshold:
                break
            time.sleep(check_interval)

    # 设定车辆位置
    def set_vehicle_from_xyzPYR(self, actor, xyzPYR):
        trans = cal_trans(xyzPYR)
        self.stop_vehicle(actor)
        actor.set_simulate_physics(False)
        self.world.tick()
        actor.set_transform(trans)
        self.world.tick()
        actor.set_simulate_physics(True)
        self.world.tick()
        self.stop_vehicle(actor)
        self.wait_until_actor_stable(actor)
        # if self.actors_dict['collision']['ego']['event'] is not None:
        #     # 当前放置点出现了碰撞
        #     return False
        # else:
        #     return True

    def reset_ego_beside_parking_plot(self, actor, pp_info):
        # 获取车位坐标
        xyzPYR_pp = pp_info['xyzPYR']

        # 设定x范围和y范围，并随机生成车辆位置
        xyzPYR = [
            xyzPYR_pp[0] + random.uniform(4.5, 6.5),
            xyzPYR_pp[1] + random.uniform(0.5, 8.0),
            xyzPYR_pp[2],
            0.0, 
            random.uniform(-100.0, -80.0), 
            0.0
        ]
        self.set_vehicle_from_xyzPYR(actor, xyzPYR)

    def reset_vehicle(self):
        role_delet_list = []
        for role, actor_dict in self.actors_dict['vehicle'].items():
            if role == 'ego':
                # 重置ego类型、坐标和车辆状态
                self.reset_ego_beside_parking_plot(actor_dict['actor'], self.agent_parking_plot.get_pp_info_from_id(self.agent_parking_plot.current_pp_id))
            elif role.startswith('obs'):
                # 销毁已经产生的障碍物车辆
                actor_dict['actor'].destroy()
                self.world.tick()
                role_delet_list.append(role)
            self.world.tick()
        
        # 从dict中移除已经删除的车辆
        for role in role_delet_list:
            self.actors_dict['vehicle'].pop(role)

    def reset_collision(self):
        self.actors_dict['collision']['ego']['event'] = None
    
    def reset_obstacle(self):
        # 随机选择车位的id
        ids = random.sample(
            [_info['id'] for _info in self.agent_parking_plot.pp_infos if _info['usable']], 
            random.randint(
                int(self.cfg.collect['num_obstacle_pp'] * 3 / 5), 
                self.cfg.collect['num_obstacle_pp']
            )
        )

        # 修改车位的属性并在该处生成车辆
        for _id in ids:
            if _id == self.agent_parking_plot.current_pp_id:
                # 避免在目标车位上生成车辆
                continue
            # 修改车位的属性
            self.agent_parking_plot.pp_infos[_id]['usable'] = False

            # 在该处生成车辆
            location = self.agent_parking_plot.pp_infos[_id]['location']
            rotation = self.agent_parking_plot.pp_infos[_id]['rotation']
            # 在location和rotation中加入一些扰动
            location = carla.Location(
                x=location.x + random.uniform(-0.3, 0.3),  # 前后 
                y=location.y + random.uniform(-0.3, 0.3),  # 左右 
                z=location.z
            )
            rotation = carla.Rotation(
                pitch=rotation.pitch,
                yaw=rotation.yaw + random.uniform(-5.0, 5.0),  # 转向
                roll=rotation.roll
            )
            try:
                self.actors_dict['vehicle'].update({
                    f'obs_{_id}': self.create_actor('vehicle', f'obs_{_id}', [False, _id], location=location, rotation=rotation)
                })
            except:
                pass

            self.tick()
    
    def reset(self):
        self.agent_parking_plot.reset_pp_infos_usable()
        self.agent_parking_plot.reset_pp_vehicle_infos()
        self.agent_parking_plot.reset_selecet_a_new_pp_id()
        self.reset_vehicle()
        self.reset_collision()
        self.reset_obstacle()
        self.tick()

    # 保存当前快照内所有车辆的信息
    def record_vehicles_from_snapshot(self, snapshot=None):
        snapshot = snapshot or self.world.get_snapshot()
        timestamp = snapshot.timestamp
        actor_states = {}
        for actor_snapshot in snapshot:
            actor_id = actor_snapshot.id
            actor = self.world.get_actor(actor_id)
            if actor.type_id.startswith("vehicle."):
                transform = actor_snapshot.get_transform()
                velocity = actor_snapshot.get_velocity()
                angular_velocity = actor_snapshot.get_angular_velocity()
                actor_states[actor_id] = {
                    "transform": transform,
                    "velocity": velocity,
                    "angular_velocity": angular_velocity
                }
        return {'timestamp': timestamp, 'actor_states': actor_states}
    
    # 恢复到某一快照的状态
    def restore_vehicles_to_snapshot(self, snapshot: dict):
        for actor_id, state in snapshot['actor_states'].items():
            actor = self.world.get_actor(actor_id)
            if actor and actor.type_id.startswith("vehicle."):
                actor.set_transform(state["transform"])
                actor.set_target_velocity(state["velocity"])
                actor.set_target_angular_velocity(state["angular_velocity"])
            else:
                raise ValueError(f"Actor {actor_id} not found in world.")
            self.world.tick()
    
    # 检查车辆是否在地图内
    def check_vehicle_in_range(self, role):
        actor = self.actors_dict['vehicle'][role]['actor']

        transform_vehicle = actor.get_transform()
        bounding_box_vehicle = actor.bounding_box
        bound_vertices_vehicle = bounding_box_vehicle.get_world_vertices(transform_vehicle)
        bev_map_box = self.agent_map.bev_map_info['box_show']
        vertice_names = [
            'rear left down', 'rear left up', 'rear right down', 'rear right up',
            'front left down', 'front left up', 'front right down', 'front right up'
        ]
        for bound_vertice, vertice_name in zip(bound_vertices_vehicle, vertice_names):
            if not bev_map_box.contains(bound_vertice, carla.Transform()):
                self.logger.debug(f"{vertice_name} Out of Range")
                return False
        return True
    
    # 检查是否发生碰撞
    def check_no_collision(self):
        return self.actors_dict['collision']['ego']['event'] is None