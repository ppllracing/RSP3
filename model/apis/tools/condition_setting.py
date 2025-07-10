import json
import copy
import numpy as np
from .util import wrap_to_2pi

Setting_Templete = {
    'weather': {  # 环境信息
        'cloudiness': float, 'precipitation': float, 'precipitation_deposits': float, 'wind_intensity': float,
        'sun_azimuth_angle': float, 'sun_altitude_angle': float, 'fog_density': float, 'fog_falloff': float, 'wetness': float
    },
    'actor': {  # carla中的Actor信息，主要包括相机和车辆
        'camera': {  # 相机信息
            'npc': {'xyzPYR': list, 'h': float, 'w': float, 'fov': float},
            'rsu_rgb': {'xyzPYR': list, 'h': float, 'w': float, 'fov': float},
            'rsu_depth': {'xyzPYR': list, 'h': float, 'w': float, 'fov': float}
        },
        'vehicle': {  # 车辆信息
            'ego': {'type_id': str, 'xyzPYR': list,'target_speed': float},
            'obstacle': [{'pp_id': int, 'type_id': str, 'xyzPYR': list}]
        }
    },
    'parking plot': {  # 停车位置信息
        'id': int, 'xyzPYR': list, 'xyzPYR_aim': list
    }
}

def check_xyzPYR_equal(xyzPYR_1, xyzPYR_2):
    # xyz的精度需求为1e-1
    if np.linalg.norm(np.array(xyzPYR_1[:3]) - np.array(xyzPYR_2[:3])) > 1e-1:
        return False
    # PYR需要转换为弧度之后，精度要求为1e-1
    if np.abs(wrap_to_2pi(np.deg2rad(np.array(xyzPYR_1[3:])) - np.deg2rad(np.array(xyzPYR_2[3:])))).max() > 1e-1:
        return False
    return True

def check_camera_equal(camera_info_1, camera_info_2):
    if not check_xyzPYR_equal(camera_info_1['xyzPYR'], camera_info_2['xyzPYR']):
        return False
    if camera_info_1['h'] != camera_info_2['h']:
        return False
    if camera_info_1['w'] != camera_info_2['w']:
        return False
    if camera_info_1['fov']!= camera_info_2['fov']:
        return False
    return True

def check_ego_equal(ego_info_1, ego_info_2):
    if ego_info_1['type_id'] != ego_info_2['type_id']:
        return False
    if not check_xyzPYR_equal(ego_info_1['xyzPYR'], ego_info_2['xyzPYR']):
        return False
    if ego_info_1['target_speed'] != ego_info_2['target_speed']:
        return False
    return True

def check_obstacle_equal(obstacle_info_1, obstacle_info_2):
    if obstacle_info_1['pp_id'] != obstacle_info_2['pp_id']:
        return False
    if obstacle_info_1['type_id'] != obstacle_info_2['type_id']:
        return False
    if not check_xyzPYR_equal(obstacle_info_1['xyzPYR'], obstacle_info_2['xyzPYR']):
        return False
    return True

def check_parking_plot_equal(parking_plot_info_1, parking_plot_info_2):
    if parking_plot_info_1['id'] != parking_plot_info_2['id']:
        return False
    if not check_xyzPYR_equal(parking_plot_info_1['xyzPYR'], parking_plot_info_2['xyzPYR']):
        return False
    if not check_xyzPYR_equal(parking_plot_info_1['xyzPYR_aim'], parking_plot_info_2['xyzPYR_aim']):
        return False
    return True

class ConditionSetting:
    def __init__(self, weather: dict=None, actors_dict: dict=None, parking_plot_info: dict=None, data: dict=None):
        self.data = copy.deepcopy(Setting_Templete)
        if data is None:
            # 设置天气
            self.data['weather'] = weather.copy()

            # 从actors_dict中获取相机数据
            cameras = actors_dict['camera']
            for camera_type, camera_info in cameras.items():
                if camera_type.startswith('obu'):
                    # 忽略obu相机，因为obu本身就是跟自车绑定的，不需要记录
                    continue
                actor = camera_info['actor']
                h_w_fov = camera_info['h_w_fov']

                # 计算xyzPYR
                xyzPYR = self.cal_xyzPYR_from_actor(actor)

                self.data['actor']['camera'][camera_type]['xyzPYR'] = xyzPYR
                self.data['actor']['camera'][camera_type]['h'] = h_w_fov[0]
                self.data['actor']['camera'][camera_type]['w'] = h_w_fov[1]
                self.data['actor']['camera'][camera_type]['fov'] = h_w_fov[2]

            vehicles = actors_dict['vehicle']
            # 从actors_dict中获取ego车辆数据
            vehicle_ego_info = vehicles['ego']
            actor = vehicle_ego_info['actor']
            type_id = vehicle_ego_info['type_id']
            xyzPYR = self.cal_xyzPYR_from_actor(actor)
            # xyzPYR[2] += 0.2  # 车辆高度修正
            target_speed = vehicle_ego_info['target_speed']
            self.data['actor']['vehicle']['ego']['type_id'] = type_id
            self.data['actor']['vehicle']['ego']['xyzPYR'] = xyzPYR
            self.data['actor']['vehicle']['ego']['target_speed'] = target_speed
            # 从actors_dict中获取其他车辆数据
            vehicle_obstacles_info = {k: vehicles[k] for k in vehicles if k.startswith('obs_')}
            for role, vehicle_info in vehicle_obstacles_info.items():
                pp_id = int(role.split('_')[1])
                actor = vehicle_info['actor']
                type_id = vehicle_info['type_id']
                xyzPYR = self.cal_xyzPYR_from_actor(actor)
                # xyzPYR[2] += 0.2  # 车辆高度修正
                self.data['actor']['vehicle']['obstacle'].append({'pp_id': pp_id, 'type_id': type_id, 'xyzPYR': xyzPYR})
            self.data['actor']['vehicle']['obstacle'].pop(0)  # 移除第一个元素，即模板
            
            # 从parking_plot_info中获取停车位置数据
            self.data['parking plot']['id'] = parking_plot_info['id']
            self.data['parking plot']['xyzPYR'] = parking_plot_info['xyzPYR']
            self.data['parking plot']['xyzPYR_aim'] = parking_plot_info['xyzPYR_aim']
        else:
            self.data = data.copy()

    def cal_xyzPYR_from_actor(self, actor):
        transform = actor.get_transform()
        location, rotation = transform.location, transform.rotation
        xyzPYR = [location.x, location.y, location.z, rotation.pitch, rotation.yaw, rotation.roll]
        assert 0 <= abs(rotation.roll) <= 90
        return xyzPYR

    def __eq__(self, value):
        if isinstance(value, ConditionSetting):
            # 检查天气
            weather_1 = self.data['weather']
            weather_2 = value.data['weather']
            for k in weather_1:
                if weather_1[k] != weather_2[k]:
                    # print('Weather', weather_1, weather_2)
                    return False
            # 检查相机
            for camera_type in self.data['actor']['camera']:
                camera_info_1 = self.data['actor']['camera'][camera_type]
                camera_info_2 = value.data['actor']['camera'][camera_type]
                if not check_camera_equal(camera_info_1, camera_info_2):
                    # print('Camera', camera_type, camera_info_1, camera_info_2)
                    return False
            # 检查车辆
            ego_info_1 = self.data['actor']['vehicle']['ego']
            ego_info_2 = value.data['actor']['vehicle']['ego']
            if not check_ego_equal(ego_info_1, ego_info_2):
                # print('Ego', ego_info_1, ego_info_2)
                return False
            obstacles_info_1 = self.data['actor']['vehicle']['obstacle']
            obstacles_info_2 = value.data['actor']['vehicle']['obstacle']
            if len(obstacles_info_1) != len(obstacles_info_2):
                # print('Obs Num', len(obstacles_info_1), len(obstacles_info_2))
                return False
            for obs_info_1, obs_info_2 in zip(obstacles_info_1, obstacles_info_2):
                if not check_obstacle_equal(obs_info_1, obs_info_2):
                    # print('Obs', obs_info_1, obs_info_2)
                    return False
            # 检查停车位置
            parking_plot_info_1 = self.data['parking plot']
            parking_plot_info_2 = value.data['parking plot']
            if not check_parking_plot_equal(parking_plot_info_1, parking_plot_info_2):
                # print('Parking Plot', parking_plot_info_1, parking_plot_info_2)
                return False
            return True
        else:
            return False
    
    def __str__(self):
        return json.dumps(self.data, indent=4)
