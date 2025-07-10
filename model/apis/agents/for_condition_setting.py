import json
import os
import carla
import time
import copy
from tqdm import tqdm

from .for_base import AgentBase
from .for_carla import AgentCarla
from .for_parking_plot import AgentParkingPlot
from ..tools.config import Configuration
from ..tools.util import read_datas_from_disk, save_datas_to_disk
from ..tools.condition_setting import ConditionSetting, Setting_Templete

class AgentConditionSetting(AgentBase):
    def __init__(self, cfg: Configuration, *args, **kwargs):
        super().__init__(cfg)
        self.settings = []
        self.init_all()

    def __len__(self):
        return len(self.settings)

    def init_all(self):
        # 尝试获取已有的settings
        try:
            # 直接读取pkl
            self.settings = read_datas_from_disk(self.cfg.path_condition_settings, 'condition_settings', 'pkl')

            # # 通过json进行构建，主要是为了在修改数据之后重新读取
            # settings = read_datas_from_disk(self.cfg.path_datas, 'condition_settings', 'json')
            # for s in settings.values():
            #     self.settings.append(ConditionSetting(data=s))
        except:
            self.settings = []

        super().init_all()

    def generate_settings(self, num, **agents):
        agent_carla: AgentCarla = agents['agent_carla']
        agent_parking_plot: AgentParkingPlot = agents['agent_parking_plot']
        tbar = tqdm(range(num), desc='Generating Condition Settings', leave=False)
        for i in tbar:
            j = 0
            while True:
                # 重置当前carla状态
                agent_carla.reset()

                # 获取当前的setting
                setting = self.get_current_setting(agent_carla, agent_parking_plot)

                if (not self.check_setting_exist(setting)) and agent_carla.check_vehicle_in_range('ego') and agent_carla.check_no_collision():
                    tbar.set_postfix()
                    self.add_setting(setting)
                    break
                else:
                    j += 1
                    tbar.set_postfix({'have_tried': j})
                    self.logger.debug(f'Setting [{setting}] already exists.')

    def check_setting_exist(self, setting):
        for s in self.settings:
            if s == setting:
                return True

    def add_setting(self, setting):
        self.settings.append(setting)

    def save_settings(self):
        save_datas_to_disk(self.settings, self.cfg.path_datas, 'condition_settings', 'pkl')
        save_datas_to_disk({f'setting_{i}': setting.data for i, setting in enumerate(self.settings)}, self.cfg.path_datas, 'condition_settings', 'json')

    def get_setting_by_id(self, id):
        return self.settings[id]

    def change_setting_by_id(self, id, setting):
        self.settings[id] = setting

    def get_current_setting(self, agent_carla: AgentCarla, agent_parking_plot: AgentParkingPlot):
        # 获取weather
        weather = agent_carla.world.get_weather()
        weather = {
            'cloudiness': weather.cloudiness,
            'precipitation': weather.precipitation,
            'precipitation_deposits': weather.precipitation_deposits,
            'wind_intensity': weather.wind_intensity,
            'sun_azimuth_angle': weather.sun_azimuth_angle,
            'sun_altitude_angle': weather.sun_altitude_angle,
            'fog_density': weather.fog_density,
            'fog_falloff': weather.fog_falloff,
            'wetness': weather.wetness
        }

        # 获取当前的各个actor的状态
        actors_dict = agent_carla.actors_dict

        # 获取当前目标车位
        parking_plot_info = agent_parking_plot.get_pp_info_from_id(agent_parking_plot.current_pp_id)

        # 判断将world_snapshot和parkin_plot_info合并转换为ConditionSetting类型
        setting = ConditionSetting(weather=weather, actors_dict=actors_dict, parking_plot_info=parking_plot_info)

        return setting

    def set_agents_from_setting(self, setting: ConditionSetting, check_equal=True, **agents):

        agent_carla: AgentCarla = agents['agent_carla']
        agent_parking_plot: AgentParkingPlot = agents['agent_parking_plot']

        # 相机传感器是不会变的，就暂时不重置了

        # 这两个和setting没啥关系
        agent_parking_plot.reset_pp_infos_usable()
        agent_parking_plot.reset_pp_vehicle_infos()

        # 重置天气
        weather = carla.WeatherParameters(**setting.data['weather'])
        agent_carla.world.set_weather(weather)

        # 重置目标车位
        pp_id = setting.data['parking plot']['id']
        agent_parking_plot.reset_selecet_a_new_pp_id(pp_id)

        # 调整自车的位置
        actor = agent_carla.actors_dict['vehicle']['ego']['actor']
        xyzPYR = setting.data['actor']['vehicle']['ego']['xyzPYR']
        agent_carla.set_vehicle_from_xyzPYR(actor, xyzPYR)

        # 重置对自车的碰撞检测
        agent_carla.reset_collision()

        # 删除已有的obstacle
        delet_list = []
        for k in agent_carla.actors_dict['vehicle']:
            if k.startswith('obs_'):
                agent_carla.actors_dict['vehicle'][k]['actor'].destroy()
                agent_carla.tick()
                delet_list.append(k)
        for k in delet_list:
            agent_carla.actors_dict['vehicle'].pop(k)

        # 根据setting重置obs
        for obs_info in setting.data['actor']['vehicle']['obstacle']:
            pp_id = obs_info['pp_id']
            type_id = obs_info['type_id']
            # 修改车位的属性
            agent_parking_plot.pp_infos[pp_id]['usable'] = False

            # 计算location和rotation
            location = carla.Location(*obs_info['xyzPYR'][:3])
            rotation = carla.Rotation(*obs_info['xyzPYR'][3:])
            actor_dict = None
            for i in range(20):
                try:
                    actor_dict = agent_carla.create_actor(
                        'vehicle', 
                        f'obs_{pp_id}', 
                        [False, pp_id], 
                        location=location, 
                        rotation=rotation, 
                        type_id=type_id
                    )
                    break
                except:
                    location.z += 0.05
            assert actor_dict is not None, 'Failed to create obstacle.'
            agent_carla.actors_dict['vehicle'].update({
                f'obs_{pp_id}': actor_dict
            })
        if check_equal:
            # 检查当前的condition和setting中描述的condition是否一致
            for _ in range(5):
                if setting == self.get_current_setting(agent_carla, agent_parking_plot):
                    break
                else:
                    time.sleep(0.05)
            assert setting == self.get_current_setting(agent_carla, agent_parking_plot), 'Current condition and setting are not consistent.'

    def analyse_settings(self):
        # 获取规整
        setting_merged = self.all_to_one_setting()

        # 输出障碍物车辆类型的分布
        obs_type_dict = {}
        for type_id in setting_merged['actor']['vehicle']['obstacle']['type_id']:
            if type_id in obs_type_dict:
                obs_type_dict[type_id] += 1
            else:
                obs_type_dict[type_id] = 1
        obs_type_list = [f'{k}: {v/sum(obs_type_dict.values())*100:.2f}% ({v})' for k, v in obs_type_dict.items()]
        self.logger.info(f'Obstacle Type Distribution: \n {json.dumps(obs_type_list, indent=4)}')

        # 输出障碍物车位的分布
        obs_pp_id_dict = {}
        for obs_pp_id in setting_merged['actor']['vehicle']['obstacle']['pp_id']:
            if obs_pp_id in obs_pp_id_dict:
                obs_pp_id_dict[obs_pp_id] += 1
            else:
                obs_pp_id_dict[obs_pp_id] = 1
        obs_pp_id_dict = {k: obs_pp_id_dict[k] for k in sorted(obs_pp_id_dict)}  # 排序
        obs_pp_id_list = [f'{k}: {v/sum(obs_pp_id_dict.values())*100:.2f}% ({v})' for k, v in obs_pp_id_dict.items()]
        self.logger.info(f'Obstacle Parking Plot Distribution: \n {json.dumps(obs_pp_id_list, indent=4)}')

        # 输出目标车位的分布
        pp_id_dict = {}
        for pp_id in setting_merged['parking plot']['id']:
            if pp_id in pp_id_dict:
                pp_id_dict[pp_id] += 1
            else:
                pp_id_dict[pp_id] = 1
        pp_id_dict = {k: pp_id_dict[k] for k in sorted(pp_id_dict)}  # 排序
        pp_id_list = [f'{k}: {v/sum(pp_id_dict.values())*100:.2f}% ({v})' for k, v in pp_id_dict.items()]
        self.logger.info(f'Parking Plot Distribution: \n {json.dumps(pp_id_list, indent=4)}')

    def all_to_one_setting(self):

        # 将所有setting合并为一个setting
        setting_merged = copy.deepcopy(Setting_Templete)
        for s in tqdm(self.settings, desc='Merging Settings', leave=False):
            data = s.data

            # weather
            weather = data['weather']
            for k in weather:
                if isinstance(setting_merged['weather'][k], type):
                    setting_merged['weather'][k] = []
                setting_merged['weather'][k].append(weather[k])
            
            # actor
            # 忽略相机，这玩意儿没必要分析
            vehicle = data['actor']['vehicle']
            ego = vehicle['ego']
            for k in ego:
                if isinstance(setting_merged['actor']['vehicle']['ego'][k], type):
                    setting_merged['actor']['vehicle']['ego'][k] = []
                setting_merged['actor']['vehicle']['ego'][k].append(ego[k])
            obstacle = vehicle['obstacle']
            if isinstance(setting_merged['actor']['vehicle']['obstacle'], list):
                    setting_merged['actor']['vehicle']['obstacle'] = {k: [] for k in obstacle[0]}
            for obs in obstacle:
                for k in obs:
                    setting_merged['actor']['vehicle']['obstacle'][k].append(obs[k])
            
            # parking plot
            parking_plot = data['parking plot']
            for k in parking_plot:
                if isinstance(setting_merged['parking plot'][k], type):
                    setting_merged['parking plot'][k] = []
                setting_merged['parking plot'][k].append(parking_plot[k])
        return setting_merged